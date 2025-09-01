import pytest
import torch
import numpy as np
from types import SimpleNamespace

from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs, SystemConfig
from orb_models.forcefield.base import AtomGraphs
from ase import Atoms


class TestOrbPairEdgesForceWrapper:
    """Test the OrbPairEdgesForceWrapper class."""

    def test_init_requires_conservative_model(self, lammps_interface_classes):
        """Test that wrapper requires a conservative model."""
        # Create a mock model without grad_forces_name
        from unittest.mock import Mock
        mock_model = Mock()
        del mock_model.grad_forces_name  # Ensure it doesn't have the attribute

        OrbPairEdgesForceWrapper = lammps_interface_classes['OrbPairEdgesForceWrapper']
        with pytest.raises(ValueError, match="Model must be a ConservativeForcefieldRegressor"):
            OrbPairEdgesForceWrapper(mock_model)

    def test_init_with_conservative_model(self, conservative_regressor, lammps_interface_classes):
        """Test successful initialization with conservative model."""
        OrbPairEdgesForceWrapper = lammps_interface_classes['OrbPairEdgesForceWrapper']
        wrapper = OrbPairEdgesForceWrapper(conservative_regressor)
        assert wrapper.model == conservative_regressor

    def test_edge_force_computation(self, conservative_regressor, lammps_interface_classes):
        """Test that edge forces are computed correctly."""
        OrbPairEdgesForceWrapper = lammps_interface_classes['OrbPairEdgesForceWrapper']
        wrapper = OrbPairEdgesForceWrapper(conservative_regressor)

        # Create a simple test system
        atoms = Atoms("H2O", positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc(True)

        system_config = SystemConfig(radius=6.0, max_num_neighbors=20)
        batch = ase_atoms_to_atom_graphs(atoms, system_config=system_config)

        # Call the wrapper
        predictions = wrapper(batch)

        # Check that we get the expected outputs
        assert "energy" in predictions
        assert "edge_forces" in predictions
        assert conservative_regressor.energy_name in predictions

        # Check shapes
        edge_forces = predictions["edge_forces"]
        assert edge_forces.shape[0] == batch.n_edge[0]  # Number of edges
        assert edge_forces.shape[1] == 3  # Force components (x, y, z)

    def test_predict_method_consistency(self, conservative_regressor, lammps_interface_classes):
        """Test that predict method returns consistent results."""
        OrbPairEdgesForceWrapper = lammps_interface_classes['OrbPairEdgesForceWrapper']
        wrapper = OrbPairEdgesForceWrapper(conservative_regressor)

        # Create test batch
        atoms = Atoms("H2", positions=np.array([[0, 0, 0], [1, 0, 0]]))
        atoms.set_cell([5, 5, 5])
        atoms.set_pbc(True)

        system_config = SystemConfig(radius=6.0, max_num_neighbors=20)
        batch = ase_atoms_to_atom_graphs(atoms, system_config=system_config)

        # Call through __call__ (which handles the gradient setup correctly)
        predictions = wrapper(batch)

        # Results should contain required outputs
        assert "energy" in predictions
        assert "edge_forces" in predictions
        assert conservative_regressor.energy_name in predictions


class TestORBMLIAPUnified:
    """Test the ORB_MLIAPUnified class."""

    def test_init(self, conservative_regressor, mock_orb_mliap_unified):
        """Test initialization of ORB_MLIAPUnified."""
        interface = mock_orb_mliap_unified(conservative_regressor, radius=8.0, max_num_neighbors=30)

        assert interface.radius == 8.0
        assert interface.max_num_neighbors == 30
        assert interface.device == torch.device("cpu")
        assert interface.dtype == torch.float32
        assert hasattr(interface.model, 'model')  # Check it has the wrapped conservative model
        assert interface.model.model == conservative_regressor

    def test_prepare_batch(self, conservative_regressor, mock_orb_mliap_unified):
        """Test batch preparation from LAMMPS data."""
        interface = mock_orb_mliap_unified(conservative_regressor)

        # Mock LAMMPS data
        data = SimpleNamespace()
        data.x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)  # 3 atoms
        data.pair_i = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)  # Sender indices
        data.pair_j = np.array([1, 2, 0, 2, 0, 1], dtype=np.int32)  # Receiver indices
        data.cell = np.array([10, 10, 10], dtype=np.float32)  # Cubic cell

        natoms = 3
        nghosts = 0
        species = torch.tensor([1, 1, 8], dtype=torch.int64)  # H, H, O

        batch = interface._prepare_batch(data, natoms, nghosts, species)

        # Check batch structure
        assert isinstance(batch, AtomGraphs)
        assert batch.n_node[0] == natoms
        assert len(batch.senders) == len(data.pair_i)
        assert len(batch.receivers) == len(data.pair_j)

        # Check node features
        assert "atomic_numbers" in batch.node_features
        assert "positions" in batch.node_features
        assert "atomic_numbers_embedding" in batch.node_features

        # Check edge features
        assert "vectors" in batch.edge_features
        assert "lengths" in batch.edge_features

        # Check system features
        assert "cell" in batch.system_features

        # Check shapes
        assert batch.node_features["atomic_numbers"].shape[0] == natoms
        assert batch.edge_features["vectors"].shape[0] == len(data.pair_i)
        assert batch.edge_features["vectors"].shape[1] == 3

    def test_compute_forces(self, conservative_regressor, mock_orb_mliap_unified):
        """Test force computation through LAMMPS interface."""
        interface = mock_orb_mliap_unified(conservative_regressor)

        # Mock LAMMPS data
        data = SimpleNamespace()
        data.nlocal = 2
        data.ntotal = 2
        data.npairs = 2
        data.x = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        data.pair_i = np.array([0, 1], dtype=np.int32)
        data.pair_j = np.array([1, 0], dtype=np.int32)
        data.type = np.array([1, 1], dtype=np.int64)  # Two hydrogens
        data.cell = np.array([5, 5, 5], dtype=np.float32)

        # Add attributes for output
        data.energy = 0.0
        data.pair_forces = np.zeros((2, 3))

        interface.compute_forces(data)

        # Check that energy and forces were set
        assert hasattr(data, 'energy')
        assert hasattr(data, 'pair_forces')
        assert data.pair_forces.shape == (2, 3)  # 2 pairs, 3 force components

    def test_empty_system_handling(self, conservative_regressor, mock_orb_mliap_unified):
        """Test handling of empty systems."""
        interface = mock_orb_mliap_unified(conservative_regressor)

        # Mock empty system
        data = SimpleNamespace()
        data.nlocal = 0
        data.ntotal = 0
        data.npairs = 0

        # Should return early without error
        result = interface.compute_forces(data)
        assert result is None


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_edge_vectors_and_lengths(self, lammps_interface_classes):
        """Test edge vector and length computation."""
        get_edge_vectors_and_lengths = lammps_interface_classes['get_edge_vectors_and_lengths']
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        edge_index = torch.tensor([[0, 1], [1, 2]])  # 0->1, 1->2

        vectors, lengths = get_edge_vectors_and_lengths(positions, edge_index)

        # Check shapes
        assert vectors.shape == (2, 3)
        assert lengths.shape == (2,)

        # Check values
        expected_vectors = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0]])
        expected_lengths = torch.tensor([1.0, torch.sqrt(torch.tensor(2.0))])

        assert torch.allclose(vectors, expected_vectors)
        assert torch.allclose(lengths, expected_lengths)

    def test_get_edge_vectors_with_shifts(self, lammps_interface_classes):
        """Test edge vector computation with periodic shifts."""
        get_edge_vectors_and_lengths = lammps_interface_classes['get_edge_vectors_and_lengths']
        positions = torch.tensor([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
        edge_index = torch.tensor([[0], [1]])  # 0->1
        shifts = torch.tensor([[-10.0, 0.0, 0.0]])  # Periodic shift

        vectors, lengths = get_edge_vectors_and_lengths(positions, edge_index, shifts)

        # Should account for periodic shift: 9 - 0 + (-10) = -1
        expected_vectors = torch.tensor([[-1.0, 0.0, 0.0]])
        expected_lengths = torch.tensor([1.0])

        assert torch.allclose(vectors, expected_vectors)
        assert torch.allclose(lengths, expected_lengths)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline(self, conservative_regressor, mock_orb_mliap_unified):
        """Test the full pipeline from LAMMPS data to force computation."""
        interface = mock_orb_mliap_unified(conservative_regressor)

        # Create realistic test system (water molecule)
        data = SimpleNamespace()
        data.nlocal = 3
        data.ntotal = 3
        data.npairs = 6  # All pairs for 3 atoms
        data.x = np.array([
            [0.0, 0.0, 0.0],      # O
            [0.96, 0.0, 0.0],     # H1
            [-0.24, 0.93, 0.0]    # H2
        ], dtype=np.float32)
        data.pair_i = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        data.pair_j = np.array([1, 2, 0, 2, 0, 1], dtype=np.int32)
        data.type = np.array([8, 1, 1], dtype=np.int64)  # O, H, H
        data.cell = np.array([10, 10, 10], dtype=np.float32)

        # Add output attributes
        data.energy = 0.0
        data.pair_forces = np.zeros((6, 3))

        # Run computation
        interface.compute_forces(data)

        # Check outputs
        assert isinstance(data.energy, float)
        assert data.pair_forces.shape == (6, 3)
        assert not np.isnan(data.pair_forces).any()
        assert not np.isinf(data.pair_forces).any()

    def test_gradient_consistency(self, conservative_regressor, lammps_interface_classes):
        """Test that edge forces are consistent with energy gradients."""
        OrbPairEdgesForceWrapper = lammps_interface_classes['OrbPairEdgesForceWrapper']
        wrapper = OrbPairEdgesForceWrapper(conservative_regressor)

        # Create test system
        atoms = Atoms("LiF", positions=np.array([[0, 0, 0], [2, 0, 0]]))
        atoms.set_cell([8, 8, 8])
        atoms.set_pbc(True)

        system_config = SystemConfig(radius=6.0, max_num_neighbors=20)
        batch = ase_atoms_to_atom_graphs(atoms, system_config=system_config)

        # Get predictions
        predictions = wrapper(batch)
        energy = predictions["energy"]
        edge_forces = predictions["edge_forces"]

        # Check that energy has correct shape (should be [1] for a single system)
        assert energy.dim() == 1 and energy.shape[0] == 1

        # Check that edge forces have correct shape
        assert edge_forces.dim() == 2 and edge_forces.shape[1] == 3
        assert edge_forces.shape[0] == batch.n_edge[0]

        # Edge forces should be finite
        assert torch.isfinite(edge_forces).all()
        assert torch.isfinite(energy).all()
