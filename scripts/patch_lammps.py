from lammps.mliap.mliap_unified_abc import MLIAPUnified
from torch import nn
import torch
from orb_models.forcefield.base import AtomGraphs
from typing import Tuple, Optional


def get_edge_vectors_and_lengths(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    shifts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute edge vectors and lengths: vectors = positions[receiver] - positions[sender] + shifts"""
    senders = edge_index[0]
    receivers = edge_index[1]
    vectors = positions[receivers] - positions[senders]
    if shifts is not None:
        vectors = vectors + shifts
    lengths = torch.sqrt(torch.sum(vectors * vectors, dim=-1) + 1e-12)
    return vectors, lengths


class OrbPairEdgesForceWrapper:
    """Wrapper for conservative ORB models that computes edge forces."""

    def __init__(self, model: nn.Module):
        self.model = model
        if not hasattr(model, "grad_forces_name"):
            raise ValueError("Model must be a ConservativeForcefieldRegressor")

    def __call__(self, batch):
        """Compute conservative predictions and edge forces."""
        vectors, stress_displacement, generator = batch.compute_differentiable_edge_vectors()
        if stress_displacement is not None:
            batch.system_features["stress_displacement"] = stress_displacement
        if generator is not None:
            batch.system_features["generator"] = generator
        batch.edge_features["vectors"] = vectors
        # Use overridden predict method that includes edge forces
        predictions = self.predict(batch)
        return predictions

    def predict(self, batch, split=False):
        """Rewritten predict method for LAMMPS with proper edge force computation."""
        # Get base model features (node embeddings)
        out = self.model.model(batch)
        node_features = out["node_features"]
        # Compute energy
        energy_head = self.model.heads[self.model.energy_name]
        base_energy = energy_head(node_features, batch)
        raw_energy = energy_head.denormalize(base_energy, batch)
        # Add pair repulsion if enabled
        if self.model.pair_repulsion:
            raw_energy += self.model.pair_repulsion_fn(batch)["energy"]
        # Normalize energy for output
        normalized_energy = energy_head.normalize(raw_energy, batch, online=False)

        # Compute edge forces via gradients w.r.t. edge vectors
        edge_vectors = batch.edge_features["vectors"]
        edge_forces = torch.autograd.grad(
            outputs=raw_energy.sum(),
            inputs=edge_vectors,
            create_graph=False,
            retain_graph=False,
            allow_unused=True
        )[0]
        if edge_forces is None:
            raise ValueError("Edge forces are None - gradient computation failed")
        edge_forces = -edge_forces  # F = -dE/dr
        # Prepare predictions dictionary
        predictions = {
            self.model.energy_name: normalized_energy,
            "energy": raw_energy,  # For LAMMPS compatibility
            "edge_forces": edge_forces,
        }
        return predictions

    def _compute_edge_forces(self, batch, predictions):
        """Compute edge forces: f_ij = -∂E/∂r_ij for each pair interaction."""
        # Get energy from predictions
        if "energy" in predictions:
            energy = predictions["energy"]
        else:
            raise ValueError("Could not find energy in predictions")
        # Get edge vectors, should already require gradients from conservative regressor
        edge_vectors = batch.edge_features["vectors"]
        # Compute edge forces as -∂E/∂edge_vectors
        edge_forces = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=edge_vectors,
            create_graph=False,
            retain_graph=False,
            allow_unused=True
        )[0]
        if edge_forces is None:
            raise ValueError(
                "Edge forces are None. The computational graph between energy and "
                "edge vectors has been broken. Make sure the edge vectors tensor has "
                "not been replaced since calling compute_differentiable_edge_vectors()"
            )
        else:
            # Apply negative sign for standard LAMMPS convention
            edge_forces = -edge_forces
        return edge_forces

    def _atom_forces_to_edge_forces(self, batch, atom_forces):
        """Convert atom forces to edge forces (fallback method)."""
        senders = batch.senders
        receivers = batch.receivers
        edge_forces = atom_forces[receivers] - atom_forces[senders]
        return edge_forces

    def _compute_edge_forces_direct(self, batch, energy):
        """Compute edge forces directly via ∂E/∂edge_vectors (alternative approach)."""
        edge_vectors = batch.edge_features["vectors"]
        # Ensure edge vectors require gradients
        if not edge_vectors.requires_grad:
            edge_vectors = edge_vectors.detach().requires_grad_(True)
        edge_forces = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=edge_vectors,
            create_graph=False,
            retain_graph=False,
        )[0]
        return -edge_forces  # Apply negative sign for force direction


class ORB_MLIAPUnified(MLIAPUnified):
    """ORB interface for LAMMPS using the MLIAP interface."""

    def __init__(
        self,
        model: nn.Module,
        radius: float = 10.0,
        max_num_neighbors: int = 20,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__()
        self.model = OrbPairEdgesForceWrapper(model)
        self.step = 0
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors
        self.num_species = 118
        self.device = torch.device(device)
        self.dtype = dtype

    def _add_pair_edges_force(self, *args, **kwargs):
        pass

    def _prepare_batch(self, data, natoms, nghosts, species):
        """Prepare ORB model input batch from LAMMPS data."""
        positions = torch.as_tensor(data.x, dtype=self.dtype, device=self.device)
        positions.requires_grad_(True)
        senders = torch.as_tensor(data.pair_i, dtype=torch.int64, device=self.device)
        receivers = torch.as_tensor(data.pair_j, dtype=torch.int64, device=self.device)
        edge_index = torch.stack([senders, receivers], dim=0)
        shifts = None
        if hasattr(data, "shifts") and data.shifts is not None:
            shifts = torch.as_tensor(data.shifts, dtype=self.dtype, device=self.device)
        elif hasattr(data, "rij_shifts") and data.rij_shifts is not None:
            shifts = torch.as_tensor(data.rij_shifts, dtype=self.dtype, device=self.device)
        edge_vectors, edge_lengths = get_edge_vectors_and_lengths(
            positions=positions, edge_index=edge_index, shifts=shifts
        )
        atomic_numbers = species.to(torch.int64).to(self.device)
        atomic_numbers_embedding = torch.nn.functional.one_hot(
            atomic_numbers, num_classes=self.num_species
        ).to(self.dtype)
        node_features = {
            "atomic_numbers": atomic_numbers,
            "atomic_numbers_embedding": atomic_numbers_embedding,
            "positions": positions,
        }
        # Create unit_shifts
        unit_shifts = torch.zeros(len(senders), 3, dtype=self.dtype, device=self.device)
        edge_features = {"vectors": edge_vectors, "lengths": edge_lengths, "unit_shifts": unit_shifts}
        if hasattr(data, "cell") and data.cell is not None:
            cell = torch.as_tensor(data.cell, dtype=self.dtype, device=self.device)
            if cell.dim() == 1:
                cell = torch.diag(cell).unsqueeze(0)
            elif cell.dim() == 2 and cell.shape[0] == 3 and cell.shape[1] == 3:
                cell = cell.unsqueeze(0)
        else:
            cell = torch.zeros(1, 3, 3, dtype=self.dtype, device=self.device)
        system_features = {"cell": cell}
        return AtomGraphs(
            senders=senders,
            receivers=receivers,
            n_node=torch.tensor([natoms], dtype=torch.int64, device=self.device),
            n_edge=torch.tensor([len(senders)], dtype=torch.int64, device=self.device),
            node_features=node_features,
            edge_features=edge_features,
            system_features=system_features,
            node_targets=None,
            edge_targets=None,
            system_targets=None,
            system_id=None,
            fix_atoms=None,
            tags=None,
            radius=self.radius,
            max_num_neighbors=self.max_num_neighbors,
        )

    def compute_forces(self, data):
        """Compute forces using conservative regressor: F = -dE/dr"""
        n_atoms = data.nlocal
        n_total = data.ntotal
        n_ghosts = n_total - n_atoms
        n_pairs = data.npairs
        self.step += 1
        if n_atoms == 0 or n_pairs < 1:
            return
        batch = self._prepare_batch(
            data, n_atoms, n_ghosts, torch.as_tensor(data.type, dtype=torch.int64)
        )
        predictions = self.model(batch)
        energy_name = self.model.model.energy_name
        energy = predictions.get(energy_name, None)
        edge_forces = predictions.get("edge_forces", None)

        if hasattr(data, "energy") and energy is not None:
            data.energy = energy.detach().cpu().item()
        if hasattr(data, "pair_forces") and edge_forces is not None:
            data.pair_forces = edge_forces.detach().cpu().numpy()
        if self.device.type != "cpu":
            torch.cuda.synchronize()
