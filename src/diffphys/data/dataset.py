"""PyTorch dataset for Laplace PDE solution fields."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .conditioning import encode_conditioning
from .observation import apply_observation_regime, REGIMES


class LaplacePDEDataset(Dataset):
    """Loads .npz files produced by diffphys.pde.generate.

    Each sample returns (conditioning, target) where:
      conditioning: (8, nx, nx) float32 tensor
      target: (1, nx, nx) float32 tensor

    Args:
        npz_path: Path to .npz file.
        regime: Observation regime. "exact" for Phase 1,
            "mixed" for random regime per sample (Phase 2 training),
            or a specific regime name for evaluation.
    """

    def __init__(self, npz_path, regime="exact"):
        data = np.load(npz_path)
        self.fields = torch.from_numpy(data["fields"])      # (N, nx, nx)
        self.bc_top = torch.from_numpy(data["bc_top"])       # (N, nx)
        self.bc_bottom = torch.from_numpy(data["bc_bottom"])
        self.bc_left = torch.from_numpy(data["bc_left"])
        self.bc_right = torch.from_numpy(data["bc_right"])
        self.regime = regime

    def __len__(self):
        return self.fields.shape[0]

    def __getitem__(self, idx):
        regime = self.regime
        if regime == "mixed":
            regime = REGIMES[torch.randint(len(REGIMES), (1,)).item()]

        bcs = [self.bc_top[idx], self.bc_bottom[idx],
               self.bc_left[idx], self.bc_right[idx]]

        if regime == "exact":
            cond = encode_conditioning(*bcs)
        else:
            obs_bcs = []
            masks = []
            for bc in bcs:
                obs, mask = apply_observation_regime(bc, regime)
                obs_bcs.append(obs)
                masks.append(mask)
            cond = encode_conditioning(*obs_bcs, *masks)

        target = self.fields[idx].unsqueeze(0)  # (1, nx, nx)
        return cond, target
