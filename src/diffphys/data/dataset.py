"""PyTorch dataset for Laplace PDE solution fields."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .conditioning import encode_conditioning


class LaplacePDEDataset(Dataset):
    """Loads .npz files produced by diffphys.pde.generate.

    Each sample returns (conditioning, target) where:
      conditioning: (8, nx, nx) float32 tensor
      target: (1, nx, nx) float32 tensor
    """

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.fields = torch.from_numpy(data["fields"])      # (N, nx, nx)
        self.bc_top = torch.from_numpy(data["bc_top"])       # (N, nx)
        self.bc_bottom = torch.from_numpy(data["bc_bottom"])
        self.bc_left = torch.from_numpy(data["bc_left"])
        self.bc_right = torch.from_numpy(data["bc_right"])

    def __len__(self):
        return self.fields.shape[0]

    def __getitem__(self, idx):
        cond = encode_conditioning(
            self.bc_top[idx], self.bc_bottom[idx],
            self.bc_left[idx], self.bc_right[idx],
        )
        target = self.fields[idx].unsqueeze(0)  # (1, nx, nx)
        return cond, target
