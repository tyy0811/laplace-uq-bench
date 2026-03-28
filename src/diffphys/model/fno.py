"""Fourier Neural Operator for 2D PDE fields.

4 Fourier layers with spectral convolutions. Input is the 8-channel
conditioning tensor + 2 grid coordinate channels, lifted to a hidden
width, processed in Fourier space, then projected to output.
~2M params with width=40, modes=12.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Spectral convolution via truncated FFT."""

    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_ch * out_ch)
        self.weights1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    def _compl_mul2d(self, x, weights):
        # (B, in_ch, M1, M2) x (in_ch, out_ch, M1, M2) -> (B, out_ch, M1, M2)
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(B, self.out_ch, x.size(2), x.size(3) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2] = \
            self._compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self._compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        return torch.fft.irfft2(out_ft, s=(x.size(2), x.size(3)))


class FNOBlock(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes, modes)
        self.linear = nn.Conv2d(width, width, 1)
        self.norm = nn.InstanceNorm2d(width)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.linear(x)))


class FNO2d(nn.Module):
    """Fourier Neural Operator for 2D fields.

    Args:
        in_ch: Input channels (8 for BC conditioning).
        out_ch: Output channels (1 for field prediction).
        width: Hidden channel width.
        modes: Number of Fourier modes to keep per dimension.
        n_layers: Number of Fourier layers.
    """

    def __init__(self, in_ch=8, out_ch=1, width=40, modes=12, n_layers=4):
        super().__init__()
        self.width = width

        # +2 for grid coordinates (x, y)
        self.lift = nn.Linear(in_ch + 2, width)

        self.layers = nn.ModuleList([
            FNOBlock(width, modes) for _ in range(n_layers)
        ])

        self.project = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, out_ch),
        )

    def _get_grid(self, x):
        B, _, H, W = x.shape
        gx = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        gy = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        return torch.cat([gx, gy], dim=1)

    def forward(self, x):
        grid = self._get_grid(x)
        x = torch.cat([x, grid], dim=1)  # (B, in_ch+2, H, W)

        # Lift: channel-wise linear (permute to last dim)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.lift(x)
        x = x.permute(0, 3, 1, 2)  # (B, width, H, W)

        for layer in self.layers:
            x = layer(x)

        # Project: channel-wise linear
        x = x.permute(0, 2, 3, 1)
        x = self.project(x)
        return x.permute(0, 3, 1, 2)  # (B, out_ch, H, W)
