"""Physics-aware evaluation metrics for PDE field predictions.

All functions expect tensors of shape (B, 1, H, W) and return
per-sample metrics of shape (B,).
"""

import torch


def relative_l2_error(pred, true):
    """||pred - true||_2 / ||true||_2, per sample."""
    diff = (pred - true).reshape(pred.shape[0], -1)
    true_flat = true.reshape(true.shape[0], -1)
    return diff.norm(dim=1) / true_flat.norm(dim=1).clamp(min=1e-8)


def pde_residual_norm(field, h=1.0 / 63):
    """RMS of discrete Laplacian on interior points.

    For a Laplace solution, this should be near zero.
    """
    f = field[:, 0]  # (B, H, W)
    lap = (
        f[:, :-2, 1:-1] + f[:, 2:, 1:-1]
        + f[:, 1:-1, :-2] + f[:, 1:-1, 2:]
        - 4 * f[:, 1:-1, 1:-1]
    ) / (h ** 2)
    return lap.reshape(field.shape[0], -1).pow(2).mean(dim=1).sqrt()


def bc_error(pred, true):
    """Mean absolute error over unique boundary pixels.

    Collects top row, bottom row, left column (excluding corners),
    and right column (excluding corners) to avoid double-counting
    the four corner pixels.
    """
    p = pred[:, 0]  # (B, H, W)
    t = true[:, 0]
    diff = (p - t).abs()

    top = diff[:, 0, :]           # (B, W)
    bot = diff[:, -1, :]          # (B, W)
    left = diff[:, 1:-1, 0]      # (B, H-2)  corners excluded
    right = diff[:, 1:-1, -1]    # (B, H-2)  corners excluded

    all_bc = torch.cat([top, bot, left, right], dim=1)  # (B, 2W + 2(H-2))
    return all_bc.mean(dim=1)


def max_principle_violations(field):
    """Count interior pixels violating the discrete maximum principle."""
    f = field[:, 0]  # (B, H, W)
    interior = f[:, 1:-1, 1:-1]

    # Boundary extremes per sample
    top = f[:, 0, :]
    bot = f[:, -1, :]
    left = f[:, :, 0]
    right = f[:, :, -1]
    all_bc = torch.cat([top, bot, left, right], dim=1)
    bc_min = all_bc.min(dim=1, keepdim=True).values.unsqueeze(2)
    bc_max = all_bc.max(dim=1, keepdim=True).values.unsqueeze(2)

    violations = ((interior < bc_min - 1e-6) | (interior > bc_max + 1e-6))
    return violations.reshape(field.shape[0], -1).sum(dim=1)


def energy_functional(field, h=1.0 / 63):
    """Dirichlet energy: 0.5 * integral(|grad T|^2) dx dy."""
    f = field[:, 0]  # (B, H, W)
    dx = (f[:, :, 1:] - f[:, :, :-1]) / h
    dy = (f[:, 1:, :] - f[:, :-1, :]) / h
    # Average over the domain (trapezoidal-ish)
    E = 0.5 * h * h * (dx.pow(2).sum(dim=(1, 2)) + dy.pow(2).sum(dim=(1, 2)))
    return E
