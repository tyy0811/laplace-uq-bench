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
    """Mean absolute error on boundary pixels."""
    B = pred.shape[0]
    errors = []
    for f_pred, f_true in zip(pred[:, 0], true[:, 0]):
        top = (f_pred[0, :] - f_true[0, :]).abs().mean()
        bot = (f_pred[-1, :] - f_true[-1, :]).abs().mean()
        left = (f_pred[:, 0] - f_true[:, 0]).abs().mean()
        right = (f_pred[:, -1] - f_true[:, -1]).abs().mean()
        errors.append((top + bot + left + right) / 4)
    return torch.stack(errors)


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
