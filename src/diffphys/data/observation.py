"""Observation regime transformations for Phase 2.

Regimes corrupt boundary conditions with noise and/or sparsity,
returning (observed_bc, mask) pairs for conditioning.
"""

import torch


REGIMES = ["exact", "dense-noisy", "sparse-clean", "sparse-noisy", "very-sparse"]

REGIME_CONFIG = {
    "exact":        {"n_points": 64, "noise_sigma": 0.0},
    "dense-noisy":  {"n_points": 64, "noise_sigma": 0.1},
    "sparse-clean": {"n_points": 16, "noise_sigma": 0.0},
    "sparse-noisy": {"n_points": 16, "noise_sigma": 0.1},
    "very-sparse":  {"n_points": 8,  "noise_sigma": 0.2},
}


def apply_observation_regime(bc, regime, rng=None):
    """Apply observation regime to a single BC edge.

    Args:
        bc: (nx,) tensor of true boundary values.
        regime: One of REGIMES.
        rng: Optional torch.Generator for reproducible noise.

    Returns:
        (observed_bc, mask): both (nx,) tensors.
        observed_bc has interpolated values between observed points.
        mask is 1.0 at observed positions, 0.0 elsewhere.
    """
    if regime not in REGIME_CONFIG:
        raise ValueError(f"Unknown regime: {regime}")

    nx = bc.shape[0]
    cfg = REGIME_CONFIG[regime]
    n_points = cfg["n_points"]
    sigma = cfg["noise_sigma"]

    if n_points >= nx:
        # All points observed
        mask = torch.ones(nx)
        observed = bc.clone()
        if sigma > 0:
            noise = torch.randn(nx, generator=rng) * sigma
            observed = observed + noise
        return observed, mask

    # Sparse: select evenly-spaced points including endpoints
    indices = torch.linspace(0, nx - 1, n_points).long()
    mask = torch.zeros(nx)
    mask[indices] = 1.0

    # Get observed values (with optional noise)
    obs_values = bc[indices].clone()
    if sigma > 0:
        noise = torch.randn(n_points, generator=rng) * sigma
        obs_values = obs_values + noise

    # Interpolate between observed points
    x_obs = indices.float()
    x_all = torch.arange(nx, dtype=torch.float32)
    observed = _linear_interp(x_obs, obs_values, x_all)

    return observed, mask


def _linear_interp(x_obs, y_obs, x_query):
    """Piecewise linear interpolation."""
    result = torch.zeros_like(x_query)
    for i in range(len(x_obs) - 1):
        x0, x1 = x_obs[i], x_obs[i + 1]
        y0, y1 = y_obs[i], y_obs[i + 1]
        mask = (x_query >= x0) & (x_query <= x1)
        t = (x_query[mask] - x0) / (x1 - x0)
        result[mask] = y0 + t * (y1 - y0)
    return result
