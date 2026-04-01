"""Derived physical quantities for functional-level UQ evaluation.

All functions take a single field of shape (H, W) and return a scalar float.
These are applied to individual sample fields, then CRPS is computed
over the ensemble of scalar predictions vs the ground truth scalar.
"""

import numpy as np


def center_temperature(field):
    """Temperature at the center of the domain.

    Uses bilinear average of the 4 center pixels. Requires even grid dimensions.
    """
    H, W = field.shape
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError(f"center_temperature requires even grid dimensions, got ({H}, {W})")
    ch, cw = H // 2, W // 2
    return float(0.25 * (field[ch-1, cw-1] + field[ch-1, cw] + field[ch, cw-1] + field[ch, cw]))


def subregion_mean_temperature(field):
    """Mean temperature over the central [0.25, 0.75]^2 subregion."""
    H, W = field.shape
    return float(field[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4].mean())


def max_interior_temperature(field):
    """Maximum temperature in the interior (excluding boundary pixels)."""
    return float(field[1:-1, 1:-1].max())


def dirichlet_energy(field):
    """Dirichlet energy: 0.5 * integral(|grad T|^2) dx dy.

    Discrete approximation using finite differences on a unit [0,1]^2 domain.
    """
    H, W = field.shape
    hx = 1.0 / (W - 1)
    hy = 1.0 / (H - 1)
    dx = np.diff(field, axis=1) / hx  # (H, W-1)
    dy = np.diff(field, axis=0) / hy  # (H-1, W)
    return float(0.5 * hx * hy * (np.sum(dx**2) + np.sum(dy**2)))


def top_edge_heat_flux(field):
    """Mean heat flux through the top boundary: -dT/dy at y=1.

    Uses one-sided finite difference from the top row.
    Positive flux = heat flowing out of the domain.
    """
    H, _ = field.shape
    hy = 1.0 / (H - 1)
    flux = -(field[0, :] - field[1, :]) / hy
    return float(flux.mean())


# Registry for iteration
FUNCTIONALS = {
    "center_T": center_temperature,
    "subregion_mean_T": subregion_mean_temperature,
    "max_interior_T": max_interior_temperature,
    "dirichlet_energy": dirichlet_energy,
    "top_edge_flux": top_edge_heat_flux,
}


def compute_crps_scalar(samples, truth):
    """Exact pairwise CRPS for scalar samples.

    CRPS = E|X - y| - 0.5 * E|X - X'|

    Args:
        samples: array of shape (K,) — K scalar predictions
        truth: scalar ground truth

    Returns:
        CRPS value (float)
    """
    K = len(samples)
    term1 = np.mean(np.abs(samples - truth))
    if K < 2:
        return float(term1)
    term2 = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            term2 += np.abs(samples[i] - samples[j])
    term2 = 2.0 * term2 / (K * (K - 1))
    return float(term1 - 0.5 * term2)


def compute_functional_crps(sample_fields, truth_field):
    """Compute CRPS for each derived quantity.

    Args:
        sample_fields: (K, H, W) array of K sample fields
        truth_field: (H, W) ground truth field

    Returns:
        Dict mapping "crps_{name}" -> float for each functional
    """
    results = {}
    for name, fn in FUNCTIONALS.items():
        truth_val = fn(truth_field)
        sample_vals = np.array([fn(s) for s in sample_fields])
        results[f"crps_{name}"] = compute_crps_scalar(sample_vals, truth_val)
        results[f"truth_{name}"] = float(truth_val)
        results[f"sample_mean_{name}"] = float(sample_vals.mean())
        results[f"sample_std_{name}"] = float(sample_vals.std())
    return results
