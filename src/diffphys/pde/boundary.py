"""Corner-consistent boundary condition sampling with 5 BC families.

The "piecewise constant" OOD family uses smooth step transitions
(via the x(1-x) endpoint envelope), not true discontinuities.
The OOD signal is the step-like interior structure, which is
absent from all training BC families.
"""

import numpy as np

BC_TYPES = ["sinusoidal", "fourier", "bump", "piecewise", "linear"]
IN_DIST_TYPES = ["sinusoidal", "fourier", "bump", "linear"]


def sample_corners(rng):
    """Sample 4 corner values from Uniform(-1, 1).

    Returns (c_tl, c_tr, c_bl, c_br) — top-left, top-right,
    bottom-left, bottom-right.
    """
    return tuple(rng.uniform(-1, 1, 4).tolist())


def sample_perturbation(rng, nx, bc_type):
    """Sample a raw perturbation profile for the given BC type.

    Returns (nx,) array. Applied BEFORE the x(1-x) endpoint envelope.
    """
    x = np.linspace(0, 1, nx)

    if bc_type == "sinusoidal":
        A = rng.uniform(0.5, 2.0)
        n = int(rng.choice([1, 2, 3, 4]))
        return A * np.sin(n * np.pi * x)

    elif bc_type == "fourier":
        K = 5
        result = np.zeros(nx)
        for k in range(1, K + 1):
            a_k = rng.normal(0, 1.0 / k**2)
            result += a_k * np.sin(k * np.pi * x)
        return result

    elif bc_type == "bump":
        A = rng.uniform(0.5, 3.0)
        mu = rng.uniform(0.3, 0.7)
        sigma = rng.uniform(0.05, 0.15)
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    elif bc_type == "piecewise":
        n_levels = int(rng.choice([2, 3]))
        levels = rng.uniform(-2, 2, size=n_levels)
        width = 0.05

        if n_levels == 2:
            t = rng.uniform(0.3, 0.7)
            step = 0.5 * (1 + np.tanh((x - t) / width))
            return levels[0] * (1 - step) + levels[1] * step
        else:
            t1, t2 = sorted(rng.uniform(0.2, 0.8, size=2))
            s1 = 0.5 * (1 + np.tanh((x - t1) / width))
            s2 = 0.5 * (1 + np.tanh((x - t2) / width))
            return levels[0] * (1 - s1) + levels[1] * (s1 - s2) + levels[2] * s2

    elif bc_type == "linear":
        return np.zeros(nx)

    else:
        raise ValueError(f"Unknown BC type: {bc_type}")


def sample_edge_profile(rng, c_start, c_end, bc_type=None, nx=64):
    """Sample an edge profile constrained to match endpoint corners.

    The x(1-x)*4 envelope forces perturbation to zero at x=0 and x=1,
    guaranteeing profile[0] = c_start, profile[-1] = c_end.
    """
    if bc_type is None:
        bc_type = rng.choice(BC_TYPES)

    x = np.linspace(0, 1, nx)
    baseline = c_start + (c_end - c_start) * x
    perturbation = sample_perturbation(rng, nx, bc_type)
    envelope = x * (1 - x) * 4  # zero at endpoints, max 1 at center

    return baseline + perturbation * envelope


def sample_four_edges(rng, allowed_types=None, nx=64):
    """Sample 4 corner-consistent edge profiles.

    Returns (bc_top, bc_bottom, bc_left, bc_right) where corners match:
    - bc_top[0] == bc_left[0]       (top-left)
    - bc_top[-1] == bc_right[0]     (top-right)
    - bc_bottom[0] == bc_left[-1]   (bottom-left)
    - bc_bottom[-1] == bc_right[-1] (bottom-right)
    """
    if allowed_types is None:
        allowed_types = BC_TYPES

    c_tl, c_tr, c_bl, c_br = sample_corners(rng)

    bc_top = sample_edge_profile(rng, c_tl, c_tr, rng.choice(allowed_types), nx)
    bc_bottom = sample_edge_profile(rng, c_bl, c_br, rng.choice(allowed_types), nx)
    bc_left = sample_edge_profile(rng, c_tl, c_bl, rng.choice(allowed_types), nx)
    bc_right = sample_edge_profile(rng, c_tr, c_br, rng.choice(allowed_types), nx)

    return bc_top, bc_bottom, bc_left, bc_right
