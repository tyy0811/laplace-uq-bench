"""Dataset generation with held-out OOD split."""

import argparse
import os

import numpy as np

from .boundary import IN_DIST_TYPES, sample_four_edges
from .laplace import LaplaceSolver


def generate_dataset(n, seed, solver, allowed_bc_types=None, nx=64):
    """Generate n PDE solution samples with given BC types.

    Args:
        n: Number of samples to generate.
        seed: Random seed for reproducibility.
        solver: LaplaceSolver instance (reused for all samples).
        allowed_bc_types: List of BC family names, or None for all in-dist.
        nx: Grid resolution.

    Returns:
        Dict with keys: fields (n, nx, nx), bc_top/bottom/left/right (n, nx).
        All arrays are float32.
    """
    if allowed_bc_types is None:
        allowed_bc_types = IN_DIST_TYPES

    rng = np.random.default_rng(seed)

    fields = np.zeros((n, nx, nx), dtype=np.float32)
    bc_tops = np.zeros((n, nx), dtype=np.float32)
    bc_bottoms = np.zeros((n, nx), dtype=np.float32)
    bc_lefts = np.zeros((n, nx), dtype=np.float32)
    bc_rights = np.zeros((n, nx), dtype=np.float32)

    for i in range(n):
        bc_top, bc_bottom, bc_left, bc_right = sample_four_edges(
            rng, allowed_types=allowed_bc_types, nx=nx
        )
        field = solver.solve(bc_top, bc_bottom, bc_left, bc_right)

        fields[i] = field.astype(np.float32)
        bc_tops[i] = bc_top.astype(np.float32)
        bc_bottoms[i] = bc_bottom.astype(np.float32)
        bc_lefts[i] = bc_left.astype(np.float32)
        bc_rights[i] = bc_right.astype(np.float32)

    return {
        "fields": fields,
        "bc_top": bc_tops,
        "bc_bottom": bc_bottoms,
        "bc_left": bc_lefts,
        "bc_right": bc_rights,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate PDE dataset splits")
    parser.add_argument("--n-train", type=int, default=40000)
    parser.add_argument("--n-val", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--n-ood", type=int, default=1000)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--output", type=str, default="data")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    solver = LaplaceSolver(nx=args.nx)

    splits = [
        ("train", args.n_train, IN_DIST_TYPES, 42),
        ("val", args.n_val, IN_DIST_TYPES, 43),
        ("test_in", args.n_test, IN_DIST_TYPES, 44),
        ("test_ood", args.n_ood, ["piecewise"], 45),
    ]

    for name, n, bc_types, seed in splits:
        print(f"Generating {name} ({n} samples)...")
        data = generate_dataset(n, seed, solver, allowed_bc_types=bc_types, nx=args.nx)
        path = os.path.join(args.output, f"{name}.npz")
        np.savez_compressed(path, **data)
        print(f"  Saved to {path}")


if __name__ == "__main__":
    main()
