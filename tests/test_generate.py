"""Tests for dataset generation with held-out OOD split."""

import numpy as np
import pytest
from diffphys.pde.generate import generate_dataset
from diffphys.pde.laplace import LaplaceSolver


@pytest.fixture
def solver():
    return LaplaceSolver(nx=64)


def test_shapes(solver):
    data = generate_dataset(10, seed=42, solver=solver, nx=64)
    assert data["fields"].shape == (10, 64, 64)
    assert data["bc_top"].shape == (10, 64)
    assert data["bc_bottom"].shape == (10, 64)
    assert data["bc_left"].shape == (10, 64)
    assert data["bc_right"].shape == (10, 64)


def test_dtype_is_float32(solver):
    data = generate_dataset(5, seed=42, solver=solver, nx=64)
    assert data["fields"].dtype == np.float32
    assert data["bc_top"].dtype == np.float32


def test_determinism(solver):
    d1 = generate_dataset(5, seed=42, solver=solver, nx=64)
    d2 = generate_dataset(5, seed=42, solver=solver, nx=64)
    np.testing.assert_array_equal(d1["fields"], d2["fields"])


def test_bc_consistency(solver):
    """Field edges should match stored BCs (both float32)."""
    data = generate_dataset(5, seed=42, solver=solver, nx=64)
    for i in range(5):
        np.testing.assert_allclose(
            data["fields"][i, 0, :], data["bc_top"][i], atol=1e-6
        )
        np.testing.assert_allclose(
            data["fields"][i, -1, :], data["bc_bottom"][i], atol=1e-6
        )
        np.testing.assert_allclose(
            data["fields"][i, :, 0], data["bc_left"][i], atol=1e-6
        )
        np.testing.assert_allclose(
            data["fields"][i, :, -1], data["bc_right"][i], atol=1e-6
        )


def test_ood_split(solver):
    """OOD dataset with piecewise-only BCs should produce valid fields."""
    data = generate_dataset(
        5, seed=45, solver=solver, allowed_bc_types=["piecewise"], nx=64
    )
    assert data["fields"].shape == (5, 64, 64)
    assert np.all(np.isfinite(data["fields"]))


def test_max_principle(solver):
    """All generated fields should satisfy the maximum principle."""
    data = generate_dataset(10, seed=42, solver=solver, nx=64)
    for i in range(10):
        interior = data["fields"][i, 1:-1, 1:-1]
        all_bc = np.concatenate(
            [
                data["bc_top"][i],
                data["bc_bottom"][i],
                data["bc_left"][i],
                data["bc_right"][i],
            ]
        )
        assert interior.min() >= all_bc.min() - 1e-5
        assert interior.max() <= all_bc.max() + 1e-5
