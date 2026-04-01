"""Tests for derived physical quantity functions."""
import numpy as np
import pytest
from diffphys.evaluation.functionals import (
    center_temperature,
    subregion_mean_temperature,
    max_interior_temperature,
    dirichlet_energy,
    top_edge_heat_flux,
)


class TestCenterTemperature:
    def test_returns_scalar(self):
        field = np.ones((64, 64))
        assert np.isscalar(center_temperature(field))

    def test_uniform_field(self):
        field = np.full((64, 64), 3.0)
        assert center_temperature(field) == pytest.approx(3.0)

    def test_center_value(self):
        field = np.zeros((64, 64))
        field[32, 32] = 5.0
        # Center is average of 4 center pixels (bilinear at midpoint)
        assert center_temperature(field) > 0

    def test_non_64_grid(self):
        field = np.full((32, 32), 7.0)
        assert center_temperature(field) == pytest.approx(7.0)

    def test_non_square_grid(self):
        field = np.full((128, 64), 2.0)
        assert center_temperature(field) == pytest.approx(2.0)

    def test_odd_grid_raises(self):
        field = np.ones((65, 65))
        with pytest.raises(ValueError, match="even grid dimensions"):
            center_temperature(field)


class TestSubregionMean:
    def test_returns_scalar(self):
        field = np.ones((64, 64))
        assert np.isscalar(subregion_mean_temperature(field))

    def test_uniform_field(self):
        field = np.full((64, 64), 2.5)
        assert subregion_mean_temperature(field) == pytest.approx(2.5, abs=0.1)

    def test_non_64_grid(self):
        field = np.full((128, 128), 4.0)
        assert subregion_mean_temperature(field) == pytest.approx(4.0)


class TestMaxInterior:
    def test_excludes_boundary(self):
        field = np.zeros((64, 64))
        field[0, :] = 100.0  # boundary
        field[32, 32] = 5.0  # interior
        assert max_interior_temperature(field) == pytest.approx(5.0)

    def test_non_64_grid(self):
        field = np.zeros((16, 16))
        field[0, :] = 100.0
        field[8, 8] = 3.0
        assert max_interior_temperature(field) == pytest.approx(3.0)


class TestDirichletEnergy:
    def test_constant_field_zero_energy(self):
        field = np.full((64, 64), 3.0)
        assert dirichlet_energy(field) == pytest.approx(0.0, abs=1e-10)

    def test_gradient_field_positive_energy(self):
        x = np.linspace(0, 1, 64)
        field = np.outer(np.ones(64), x)  # linear gradient
        assert dirichlet_energy(field) > 0

    def test_linear_gradient_exact(self):
        # T(x,y) = x on [0,1]^2 => |grad T|^2 = 1 everywhere
        # Dirichlet energy = 0.5 * integral(1) = 0.5
        # Discrete approx has O(1/N) error: 0.5*N/(N-1)
        for N in [32, 64, 128]:
            x = np.linspace(0, 1, N)
            field = np.outer(np.ones(N), x)
            assert dirichlet_energy(field) == pytest.approx(0.5, rel=0.05), f"Failed for N={N}"

    def test_resolution_invariant(self):
        # Same physical field at different resolutions should give similar energy
        e32 = dirichlet_energy(np.outer(np.ones(32), np.linspace(0, 1, 32)))
        e128 = dirichlet_energy(np.outer(np.ones(128), np.linspace(0, 1, 128)))
        assert e32 == pytest.approx(e128, rel=0.05)


class TestHeatFlux:
    def test_returns_scalar(self):
        field = np.ones((64, 64))
        assert np.isscalar(top_edge_heat_flux(field))

    def test_uniform_field_zero_flux(self):
        field = np.full((64, 64), 3.0)
        assert top_edge_heat_flux(field) == pytest.approx(0.0, abs=1e-6)

    def test_gradient_field_nonzero_flux(self):
        y = np.linspace(0, 1, 64)
        field = np.outer(y, np.ones(64))  # gradient in y
        assert abs(top_edge_heat_flux(field)) > 0

    def test_linear_gradient_exact(self):
        # T(x,y) = y on [0,1]^2 with y=0 at top row, y=1 at bottom
        # Heat flows from hot (bottom) to cold (top), i.e. out through top
        # Positive flux = outward, so expect +1.0
        for N in [32, 64, 128]:
            y = np.linspace(0, 1, N)
            field = np.outer(y, np.ones(N))
            assert top_edge_heat_flux(field) == pytest.approx(1.0, rel=0.02), f"Failed for N={N}"

    def test_resolution_invariant(self):
        f32 = top_edge_heat_flux(np.outer(np.linspace(0, 1, 32), np.ones(32)))
        f128 = top_edge_heat_flux(np.outer(np.linspace(0, 1, 128), np.ones(128)))
        assert f32 == pytest.approx(f128, rel=0.05)


class TestFunctionalCRPS:
    def test_perfect_samples_low_crps(self):
        from diffphys.evaluation.functionals import compute_functional_crps
        # 5 samples all equal to truth
        truth = np.random.randn(64, 64)
        samples = np.stack([truth] * 5)
        crps = compute_functional_crps(samples, truth)
        for name, val in crps.items():
            if name.startswith("crps_"):
                assert val == pytest.approx(0.0, abs=1e-6), f"{name} CRPS should be ~0"

    def test_bad_samples_higher_crps(self):
        from diffphys.evaluation.functionals import compute_functional_crps
        truth = np.zeros((64, 64))
        samples = np.random.randn(5, 64, 64) * 10  # way off
        crps = compute_functional_crps(samples, truth)
        for name, val in crps.items():
            if name.startswith("crps_"):
                assert val > 0, f"{name} CRPS should be > 0 for bad samples"

    def test_returns_all_quantities(self):
        from diffphys.evaluation.functionals import compute_functional_crps, FUNCTIONALS
        truth = np.random.randn(64, 64)
        samples = np.random.randn(5, 64, 64)
        crps = compute_functional_crps(samples, truth)
        for name in FUNCTIONALS:
            assert f"crps_{name}" in crps
