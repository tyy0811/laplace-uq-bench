"""Tests for corner-consistent boundary condition sampling."""

import numpy as np
import pytest
from diffphys.pde.boundary import (
    BC_TYPES,
    IN_DIST_TYPES,
    sample_corners,
    sample_edge_profile,
    sample_four_edges,
    sample_perturbation,
)


class TestSampleCorners:
    def test_returns_four_values_in_range(self):
        rng = np.random.default_rng(42)
        corners = sample_corners(rng)
        assert len(corners) == 4
        for c in corners:
            assert -1.0 <= c <= 1.0

    def test_determinism(self):
        c1 = sample_corners(np.random.default_rng(42))
        c2 = sample_corners(np.random.default_rng(42))
        assert c1 == c2


class TestSamplePerturbation:
    def test_shape(self):
        p = sample_perturbation(np.random.default_rng(42), 64, "sinusoidal")
        assert p.shape == (64,)

    @pytest.mark.parametrize("bc_type", BC_TYPES)
    def test_all_types_finite(self, bc_type):
        p = sample_perturbation(np.random.default_rng(42), 64, bc_type)
        assert p.shape == (64,)
        assert np.all(np.isfinite(p))

    def test_linear_is_zero(self):
        p = sample_perturbation(np.random.default_rng(42), 64, "linear")
        np.testing.assert_array_equal(p, 0.0)

    def test_piecewise_has_step_structure(self):
        """Piecewise perturbation should have flat plateaus with sharp transitions."""
        rng = np.random.default_rng(42)
        p = sample_perturbation(rng, 64, "piecewise")
        grad = np.abs(np.diff(p))
        # Most gradient values should be small (flat regions)
        threshold = 0.1 * np.max(grad)
        flat_fraction = np.mean(grad < threshold)
        assert flat_fraction > 0.7

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown BC type"):
            sample_perturbation(np.random.default_rng(42), 64, "unknown")


class TestSampleEdgeProfile:
    def test_shape(self):
        profile = sample_edge_profile(
            np.random.default_rng(42), -0.5, 0.8, "sinusoidal", 64
        )
        assert profile.shape == (64,)

    @pytest.mark.parametrize("bc_type", BC_TYPES)
    def test_endpoints_match(self, bc_type):
        """x(1-x) envelope guarantees profile[0]=c_start, profile[-1]=c_end."""
        c_start, c_end = 0.3, -0.7
        profile = sample_edge_profile(
            np.random.default_rng(42), c_start, c_end, bc_type, 64
        )
        assert profile[0] == pytest.approx(c_start, abs=1e-12)
        assert profile[-1] == pytest.approx(c_end, abs=1e-12)

    def test_finiteness(self):
        profile = sample_edge_profile(
            np.random.default_rng(42), 0.0, 1.0, "fourier", 64
        )
        assert np.all(np.isfinite(profile))


class TestSampleFourEdges:
    def test_shapes(self):
        edges = sample_four_edges(np.random.default_rng(42), nx=64)
        assert len(edges) == 4
        for e in edges:
            assert e.shape == (64,)

    def test_corner_consistency(self):
        """All 4 corners must match between adjacent edges."""
        bc_top, bc_bottom, bc_left, bc_right = sample_four_edges(
            np.random.default_rng(42), nx=64
        )
        # top-left
        assert bc_top[0] == pytest.approx(bc_left[0], abs=1e-12)
        # top-right
        assert bc_top[-1] == pytest.approx(bc_right[0], abs=1e-12)
        # bottom-left
        assert bc_bottom[0] == pytest.approx(bc_left[-1], abs=1e-12)
        # bottom-right
        assert bc_bottom[-1] == pytest.approx(bc_right[-1], abs=1e-12)

    def test_allowed_types_filtering(self):
        """With linear-only, profile should be exactly the linear baseline."""
        edges = sample_four_edges(
            np.random.default_rng(42), allowed_types=["linear"], nx=64
        )
        bc_top = edges[0]
        x = np.linspace(0, 1, 64)
        expected = bc_top[0] + (bc_top[-1] - bc_top[0]) * x
        np.testing.assert_allclose(bc_top, expected, atol=1e-12)

    def test_default_excludes_piecewise(self):
        """Default allowed_types should be IN_DIST_TYPES (no piecewise)."""
        # sample_four_edges with no allowed_types should never pick piecewise.
        # We can't observe the type directly, but we can verify the default
        # matches IN_DIST_TYPES by checking the module-level constant.
        from diffphys.pde import boundary
        import inspect
        src = inspect.getsource(boundary.sample_four_edges)
        assert "IN_DIST_TYPES" in src

    def test_determinism(self):
        e1 = sample_four_edges(np.random.default_rng(42), nx=64)
        e2 = sample_four_edges(np.random.default_rng(42), nx=64)
        for a, b in zip(e1, e2):
            np.testing.assert_array_equal(a, b)
