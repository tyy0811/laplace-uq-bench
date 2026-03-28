"""Tests for physics-aware evaluation metrics."""

import torch
import numpy as np
import pytest
from diffphys.evaluation.metrics import (
    relative_l2_error,
    pde_residual_norm,
    bc_error,
    max_principle_violations,
    energy_functional,
)


class TestRelativeL2Error:
    def test_identical_is_zero(self):
        x = torch.randn(4, 1, 64, 64)
        err = relative_l2_error(x, x)
        assert err.shape == (4,)
        torch.testing.assert_close(err, torch.zeros(4), atol=1e-7, rtol=0)

    def test_known_value(self):
        pred = torch.ones(1, 1, 4, 4) * 2.0
        true = torch.ones(1, 1, 4, 4) * 1.0
        err = relative_l2_error(pred, true)
        # ||pred - true|| / ||true|| = ||1|| / ||1|| = 1.0
        assert err.item() == pytest.approx(1.0, abs=1e-6)

    def test_batch_dimension(self):
        pred = torch.randn(8, 1, 16, 16)
        true = torch.randn(8, 1, 16, 16)
        err = relative_l2_error(pred, true)
        assert err.shape == (8,)


class TestPDEResidualNorm:
    def test_constant_field_is_zero(self):
        """Laplacian of a constant field is zero."""
        field = torch.ones(1, 1, 16, 16) * 3.0
        res = pde_residual_norm(field, h=1.0 / 15)
        assert res.item() == pytest.approx(0.0, abs=1e-6)

    def test_linear_field_is_zero(self):
        """Laplacian of a linear field (ax + by + c) is zero."""
        x = torch.linspace(0, 1, 16)
        y = torch.linspace(0, 1, 16)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        field = (2 * X + 3 * Y + 1).unsqueeze(0).unsqueeze(0)
        res = pde_residual_norm(field, h=1.0 / 15)
        assert res.item() == pytest.approx(0.0, abs=1e-3)

    def test_nonzero_for_nonharmonic(self):
        """Laplacian of x^2 + y^2 is 4 (not zero)."""
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        field = (X ** 2 + Y ** 2).unsqueeze(0).unsqueeze(0)
        res = pde_residual_norm(field, h=1.0 / 63)
        assert res.item() > 0.1

    def test_batch_dimension(self):
        field = torch.randn(4, 1, 16, 16)
        res = pde_residual_norm(field, h=1.0 / 15)
        assert res.shape == (4,)


class TestBCError:
    def test_exact_bcs_zero(self):
        pred = torch.randn(2, 1, 16, 16)
        true = pred.clone()
        err = bc_error(pred, true)
        torch.testing.assert_close(err, torch.zeros(2), atol=1e-7, rtol=0)

    def test_measures_boundary_difference(self):
        pred = torch.zeros(1, 1, 8, 8)
        true = torch.ones(1, 1, 8, 8)
        err = bc_error(pred, true)
        assert err.item() > 0

    def test_no_corner_double_counting(self):
        """Each corner pixel should be counted exactly once."""
        pred = torch.zeros(1, 1, 8, 8)
        true = torch.zeros(1, 1, 8, 8)
        # Set only the top-left corner to differ by 1.0
        true[:, :, 0, 0] = 1.0
        err = bc_error(pred, true)
        # 28 unique boundary pixels on an 8x8 grid: 2*8 + 2*(8-2) = 28
        expected = 1.0 / 28.0
        assert err.item() == pytest.approx(expected, abs=1e-6)


class TestMaxPrincipleViolations:
    def test_harmonic_field_no_violations(self):
        """Harmonic field satisfies max principle."""
        field = torch.ones(1, 1, 16, 16) * 0.5
        field[:, :, 0, :] = 0.0   # top
        field[:, :, -1, :] = 1.0  # bottom
        n_viol = max_principle_violations(field)
        assert n_viol.item() == 0

    def test_detects_violations(self):
        """Interior exceeding boundary extremes is a violation."""
        field = torch.zeros(1, 1, 8, 8)
        field[:, :, 4, 4] = 2.0  # interior > max boundary (0)
        n_viol = max_principle_violations(field)
        assert n_viol.item() > 0


class TestEnergyFunctional:
    def test_constant_field_zero_energy(self):
        """Constant field has zero gradient energy."""
        field = torch.ones(1, 1, 16, 16) * 5.0
        E = energy_functional(field, h=1.0 / 15)
        assert E.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive(self):
        field = torch.randn(2, 1, 16, 16)
        E = energy_functional(field, h=1.0 / 15)
        assert (E >= 0).all()

    def test_batch_dimension(self):
        field = torch.randn(4, 1, 16, 16)
        E = energy_functional(field, h=1.0 / 15)
        assert E.shape == (4,)
