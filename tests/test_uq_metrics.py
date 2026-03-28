"""Tests for uncertainty quantification metrics."""

import torch
import pytest
from diffphys.evaluation.uq_metrics import (
    pixelwise_coverage,
    crps_gaussian,
    calibration_error,
    sharpness,
)


class TestPixelwiseCoverage:
    def test_perfect_coverage_at_wide_interval(self):
        """Very wide intervals should give ~100% coverage."""
        true = torch.zeros(10, 1, 8, 8)
        mean = torch.zeros(10, 1, 8, 8)
        std = torch.ones(10, 1, 8, 8) * 100  # huge std
        cov = pixelwise_coverage(true, mean, std, level=0.95)
        assert cov > 0.99

    def test_zero_coverage_at_narrow_interval(self):
        """Extremely narrow intervals on offset predictions -> low coverage."""
        true = torch.ones(10, 1, 8, 8)
        mean = torch.zeros(10, 1, 8, 8)
        std = torch.ones(10, 1, 8, 8) * 1e-6
        cov = pixelwise_coverage(true, mean, std, level=0.95)
        assert cov < 0.01

    def test_returns_scalar(self):
        true = torch.randn(4, 1, 8, 8)
        mean = torch.randn(4, 1, 8, 8)
        std = torch.ones(4, 1, 8, 8)
        cov = pixelwise_coverage(true, mean, std, level=0.90)
        assert cov.dim() == 0


class TestCRPSGaussian:
    def test_perfect_prediction_near_zero(self):
        """CRPS should be near zero when prediction matches truth exactly."""
        true = torch.zeros(4, 1, 8, 8)
        mean = torch.zeros(4, 1, 8, 8)
        std = torch.ones(4, 1, 8, 8) * 0.01
        crps = crps_gaussian(true, mean, std)
        assert crps.mean() < 0.01

    def test_increases_with_error(self):
        true = torch.zeros(4, 1, 8, 8)
        std = torch.ones(4, 1, 8, 8) * 0.5
        crps_close = crps_gaussian(true, torch.zeros(4, 1, 8, 8), std)
        crps_far = crps_gaussian(true, torch.ones(4, 1, 8, 8) * 5.0, std)
        assert crps_far.mean() > crps_close.mean()

    def test_output_shape(self):
        true = torch.randn(4, 1, 8, 8)
        mean = torch.randn(4, 1, 8, 8)
        std = torch.ones(4, 1, 8, 8)
        crps = crps_gaussian(true, mean, std)
        assert crps.shape == (4,)


class TestCalibrationError:
    def test_well_calibrated_low_error(self):
        """Standard normal samples should be well-calibrated."""
        torch.manual_seed(42)
        N = 1000
        true = torch.randn(N, 1, 1, 1)
        mean = torch.zeros(N, 1, 1, 1)
        std = torch.ones(N, 1, 1, 1)
        err = calibration_error(true, mean, std)
        assert err < 0.1

    def test_miscalibrated_high_error(self):
        """Overconfident predictions should have high calibration error."""
        torch.manual_seed(42)
        N = 1000
        true = torch.randn(N, 1, 1, 1) * 5
        mean = torch.zeros(N, 1, 1, 1)
        std = torch.ones(N, 1, 1, 1) * 0.1  # way too confident
        err = calibration_error(true, mean, std)
        assert err > 0.3


class TestSharpness:
    def test_narrow_is_sharper(self):
        std_narrow = torch.ones(4, 1, 8, 8) * 0.1
        std_wide = torch.ones(4, 1, 8, 8) * 10.0
        assert sharpness(std_narrow) < sharpness(std_wide)

    def test_returns_scalar(self):
        std = torch.ones(4, 1, 8, 8)
        s = sharpness(std)
        assert s.dim() == 0
