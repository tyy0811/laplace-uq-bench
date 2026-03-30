"""Tests for conformal prediction."""

import numpy as np
import pytest
from diffphys.evaluation.conformal import (
    SpatialConformalPredictor,
    PixelwiseConformalPredictor,
)


class TestSpatialConformalPredictor:
    def test_calibrate_sets_q_hat(self):
        cp = SpatialConformalPredictor(alpha=0.1)
        rng = np.random.default_rng(42)
        preds = rng.standard_normal((100, 8, 8))
        sigma = np.abs(rng.standard_normal((100, 8, 8))) + 0.1
        truth = preds + 0.5 * rng.standard_normal((100, 8, 8))
        q = cp.calibrate(preds, sigma, truth)
        assert q > 0
        assert cp.q_hat == q

    def test_predict_intervals_shape(self):
        cp = SpatialConformalPredictor(alpha=0.1)
        cp.q_hat = 2.0
        preds = np.random.randn(10, 8, 8)
        sigma = np.ones((10, 8, 8))
        lower, upper = cp.predict_intervals(preds, sigma)
        assert lower.shape == (10, 8, 8)
        assert upper.shape == (10, 8, 8)

    def test_intervals_widen_with_uncertainty(self):
        cp = SpatialConformalPredictor(alpha=0.1)
        cp.q_hat = 2.0
        preds = np.zeros((5, 8, 8))
        lo1, hi1 = cp.predict_intervals(preds, np.ones((5, 8, 8)) * 0.1)
        lo2, hi2 = cp.predict_intervals(preds, np.ones((5, 8, 8)) * 1.0)
        assert (hi2 - lo2).mean() > (hi1 - lo1).mean()

    def test_uncalibrated_raises(self):
        cp = SpatialConformalPredictor(alpha=0.1)
        with pytest.raises(RuntimeError, match="calibrate"):
            cp.predict_intervals(np.zeros((1, 8, 8)), np.ones((1, 8, 8)))

    def test_zero_uncertainty_handled(self):
        cp = SpatialConformalPredictor(alpha=0.1)
        cp.q_hat = 2.0
        preds = np.zeros((5, 8, 8))
        sigma = np.zeros((5, 8, 8))
        lower, upper = cp.predict_intervals(preds, sigma)
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))


class TestPixelwiseConformalPredictor:
    def test_coverage_guarantee_empirical(self):
        np.random.seed(42)
        N_cal, N_test = 500, 1000
        preds_cal = np.random.randn(N_cal, 8, 8)
        sigma_cal = np.ones((N_cal, 8, 8))
        truth_cal = preds_cal + np.random.randn(N_cal, 8, 8)

        cp = PixelwiseConformalPredictor(alpha=0.1)
        cp.calibrate(preds_cal, sigma_cal, truth_cal)

        preds_test = np.random.randn(N_test, 8, 8)
        sigma_test = np.ones((N_test, 8, 8))
        truth_test = preds_test + np.random.randn(N_test, 8, 8)
        lower, upper = cp.predict_intervals(preds_test, sigma_test)

        coverage = ((truth_test >= lower) & (truth_test <= upper)).mean()
        assert coverage >= 0.89  # allow small finite-sample slack

    def test_tighter_than_spatial(self):
        """Pixelwise intervals should be tighter than spatial."""
        np.random.seed(42)
        N = 200
        preds = np.random.randn(N, 8, 8)
        sigma = np.ones((N, 8, 8)) * 0.5
        truth = preds + 0.5 * np.random.randn(N, 8, 8)

        cp_spatial = SpatialConformalPredictor(alpha=0.1)
        cp_pixel = PixelwiseConformalPredictor(alpha=0.1)
        cp_spatial.calibrate(preds, sigma, truth)
        cp_pixel.calibrate(preds, sigma, truth)

        assert cp_pixel.q_hat < cp_spatial.q_hat
