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


    def test_quantile_is_exact_order_statistic(self):
        """q_hat must be an actual calibration score, not interpolated."""
        cp = SpatialConformalPredictor(alpha=0.4)
        # 5 samples, 1x1 spatial so sample_scores = raw scores
        preds = np.zeros((5, 1, 1))
        sigma = np.ones((5, 1, 1))
        # Ground truth chosen so |truth - pred|/sigma = [0, 1, 4, 8, 100]
        truth = np.array([0, 1, 4, 8, 100]).reshape(5, 1, 1).astype(float)
        q = cp.calibrate(preds, sigma, truth)
        # k = ceil(6 * 0.6) = 4 → 4th order statistic = sorted_scores[3] = 8
        assert q == 8.0, f"Expected exact order statistic 8, got {q}"


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


class TestConformalEvaluation:
    def test_evaluate_conformal_for_model_keys(self):
        """Should return spatial and pixelwise results at each target."""
        from diffphys.evaluation.evaluate_uq import evaluate_conformal_for_model

        rng = np.random.default_rng(42)
        N = 100
        cal_mean = rng.standard_normal((N, 8, 8))
        cal_std = np.abs(rng.standard_normal((N, 8, 8))) + 0.1
        cal_truth = cal_mean + 0.5 * rng.standard_normal((N, 8, 8))
        test_mean = rng.standard_normal((N, 8, 8))
        test_std = np.abs(rng.standard_normal((N, 8, 8))) + 0.1
        test_truth = test_mean + 0.5 * rng.standard_normal((N, 8, 8))

        results = evaluate_conformal_for_model(
            cal_mean, cal_std, cal_truth,
            test_mean, test_std, test_truth,
            targets=[0.50, 0.90, 0.95],
        )
        assert "raw_coverage_90" in results
        assert "spatial_90_coverage" in results
        assert "spatial_90_q_hat" in results
        assert "spatial_90_mean_width" in results
        assert "pixelwise_90_coverage" in results
        assert "pixelwise_90_q_hat" in results

    def test_conformal_improves_coverage(self):
        """Conformal should achieve >= target coverage on well-behaved data."""
        from diffphys.evaluation.evaluate_uq import evaluate_conformal_for_model

        rng = np.random.default_rng(123)
        N = 500
        cal_mean = rng.standard_normal((N, 8, 8))
        cal_std = np.ones((N, 8, 8))
        cal_truth = cal_mean + rng.standard_normal((N, 8, 8))
        test_mean = rng.standard_normal((N, 8, 8))
        test_std = np.ones((N, 8, 8))
        test_truth = test_mean + rng.standard_normal((N, 8, 8))

        results = evaluate_conformal_for_model(
            cal_mean, cal_std, cal_truth,
            test_mean, test_std, test_truth,
            targets=[0.90],
        )
        assert results["pixelwise_90_coverage"] >= 0.88


class TestCollectPredictions:
    def test_collect_ensemble_predictions_shapes(self):
        """Helper should return (N, H, W) numpy arrays."""
        from diffphys.evaluation.evaluate_uq import collect_ensemble_predictions
        import torch

        class FakeModel(torch.nn.Module):
            def __init__(self, val):
                super().__init__()
                self.val = val
            def forward(self, x):
                return torch.full((x.shape[0], 1, 8, 8), self.val)

        from diffphys.model.ensemble import EnsemblePredictor
        ensemble = EnsemblePredictor([FakeModel(1.0), FakeModel(3.0)])

        conds = torch.randn(10, 8, 8, 8)
        targets = torch.randn(10, 1, 8, 8)
        ds = torch.utils.data.TensorDataset(conds, targets)
        loader = torch.utils.data.DataLoader(ds, batch_size=4)

        mean, std, truth = collect_ensemble_predictions(ensemble, loader, "cpu")
        assert mean.shape == (10, 8, 8)
        assert std.shape == (10, 8, 8)
        assert truth.shape == (10, 8, 8)
        assert np.allclose(mean, 2.0, atol=1e-5)
