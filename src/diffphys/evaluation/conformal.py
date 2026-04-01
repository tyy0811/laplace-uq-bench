"""Conformal prediction for PDE surrogate uncertainty quantification.

Provides distribution-free coverage guarantees on top of any base model
that produces predictions with uncertainty estimates. Based on split
conformal prediction (Vovk et al., 2005; Angelopoulos & Bates, 2021).
"""

import numpy as np
from typing import Tuple


class SpatialConformalPredictor:
    """Conformal prediction with spatial (simultaneous) coverage.

    Uses max nonconformity score across all pixels per sample,
    guaranteeing that prediction bands cover ALL pixels simultaneously.

    Usage:
        1. Get ensemble predictions on calibration set (held-out from val)
        2. Call .calibrate() with predictions, uncertainties, ground truth
        3. Call .predict_intervals() on new test inputs
    """

    def __init__(self, alpha=0.1):
        """Args: alpha: Miscoverage rate. 0.1 -> 90% coverage guarantee."""
        self.alpha = alpha
        self.q_hat = None

    def calibrate(self, cal_predictions, cal_uncertainties, cal_ground_truth):
        """Calibrate on held-out data.

        Args:
            cal_predictions: (N, H, W) ensemble mean
            cal_uncertainties: (N, H, W) ensemble std
            cal_ground_truth: (N, H, W) true solution

        Returns:
            q_hat: calibrated quantile threshold
        """
        N = cal_predictions.shape[0]
        residuals = np.abs(cal_ground_truth - cal_predictions)
        sigma_safe = np.maximum(cal_uncertainties, 1e-8)
        scores = residuals / sigma_safe
        sample_scores = scores.max(axis=(1, 2))  # spatial max per sample

        # Exact order statistic — no interpolation — for finite-sample guarantee
        # k is 1-indexed: the k-th smallest score
        k = int(np.ceil((N + 1) * (1 - self.alpha)))
        sorted_scores = np.sort(sample_scores)
        if k > N:
            self.q_hat = float("inf")
        else:
            self.q_hat = float(sorted_scores[k - 1])
        return self.q_hat

    def predict_intervals(self, predictions, uncertainties):
        """Compute conformalized prediction intervals.

        Args:
            predictions: (B, H, W) point predictions
            uncertainties: (B, H, W) uncertainty estimates (std)

        Returns:
            (lower, upper): each (B, H, W)
        """
        if self.q_hat is None:
            raise RuntimeError("Must call calibrate() before predict_intervals()")
        sigma_safe = np.maximum(uncertainties, 1e-8)
        half_width = self.q_hat * sigma_safe
        return predictions - half_width, predictions + half_width


class PixelwiseConformalPredictor:
    """Conformal prediction with marginal (per-pixel) coverage.

    Tighter intervals but only guarantees coverage at a random pixel,
    not simultaneously across the full field.
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q_hat = None

    def calibrate(self, cal_predictions, cal_uncertainties, cal_ground_truth):
        """Calibrate using all pixels from all calibration samples."""
        residuals = np.abs(cal_ground_truth - cal_predictions)
        sigma_safe = np.maximum(cal_uncertainties, 1e-8)
        scores = (residuals / sigma_safe).flatten()

        N = len(scores)
        # Exact order statistic — no interpolation — for finite-sample guarantee
        # k is 1-indexed: the k-th smallest score
        k = int(np.ceil((N + 1) * (1 - self.alpha)))
        sorted_scores = np.sort(scores)
        if k > N:
            self.q_hat = float("inf")
        else:
            self.q_hat = float(sorted_scores[k - 1])
        return self.q_hat

    def predict_intervals(self, predictions, uncertainties):
        if self.q_hat is None:
            raise RuntimeError("Must call calibrate() before predict_intervals()")
        sigma_safe = np.maximum(uncertainties, 1e-8)
        half_width = self.q_hat * sigma_safe
        return predictions - half_width, predictions + half_width
