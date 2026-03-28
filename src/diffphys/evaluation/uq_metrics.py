"""Uncertainty quantification metrics for Phase 2 evaluation.

All functions expect (B, 1, H, W) tensors for true, mean, std.
"""

import math

import torch


def pixelwise_coverage(true, mean, std, level=0.95):
    """Fraction of pixels where true value falls within the prediction interval.

    Uses Gaussian assumption: interval = mean +/- z * std.
    """
    z = torch.distributions.Normal(0, 1).icdf(torch.tensor(0.5 + level / 2))
    lower = mean - z * std
    upper = mean + z * std
    covered = ((true >= lower) & (true <= upper)).float()
    return covered.mean()


def crps_gaussian(true, mean, std):
    """Continuous Ranked Probability Score for Gaussian predictive distribution.

    CRPS(N(mu, sigma^2), y) = sigma * [z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]
    where z = (y - mu) / sigma, Phi = CDF, phi = PDF.

    Returns per-sample mean CRPS, shape (B,).
    """
    z = (true - mean) / std.clamp(min=1e-8)
    normal = torch.distributions.Normal(0, 1)
    crps_pixel = std * (
        z * (2 * normal.cdf(z) - 1)
        + 2 * normal.log_prob(z).exp()
        - 1.0 / math.sqrt(math.pi)
    )
    return crps_pixel.reshape(true.shape[0], -1).mean(dim=1)


def calibration_error(true, mean, std, n_bins=10):
    """Mean absolute calibration error across quantile levels.

    For each nominal level p in [0.1, ..., 0.9], compute empirical
    coverage and return mean |empirical - nominal|.
    """
    levels = torch.linspace(0.1, 0.9, n_bins)
    errors = []
    for p in levels:
        empirical = pixelwise_coverage(true, mean, std, level=p.item())
        errors.append((empirical - p).abs())
    return torch.stack(errors).mean()


def sharpness(std):
    """Mean prediction interval width (proportional to mean std)."""
    return std.mean()
