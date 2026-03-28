"""Deep ensemble of U-Net regressors.

Train K independent U-Nets with different seeds, then aggregate
predictions at inference via mean and variance.
"""

import torch


class EnsemblePredictor:
    """Wraps K trained models for ensemble prediction.

    Args:
        models: List of trained nn.Module instances.
    """

    def __init__(self, models):
        self.models = models
        for m in self.models:
            m.eval()

    @torch.no_grad()
    def predict_all(self, x):
        """Return all member predictions. Shape: (K, B, C, H, W)."""
        preds = [m(x) for m in self.models]
        return torch.stack(preds, dim=0)

    @torch.no_grad()
    def predict(self, x):
        """Return (mean, variance) over ensemble members."""
        all_preds = self.predict_all(x)  # (K, B, C, H, W)
        mean = all_preds.mean(dim=0)
        var = all_preds.var(dim=0, correction=0)
        return mean, var
