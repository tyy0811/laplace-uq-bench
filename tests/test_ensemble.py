"""Tests for deep ensemble inference."""

import torch
import pytest
from diffphys.model.ensemble import EnsemblePredictor
from diffphys.model.unet import ConditionalUNet


class TestEnsemblePredictor:
    @pytest.fixture
    def ensemble(self):
        """3 tiny U-Nets (instead of 5) for fast testing."""
        models = []
        for seed in range(3):
            torch.manual_seed(seed)
            models.append(ConditionalUNet(in_ch=8, out_ch=1, base_ch=8, ch_mult=(1, 2)))
        return EnsemblePredictor(models)

    def test_mean_shape(self, ensemble):
        x = torch.randn(2, 8, 16, 16)
        mean, var = ensemble.predict(x)
        assert mean.shape == (2, 1, 16, 16)

    def test_variance_shape(self, ensemble):
        x = torch.randn(2, 8, 16, 16)
        mean, var = ensemble.predict(x)
        assert var.shape == (2, 1, 16, 16)

    def test_variance_non_negative(self, ensemble):
        x = torch.randn(2, 8, 16, 16)
        _, var = ensemble.predict(x)
        assert (var >= 0).all()

    def test_mean_is_average_of_members(self, ensemble):
        x = torch.randn(1, 8, 16, 16)
        mean, _ = ensemble.predict(x)
        # Manually compute
        preds = [m(x) for m in ensemble.models]
        manual_mean = torch.stack(preds).mean(dim=0)
        torch.testing.assert_close(mean, manual_mean)

    def test_single_model_zero_variance(self):
        """Ensemble of 1 should have zero variance."""
        model = ConditionalUNet(in_ch=8, out_ch=1, base_ch=8, ch_mult=(1, 2))
        ens = EnsemblePredictor([model])
        x = torch.randn(1, 8, 16, 16)
        _, var = ens.predict(x)
        torch.testing.assert_close(var, torch.zeros_like(var), atol=1e-7, rtol=0)

    def test_get_all_predictions(self, ensemble):
        x = torch.randn(2, 8, 16, 16)
        all_preds = ensemble.predict_all(x)
        assert all_preds.shape == (3, 2, 1, 16, 16)
