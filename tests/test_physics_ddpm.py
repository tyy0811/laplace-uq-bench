"""Tests for physics-informed DDPM."""

import torch
import pytest
from diffphys.model.physics_ddpm import PhysicsDDPM
from diffphys.model.unet import ConditionalUNet


class TestPhysicsDDPM:
    @pytest.fixture
    def phys_ddpm(self):
        model = ConditionalUNet(in_ch=9, out_ch=1, base_ch=8, ch_mult=(1, 2), time_emb_dim=64)
        return PhysicsDDPM(model, T=20, residual_weight=0.1)

    def test_training_step_returns_dict(self, phys_ddpm):
        cond = torch.randn(2, 8, 16, 16)
        target = torch.randn(2, 1, 16, 16)
        losses = phys_ddpm.training_step(cond, target)
        assert "total" in losses
        assert "mse" in losses
        assert "residual" in losses

    def test_residual_loss_is_finite(self, phys_ddpm):
        cond = torch.randn(2, 8, 16, 16)
        target = torch.randn(2, 1, 16, 16)
        losses = phys_ddpm.training_step(cond, target)
        assert torch.isfinite(losses["total"])
        assert torch.isfinite(losses["residual"])

    def test_total_loss_includes_residual(self, phys_ddpm):
        cond = torch.randn(2, 8, 16, 16)
        target = torch.randn(2, 1, 16, 16)
        losses = phys_ddpm.training_step(cond, target)
        assert losses["total"] >= losses["mse"]

    def test_residual_weight_scales_contribution(self):
        model = ConditionalUNet(in_ch=9, out_ch=1, base_ch=8, ch_mult=(1, 2), time_emb_dim=64)
        low = PhysicsDDPM(model, T=20, residual_weight=0.01)
        high = PhysicsDDPM(model, T=20, residual_weight=10.0)
        cond = torch.randn(2, 8, 16, 16)
        target = torch.randn(2, 1, 16, 16)
        torch.manual_seed(42)
        l_low = low.training_step(cond, target)
        torch.manual_seed(42)
        l_high = high.training_step(cond, target)
        # Higher weight -> higher residual contribution
        assert l_high["total"] > l_low["total"] or l_low["residual"].item() < 1e-8

    def test_sampling_still_works(self, phys_ddpm):
        """Physics loss is training-only; sampling should be unchanged."""
        cond = torch.randn(1, 8, 16, 16)
        samples = phys_ddpm.sample(cond, n_samples=2)
        assert samples.shape == (2, 1, 1, 16, 16)
