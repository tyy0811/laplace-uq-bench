"""Tests for unconditional DDPM (prior model for DPS)."""

import numpy as np
import torch
import pytest

from diffphys.model.unet import ConditionalUNet
from diffphys.model.unconditional_ddpm import UnconditionalDDPM


@pytest.fixture
def small_model():
    """Small unconditional DDPM for testing."""
    unet = ConditionalUNet(in_ch=1, out_ch=1, base_ch=16,
                           ch_mult=[1, 2], time_emb_dim=32)
    return UnconditionalDDPM(unet, T=20, prediction="v", min_snr_gamma=5.0,
                             schedule="cosine")


@pytest.fixture
def tiny_fields():
    """10 random 16x16 fields for overfit test."""
    torch.manual_seed(42)
    return torch.randn(10, 1, 16, 16)


class TestUnconditionalDDPM:
    def test_training_step_returns_scalar(self, small_model, tiny_fields):
        loss = small_model.training_step(tiny_fields[:4])
        assert loss.shape == ()
        assert loss.item() > 0
        assert np.isfinite(loss.item())

    def test_training_step_gradient_flows(self, small_model, tiny_fields):
        loss = small_model.training_step(tiny_fields[:4])
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in small_model.parameters())
        assert has_grad

    def test_sample_output_shape(self, small_model):
        samples = small_model.sample(3, H=16, W=16)
        assert samples.shape == (3, 1, 16, 16)

    def test_sample_deterministic_with_seed(self, small_model):
        torch.manual_seed(0)
        s1 = small_model.sample(2, H=16, W=16)
        torch.manual_seed(0)
        s2 = small_model.sample(2, H=16, W=16)
        assert torch.allclose(s1, s2)

    def test_overfits_small_data(self, tiny_fields):
        """Deterministic overfit test: fixed seed, 10 samples, 20 epochs."""
        torch.manual_seed(123)
        unet = ConditionalUNet(in_ch=1, out_ch=1, base_ch=16,
                               ch_mult=[1, 2], time_emb_dim=32)
        ddpm = UnconditionalDDPM(unet, T=20, prediction="v",
                                 min_snr_gamma=5.0, schedule="cosine")
        optimizer = torch.optim.Adam(ddpm.parameters(), lr=1e-3)

        initial_loss = None
        for epoch in range(20):
            ddpm.train()
            loss = ddpm.training_step(tiny_fields)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if initial_loss is None:
                initial_loss = loss.item()

        final_loss = loss.item()
        assert final_loss < 0.5 * initial_loss, (
            f"Loss did not decrease enough: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_epsilon_prediction_mode(self):
        unet = ConditionalUNet(in_ch=1, out_ch=1, base_ch=16,
                               ch_mult=[1, 2], time_emb_dim=32)
        ddpm = UnconditionalDDPM(unet, T=20, prediction="epsilon",
                                 min_snr_gamma=None, schedule="linear")
        target = torch.randn(4, 1, 16, 16)
        loss = ddpm.training_step(target)
        assert loss.shape == ()
        samples = ddpm.sample(2, H=16, W=16)
        assert samples.shape == (2, 1, 16, 16)
