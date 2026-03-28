"""Tests for conditional DDPM."""

import torch
import pytest
from diffphys.model.ddpm import NoiseSchedule, DDPM
from diffphys.model.unet import ConditionalUNet


class TestNoiseSchedule:
    @pytest.fixture
    def schedule(self):
        return NoiseSchedule(T=200, beta_start=1e-4, beta_end=0.02)

    def test_beta_shape(self, schedule):
        assert schedule.betas.shape == (201,)  # index 0 unused, 1..T

    def test_beta_range(self, schedule):
        assert schedule.betas[1] == pytest.approx(1e-4, rel=1e-3)
        assert schedule.betas[200] == pytest.approx(0.02, rel=1e-3)

    def test_alpha_bar_monotone_decreasing(self, schedule):
        ab = schedule.alpha_bars[1:]
        assert (ab[1:] < ab[:-1]).all()

    def test_alpha_bar_range(self, schedule):
        assert schedule.alpha_bars[1] > 0.99  # nearly 1 at t=1
        assert schedule.alpha_bars[200] < 0.15  # small at t=T

    def test_add_noise_shape(self, schedule):
        x0 = torch.randn(4, 1, 16, 16)
        noise = torch.randn_like(x0)
        t = torch.tensor([1, 50, 100, 200])
        x_t = schedule.add_noise(x0, noise, t)
        assert x_t.shape == x0.shape

    def test_add_noise_t1_close_to_x0(self, schedule):
        """At t=1, noisy sample should be close to original."""
        x0 = torch.randn(1, 1, 16, 16)
        noise = torch.randn_like(x0)
        x_t = schedule.add_noise(x0, noise, torch.tensor([1]))
        assert (x_t - x0).abs().mean() < 0.1

    def test_add_noise_tT_close_to_noise(self, schedule):
        """At t=T, noisy sample should be mostly noise."""
        x0 = torch.zeros(1, 1, 16, 16)
        noise = torch.randn_like(x0)
        x_t = schedule.add_noise(x0, noise, torch.tensor([200]))
        # alpha_bar_T is small, so x_t ≈ sqrt(1 - alpha_bar_T) * noise ≈ noise
        correlation = torch.corrcoef(torch.stack([x_t.flatten(), noise.flatten()]))[0, 1]
        assert correlation > 0.9


class TestDDPM:
    @pytest.fixture
    def ddpm(self):
        model = ConditionalUNet(in_ch=9, out_ch=1, base_ch=8, ch_mult=(1, 2), time_emb_dim=64)
        return DDPM(model, T=20)  # small T for fast testing

    def test_training_step_returns_loss(self, ddpm):
        cond = torch.randn(2, 8, 16, 16)
        target = torch.randn(2, 1, 16, 16)
        loss = ddpm.training_step(cond, target)
        assert loss.shape == ()
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_training_step_gradient_flow(self, ddpm):
        cond = torch.randn(2, 8, 16, 16)
        target = torch.randn(2, 1, 16, 16)
        loss = ddpm.training_step(cond, target)
        loss.backward()
        for p in ddpm.model.parameters():
            assert p.grad is not None

    def test_sample_shape(self, ddpm):
        cond = torch.randn(2, 8, 16, 16)
        samples = ddpm.sample(cond, n_samples=3)
        assert samples.shape == (3, 2, 1, 16, 16)

    def test_sample_finite(self, ddpm):
        cond = torch.randn(1, 8, 16, 16)
        samples = ddpm.sample(cond, n_samples=2)
        assert torch.isfinite(samples).all()

    def test_sample_different_per_call(self, ddpm):
        """Different random seeds should give different samples."""
        cond = torch.randn(1, 8, 16, 16)
        s1 = ddpm.sample(cond, n_samples=1)
        s2 = ddpm.sample(cond, n_samples=1)
        assert not torch.allclose(s1, s2)
