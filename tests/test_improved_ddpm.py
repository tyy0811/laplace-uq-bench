"""Tests for improved DDPM (cosine schedule, v-prediction, Min-SNR)."""

import torch
import pytest
from diffphys.model.ddpm import (
    NoiseSchedule,
    cosine_beta_schedule,
    compute_min_snr_weight,
    compute_v_target,
    recover_x0_from_v,
    ImprovedDDPM,
)
from diffphys.model.unet import ConditionalUNet


class TestCosineSchedule:
    def test_alpha_bar_monotone_decreasing(self):
        betas = cosine_beta_schedule(200)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        assert (alpha_bars[1:] <= alpha_bars[:-1]).all()

    def test_zero_terminal_snr(self):
        betas = cosine_beta_schedule(200)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        assert alpha_bars[-1] < 1e-4

    def test_noise_schedule_cosine_mode(self):
        schedule = NoiseSchedule(T=200, schedule="cosine")
        assert schedule.alpha_bars[200] < 1e-4


class TestMinSNR:
    def test_output_shape(self):
        schedule = NoiseSchedule(T=200)
        t = torch.randint(1, 201, (8,))
        weights = compute_min_snr_weight(schedule.alpha_bars, t, gamma=5.0)
        assert weights.shape == (8,)

    def test_weights_bounded(self):
        schedule = NoiseSchedule(T=200)
        t = torch.randint(1, 201, (100,))
        weights = compute_min_snr_weight(schedule.alpha_bars, t, gamma=5.0)
        assert (weights > 0).all()
        assert (weights <= 1.0).all()

    def test_high_noise_downweighted(self):
        schedule = NoiseSchedule(T=200)
        t_low = torch.tensor([10])
        t_high = torch.tensor([190])
        w_low = compute_min_snr_weight(schedule.alpha_bars, t_low)
        w_high = compute_min_snr_weight(schedule.alpha_bars, t_high)
        assert w_high < w_low


class TestVPrediction:
    def test_roundtrip(self):
        x_0 = torch.randn(4, 1, 8, 8)
        noise = torch.randn_like(x_0)
        ab = torch.tensor([0.9, 0.5, 0.2, 0.05]).view(4, 1, 1, 1)
        x_t = ab.sqrt() * x_0 + (1 - ab).sqrt() * noise
        v = compute_v_target(x_0, noise, ab)
        x_0_recovered = recover_x0_from_v(x_t, v, ab)
        torch.testing.assert_close(x_0_recovered, x_0, atol=1e-5, rtol=1e-5)


class TestImprovedDDPM:
    @pytest.fixture
    def iddpm(self):
        model = ConditionalUNet(in_ch=9, out_ch=1, base_ch=8, ch_mult=(1, 2), time_emb_dim=64)
        return ImprovedDDPM(model, T=20, prediction="v", min_snr_gamma=5.0)

    def test_training_step_returns_loss(self, iddpm):
        cond = torch.randn(2, 8, 16, 16)
        target = torch.randn(2, 1, 16, 16)
        loss = iddpm.training_step(cond, target)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_training_step_gradient_flow(self, iddpm):
        cond = torch.randn(2, 8, 16, 16)
        target = torch.randn(2, 1, 16, 16)
        loss = iddpm.training_step(cond, target)
        loss.backward()
        for p in iddpm.model.parameters():
            assert p.grad is not None

    def test_sample_shape(self, iddpm):
        cond = torch.randn(2, 8, 16, 16)
        samples = iddpm.sample(cond, n_samples=3)
        assert samples.shape == (3, 2, 1, 16, 16)

    def test_sample_finite(self, iddpm):
        cond = torch.randn(1, 8, 16, 16)
        samples = iddpm.sample(cond, n_samples=2)
        assert torch.isfinite(samples).all()

    def test_epsilon_mode_still_works(self):
        model = ConditionalUNet(in_ch=9, out_ch=1, base_ch=8, ch_mult=(1, 2), time_emb_dim=64)
        iddpm = ImprovedDDPM(model, T=20, prediction="epsilon", min_snr_gamma=5.0)
        cond = torch.randn(2, 8, 16, 16)
        target = torch.randn(2, 1, 16, 16)
        loss = iddpm.training_step(cond, target)
        assert torch.isfinite(loss)
