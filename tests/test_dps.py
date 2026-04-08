"""Tests for DPS sampler (§10)."""

import torch
import pytest

from diffphys.model.unet import ConditionalUNet
from diffphys.model.unconditional_ddpm import UnconditionalDDPM
from diffphys.model.dps_sampler import DPSSampler


@pytest.fixture
def small_uncond_ddpm():
    """Tiny unconditional DDPM for testing."""
    torch.manual_seed(0)
    unet = ConditionalUNet(in_ch=1, out_ch=1, base_ch=16,
                           ch_mult=[1, 2], time_emb_dim=32)
    return UnconditionalDDPM(unet, T=10, prediction="v",
                             min_snr_gamma=5.0, schedule="cosine")


@pytest.fixture
def identity_obs_operator():
    """Observation operator that extracts top-edge boundary."""
    def obs_op(x):
        # x: (B, 1, H, W) -> top edge: (B, W)
        return x[:, 0, 0, :]
    return obs_op


class TestDPSSampler:
    def test_output_shape(self, small_uncond_ddpm, identity_obs_operator):
        sampler = DPSSampler(small_uncond_ddpm, zeta_obs=0.1, zeta_pde=0.0)
        y_obs = torch.zeros(16)  # fake top-edge observations for 16x16
        samples = sampler.sample(y_obs, identity_obs_operator,
                                 n_samples=3, H=16, W=16)
        assert samples.shape == (3, 1, 16, 16)

    def test_output_finite(self, small_uncond_ddpm, identity_obs_operator):
        sampler = DPSSampler(small_uncond_ddpm, zeta_obs=0.1, zeta_pde=0.01)
        y_obs = torch.zeros(16)
        samples = sampler.sample(y_obs, identity_obs_operator,
                                 n_samples=2, H=16, W=16)
        assert torch.isfinite(samples).all()

    def test_zero_guidance_equals_unconditional(self, small_uncond_ddpm):
        """With zeta=0, DPS should match unconditional sampling."""
        sampler = DPSSampler(small_uncond_ddpm, zeta_obs=0.0, zeta_pde=0.0)
        dummy_op = lambda x: x[:, 0, 0, :]  # noqa: E731
        y_obs = torch.zeros(16)

        torch.manual_seed(42)
        dps_samples = sampler.sample(y_obs, dummy_op, n_samples=2, H=16, W=16)

        torch.manual_seed(42)
        uncond_samples = small_uncond_ddpm.sample(2, H=16, W=16)

        assert torch.allclose(dps_samples, uncond_samples, atol=1e-5)

    def test_gradient_through_x0_hat_is_nonzero(self, small_uncond_ddpm):
        """Sanity check: autograd works through Tweedie formula."""
        sampler = DPSSampler(small_uncond_ddpm, zeta_obs=1.0, zeta_pde=0.0)

        x_t = torch.randn(2, 1, 16, 16, requires_grad=True)
        t_batch = torch.full((2,), 5, dtype=torch.long)
        small_uncond_ddpm.schedule.to(x_t.device)

        pred = small_uncond_ddpm.model(x_t, t_batch)
        ab = small_uncond_ddpm.schedule.alpha_bars[5]
        while ab.dim() < x_t.dim():
            ab = ab.unsqueeze(-1)

        from diffphys.model.ddpm import recover_x0_from_v
        x0_hat = recover_x0_from_v(x_t, pred, ab)

        # Compute a loss on x0_hat and check gradient flows to x_t
        loss = x0_hat.sum()
        grad = torch.autograd.grad(loss, x_t)[0]
        assert grad.abs().sum() > 0

    def test_deterministic_with_fixed_seed(self, small_uncond_ddpm,
                                           identity_obs_operator):
        sampler = DPSSampler(small_uncond_ddpm, zeta_obs=0.1, zeta_pde=0.01)
        y_obs = torch.zeros(16)

        torch.manual_seed(99)
        s1 = sampler.sample(y_obs, identity_obs_operator,
                            n_samples=2, H=16, W=16)
        torch.manual_seed(99)
        s2 = sampler.sample(y_obs, identity_obs_operator,
                            n_samples=2, H=16, W=16)
        assert torch.allclose(s1, s2)

    def test_measurement_guidance_reduces_observation_error(
            self, small_uncond_ddpm, identity_obs_operator):
        """Higher zeta_obs should produce samples closer to observations."""
        y_obs = torch.ones(16) * 0.5  # target top edge = 0.5

        torch.manual_seed(7)
        sampler_weak = DPSSampler(small_uncond_ddpm, zeta_obs=0.01,
                                  zeta_pde=0.0)
        s_weak = sampler_weak.sample(y_obs, identity_obs_operator,
                                     n_samples=3, H=16, W=16)

        torch.manual_seed(7)
        sampler_strong = DPSSampler(small_uncond_ddpm, zeta_obs=10.0,
                                    zeta_pde=0.0)
        s_strong = sampler_strong.sample(y_obs, identity_obs_operator,
                                         n_samples=3, H=16, W=16)

        err_weak = (identity_obs_operator(s_weak) - y_obs).pow(2).mean()
        err_strong = (identity_obs_operator(s_strong) - y_obs).pow(2).mean()

        # Strong guidance should have smaller observation error
        assert err_strong < err_weak, (
            f"Strong guidance err {err_strong:.4f} >= weak {err_weak:.4f}"
        )

    def test_laplacian_loss_zero_for_harmonic(self):
        """A linear field satisfies Laplace's equation exactly."""
        # T(x,y) = x + y is harmonic (Laplacian = 0)
        H, W = 16, 16
        y = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
        x = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)
        field = (x + y).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        loss = DPSSampler._laplacian_loss(field)
        assert loss.item() < 1e-6

    def test_custom_guidance_schedule(self, small_uncond_ddpm,
                                      identity_obs_operator):
        """Custom schedule should be called and produce finite samples."""
        calls = []

        def custom_schedule(t, T):
            calls.append(t)
            return 0.05, 0.005

        sampler = DPSSampler(small_uncond_ddpm, guidance_schedule=custom_schedule)
        y_obs = torch.zeros(16)
        samples = sampler.sample(y_obs, identity_obs_operator,
                                 n_samples=1, H=16, W=16)
        assert len(calls) == small_uncond_ddpm.T
        assert torch.isfinite(samples).all()
