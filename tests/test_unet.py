"""Tests for the conditional U-Net architecture."""

import torch
import pytest
from diffphys.model.unet import ConditionalUNet


class TestConditionalUNet:
    def test_regressor_forward_shape(self):
        """Regressor mode: 8 input channels, 1 output channel."""
        model = ConditionalUNet(in_ch=8, out_ch=1)
        x = torch.randn(2, 8, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)

    def test_ddpm_forward_shape(self):
        """DDPM mode: 9 input channels + time embedding."""
        model = ConditionalUNet(in_ch=9, out_ch=1, time_emb_dim=256)
        x = torch.randn(2, 9, 64, 64)
        t = torch.tensor([10, 50])
        out = model(x, t)
        assert out.shape == (2, 1, 64, 64)

    def test_param_count_regressor(self):
        """Regressor should be ~9.4M params (2 ResBlocks per level)."""
        model = ConditionalUNet(in_ch=8, out_ch=1)
        n_params = sum(p.numel() for p in model.parameters())
        assert 8_000_000 < n_params < 11_000_000

    def test_param_count_ddpm(self):
        """DDPM should be slightly more than regressor."""
        reg = ConditionalUNet(in_ch=8, out_ch=1)
        ddpm = ConditionalUNet(in_ch=9, out_ch=1, time_emb_dim=256)
        n_reg = sum(p.numel() for p in reg.parameters())
        n_ddpm = sum(p.numel() for p in ddpm.parameters())
        assert n_ddpm > n_reg

    def test_gradient_flow(self):
        """All parameters should receive gradients."""
        model = ConditionalUNet(in_ch=8, out_ch=1)
        x = torch.randn(2, 8, 64, 64)
        loss = model(x).sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_ddpm_gradient_flow(self):
        """DDPM U-Net: time embedding params should also get gradients."""
        model = ConditionalUNet(in_ch=9, out_ch=1, time_emb_dim=256)
        x = torch.randn(2, 9, 64, 64)
        t = torch.tensor([10, 50])
        loss = model(x, t).sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_different_grid_sizes(self):
        """Should work with any power-of-2 grid that's >= 8."""
        for nx in [16, 32, 64]:
            model = ConditionalUNet(in_ch=8, out_ch=1)
            x = torch.randn(1, 8, nx, nx)
            out = model(x)
            assert out.shape == (1, 1, nx, nx)

    def test_deterministic_eval(self):
        """Same input should give same output in eval mode."""
        model = ConditionalUNet(in_ch=8, out_ch=1)
        model.eval()
        x = torch.randn(1, 8, 64, 64)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)

    def test_without_time_emb_ignores_t(self):
        """Regressor U-Net should work with or without t argument."""
        model = ConditionalUNet(in_ch=8, out_ch=1)
        x = torch.randn(1, 8, 64, 64)
        out1 = model(x)
        out2 = model(x, None)
        torch.testing.assert_close(out1, out2)
