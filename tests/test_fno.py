"""Tests for the Fourier Neural Operator."""

import torch
import pytest
from diffphys.model.fno import FNO2d


class TestFNO2d:
    def test_forward_shape(self):
        model = FNO2d(in_ch=8, out_ch=1, width=40, modes=12, n_layers=4)
        x = torch.randn(2, 8, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)

    def test_param_count(self):
        model = FNO2d(in_ch=8, out_ch=1, width=40, modes=12, n_layers=4)
        n_params = sum(p.numel() for p in model.parameters())
        assert 1_000_000 < n_params < 3_000_000

    def test_gradient_flow(self):
        model = FNO2d(in_ch=8, out_ch=1, width=40, modes=12, n_layers=4)
        x = torch.randn(2, 8, 64, 64)
        loss = model(x).sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_different_grid_sizes(self):
        model = FNO2d(in_ch=8, out_ch=1, width=20, modes=6, n_layers=2)
        for nx in [16, 32, 64]:
            x = torch.randn(1, 8, nx, nx)
            out = model(x)
            assert out.shape == (1, 1, nx, nx)

    def test_deterministic_eval(self):
        model = FNO2d(in_ch=8, out_ch=1, width=20, modes=6, n_layers=2)
        model.eval()
        x = torch.randn(1, 8, 32, 32)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)

    def test_modes_exceeding_grid_raises(self):
        """Modes larger than FFT dimensions should raise ValueError."""
        model = FNO2d(in_ch=8, out_ch=1, width=20, modes=12, n_layers=2)
        x = torch.randn(1, 8, 8, 8)  # rfft_w=5, modes=12 > 5
        with pytest.raises(ValueError, match="exceed grid FFT dimensions"):
            model(x)
