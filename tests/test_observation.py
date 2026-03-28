"""Tests for observation regime transformations."""

import torch
import pytest
from diffphys.data.observation import apply_observation_regime, REGIMES


class TestObservationRegimes:
    @pytest.fixture
    def bc(self):
        torch.manual_seed(42)
        return torch.randn(64)

    def test_exact_returns_original(self, bc):
        obs, mask = apply_observation_regime(bc, "exact")
        torch.testing.assert_close(obs, bc)
        torch.testing.assert_close(mask, torch.ones(64))

    def test_dense_noisy_adds_noise(self, bc):
        obs, mask = apply_observation_regime(bc, "dense-noisy", rng=torch.Generator().manual_seed(0))
        assert not torch.allclose(obs, bc)
        # All points observed
        torch.testing.assert_close(mask, torch.ones(64))
        # Noise should be moderate
        assert (obs - bc).abs().mean() < 0.5

    def test_sparse_clean_mask_has_16_ones(self, bc):
        obs, mask = apply_observation_regime(bc, "sparse-clean")
        assert mask.sum().item() == 16

    def test_sparse_clean_no_noise(self, bc):
        obs, mask = apply_observation_regime(bc, "sparse-clean")
        # At observed positions, values should match exactly
        observed_idx = mask.bool()
        torch.testing.assert_close(obs[observed_idx], bc[observed_idx])

    def test_sparse_noisy_mask_has_16_ones(self, bc):
        obs, mask = apply_observation_regime(bc, "sparse-noisy", rng=torch.Generator().manual_seed(0))
        assert mask.sum().item() == 16

    def test_very_sparse_mask_has_8_ones(self, bc):
        obs, mask = apply_observation_regime(bc, "very-sparse", rng=torch.Generator().manual_seed(0))
        assert mask.sum().item() == 8

    def test_interpolation_preserves_endpoints(self, bc):
        """Sparse regimes should preserve first and last points."""
        obs, mask = apply_observation_regime(bc, "sparse-clean")
        assert mask[0].item() == 1.0
        assert mask[-1].item() == 1.0
        torch.testing.assert_close(obs[0], bc[0])
        torch.testing.assert_close(obs[-1], bc[-1])

    def test_output_shapes(self, bc):
        for regime in REGIMES:
            obs, mask = apply_observation_regime(bc, regime, rng=torch.Generator().manual_seed(0))
            assert obs.shape == (64,)
            assert mask.shape == (64,)

    def test_unknown_regime_raises(self, bc):
        with pytest.raises(ValueError, match="Unknown regime"):
            apply_observation_regime(bc, "unknown")

    def test_all_outputs_finite(self, bc):
        for regime in REGIMES:
            obs, mask = apply_observation_regime(bc, regime, rng=torch.Generator().manual_seed(0))
            assert torch.isfinite(obs).all()
            assert torch.isfinite(mask).all()
