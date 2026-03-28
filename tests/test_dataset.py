"""Tests for PyTorch dataset wrapper."""

import numpy as np
import torch
import pytest
from diffphys.data.dataset import LaplacePDEDataset


@pytest.fixture
def tiny_npz(tmp_path):
    """Create a small .npz file for testing."""
    n, nx = 8, 16
    rng = np.random.default_rng(42)
    np.savez(
        tmp_path / "tiny.npz",
        fields=rng.standard_normal((n, nx, nx)).astype(np.float32),
        bc_top=rng.standard_normal((n, nx)).astype(np.float32),
        bc_bottom=rng.standard_normal((n, nx)).astype(np.float32),
        bc_left=rng.standard_normal((n, nx)).astype(np.float32),
        bc_right=rng.standard_normal((n, nx)).astype(np.float32),
    )
    return tmp_path / "tiny.npz"


class TestLaplacePDEDataset:
    def test_length(self, tiny_npz):
        ds = LaplacePDEDataset(tiny_npz)
        assert len(ds) == 8

    def test_getitem_shapes(self, tiny_npz):
        ds = LaplacePDEDataset(tiny_npz)
        cond, target = ds[0]
        assert cond.shape == (8, 16, 16)
        assert target.shape == (1, 16, 16)

    def test_dtypes(self, tiny_npz):
        ds = LaplacePDEDataset(tiny_npz)
        cond, target = ds[0]
        assert cond.dtype == torch.float32
        assert target.dtype == torch.float32

    def test_target_matches_field(self, tiny_npz):
        ds = LaplacePDEDataset(tiny_npz)
        data = np.load(tiny_npz)
        cond, target = ds[3]
        np.testing.assert_allclose(
            target.squeeze(0).numpy(), data["fields"][3], atol=1e-7
        )

    def test_conditioning_encodes_bcs(self, tiny_npz):
        """Channel 0 row should equal bc_top for that sample."""
        ds = LaplacePDEDataset(tiny_npz)
        data = np.load(tiny_npz)
        cond, _ = ds[2]
        np.testing.assert_allclose(
            cond[0, 0, :].numpy(), data["bc_top"][2], atol=1e-7
        )

    def test_masks_all_ones_by_default(self, tiny_npz):
        ds = LaplacePDEDataset(tiny_npz)
        cond, _ = ds[0]
        torch.testing.assert_close(cond[4:8], torch.ones(4, 16, 16))

    def test_dataloader_batching(self, tiny_npz):
        ds = LaplacePDEDataset(tiny_npz)
        loader = torch.utils.data.DataLoader(ds, batch_size=4)
        cond_batch, target_batch = next(iter(loader))
        assert cond_batch.shape == (4, 8, 16, 16)
        assert target_batch.shape == (4, 1, 16, 16)

    def test_sparse_regime_sets_masks(self, tiny_npz):
        """very-sparse regime should produce non-all-ones masks."""
        ds = LaplacePDEDataset(tiny_npz, regime="very-sparse")
        cond, _ = ds[0]
        assert cond.shape == (8, 16, 16)
        # Mask channels should not all be 1.0 (n_points=8 < nx=16)
        assert not torch.all(cond[4:8] == 1.0)

    def test_mixed_regime_returns_valid_output(self, tiny_npz):
        """Mixed regime should produce valid 8-channel conditioning."""
        ds = LaplacePDEDataset(tiny_npz, regime="mixed")
        cond, target = ds[0]
        assert cond.shape == (8, 16, 16)
        assert target.shape == (1, 16, 16)
        assert torch.isfinite(cond).all()
