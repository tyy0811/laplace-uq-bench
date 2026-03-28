"""Tests for 8-channel BC conditioning tensor encoding."""

import torch
import pytest
from diffphys.data.conditioning import encode_conditioning


class TestEncodeConditioning:
    @pytest.fixture
    def bcs(self):
        """Sample BCs as 1D tensors of length 64."""
        torch.manual_seed(42)
        return tuple(torch.randn(64) for _ in range(4))

    def test_output_shape(self, bcs):
        cond = encode_conditioning(*bcs)
        assert cond.shape == (8, 64, 64)

    def test_dtype_float32(self, bcs):
        cond = encode_conditioning(*bcs)
        assert cond.dtype == torch.float32

    def test_top_broadcast_along_rows(self, bcs):
        """Channel 0: every row should equal bc_top."""
        bc_top = bcs[0]
        cond = encode_conditioning(*bcs)
        for i in range(64):
            torch.testing.assert_close(cond[0, i, :], bc_top)

    def test_bottom_broadcast_along_rows(self, bcs):
        """Channel 1: every row should equal bc_bottom."""
        bc_bottom = bcs[1]
        cond = encode_conditioning(*bcs)
        for i in range(64):
            torch.testing.assert_close(cond[1, i, :], bc_bottom)

    def test_left_broadcast_along_cols(self, bcs):
        """Channel 2: every column should equal bc_left."""
        bc_left = bcs[2]
        cond = encode_conditioning(*bcs)
        for j in range(64):
            torch.testing.assert_close(cond[2, :, j], bc_left)

    def test_right_broadcast_along_cols(self, bcs):
        """Channel 3: every column should equal bc_right."""
        bc_right = bcs[3]
        cond = encode_conditioning(*bcs)
        for j in range(64):
            torch.testing.assert_close(cond[3, :, j], bc_right)

    def test_masks_default_all_ones(self, bcs):
        """Phase 1: channels 4-7 are all 1.0."""
        cond = encode_conditioning(*bcs)
        torch.testing.assert_close(cond[4:8], torch.ones(4, 64, 64))

    def test_custom_masks(self, bcs):
        """Channels 4-7 should broadcast masks like values."""
        mask_top = torch.zeros(64)
        mask_top[::4] = 1.0  # every 4th point observed
        masks = (mask_top, torch.ones(64), torch.ones(64), torch.ones(64))
        cond = encode_conditioning(*bcs, *masks)
        # Mask channel 4 should broadcast top mask along rows
        for i in range(64):
            torch.testing.assert_close(cond[4, i, :], mask_top)

    def test_different_nx(self):
        """Should work with non-default grid sizes."""
        bcs = tuple(torch.randn(16) for _ in range(4))
        cond = encode_conditioning(*bcs)
        assert cond.shape == (8, 16, 16)

    def test_partial_masks_raises(self, bcs):
        """Passing only some masks should raise ValueError."""
        with pytest.raises(ValueError, match="all 4 masks or none"):
            encode_conditioning(*bcs, mask_top=torch.ones(64))
