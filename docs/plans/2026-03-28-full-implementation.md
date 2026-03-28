# Full Implementation Plan: Generative Surrogates for PDE Fields

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete benchmark comparing deterministic U-Net, FNO, deep ensemble, and conditional DDPM surrogates for 2D Laplace equation under exact and noisy boundary observations, evaluated with physics-aware metrics.

**Architecture:** PyTorch models operating on 8-channel BC conditioning tensors (4 value + 4 mask channels). U-Net backbone (~5M params) shared between regressor, ensemble, and DDPM. FNO (~2M params) as operator-learning baseline. DDPM uses T=200 linear beta schedule with epsilon prediction. Training on Modal A100; evaluation on CPU.

**Tech Stack:** Python 3.9+, PyTorch, NumPy, SciPy, matplotlib, PyYAML, Modal (remote GPU)

**Reference:** See `docs/plans/2026-03-27-design.md` for full project design.

**What's already done (Day 1):**
- LU-factorized Laplace solver (src/diffphys/pde/laplace.py)
- 5 BC families with corner consistency (src/diffphys/pde/boundary.py)
- Dataset generation with held-out OOD split (src/diffphys/pde/generate.py)
- Generated data: train (40K), val (5K), test_in (5K), test_ood (1K)
- 54 tests passing

**End-of-project targets:**
- 4 trained models evaluated on 2 test splits + 5 observation regimes
- Tables 1-7 from design doc populated
- 120+ tests passing

---

### Task 2: PyTorch Dataset + 8-Channel Conditioning

**Files:**
- Create: `src/diffphys/data/conditioning.py`
- Create: `src/diffphys/data/dataset.py`
- Create: `tests/test_conditioning.py`
- Create: `tests/test_dataset.py`

**Context:** All models consume the same 8-channel conditioning tensor: channels 0-3 are BC values broadcast along the perpendicular axis (top/bottom broadcast vertically, left/right horizontally), channels 4-7 are observation masks with the same broadcast pattern. Phase 1 masks are all 1.0. The dataset loads .npz files and returns (conditioning, target_field) pairs.

**Step 1: Write conditioning tests**

Create `tests/test_conditioning.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_conditioning.py -v
```

Expected: `ERROR` -- `ModuleNotFoundError: No module named 'diffphys.data.conditioning'`

**Step 3: Write conditioning implementation**

Create `src/diffphys/data/conditioning.py`:

```python
"""8-channel BC conditioning tensor encoding.

Channels 0-3: BC values broadcast along perpendicular axis.
  0 (top):    bc_top[j] broadcast to every row i
  1 (bottom): bc_bottom[j] broadcast to every row i
  2 (left):   bc_left[i] broadcast to every column j
  3 (right):  bc_right[i] broadcast to every column j

Channels 4-7: observation masks with same broadcast pattern.
  Phase 1: all 1.0 (fully observed exact BCs).
  Phase 2: 1.0 at observed positions, 0.0 at interpolated.
"""

import torch


def encode_conditioning(
    bc_top, bc_bottom, bc_left, bc_right,
    mask_top=None, mask_bottom=None, mask_left=None, mask_right=None,
):
    """Build (8, nx, nx) conditioning tensor from boundary arrays.

    Args:
        bc_top, bc_bottom: (nx,) tensors, broadcast along rows.
        bc_left, bc_right: (nx,) tensors, broadcast along columns.
        mask_*: optional (nx,) tensors; default all 1.0.

    Returns:
        (8, nx, nx) float32 tensor.
    """
    nx = bc_top.shape[0]
    cond = torch.zeros(8, nx, nx, dtype=torch.float32)

    # Value channels
    cond[0] = bc_top.unsqueeze(0).expand(nx, nx)
    cond[1] = bc_bottom.unsqueeze(0).expand(nx, nx)
    cond[2] = bc_left.unsqueeze(1).expand(nx, nx)
    cond[3] = bc_right.unsqueeze(1).expand(nx, nx)

    # Mask channels
    if mask_top is None:
        cond[4:8] = 1.0
    else:
        cond[4] = mask_top.unsqueeze(0).expand(nx, nx)
        cond[5] = mask_bottom.unsqueeze(0).expand(nx, nx)
        cond[6] = mask_left.unsqueeze(1).expand(nx, nx)
        cond[7] = mask_right.unsqueeze(1).expand(nx, nx)

    return cond
```

**Step 4: Run conditioning tests**

```bash
pytest tests/test_conditioning.py -v
```

Expected: all passed

**Step 5: Write dataset tests**

Create `tests/test_dataset.py`:

```python
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
```

**Step 6: Write dataset implementation**

Create `src/diffphys/data/dataset.py`:

```python
"""PyTorch dataset for Laplace PDE solution fields."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .conditioning import encode_conditioning


class LaplacePDEDataset(Dataset):
    """Loads .npz files produced by diffphys.pde.generate.

    Each sample returns (conditioning, target) where:
      conditioning: (8, nx, nx) float32 tensor
      target: (1, nx, nx) float32 tensor
    """

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.fields = torch.from_numpy(data["fields"])      # (N, nx, nx)
        self.bc_top = torch.from_numpy(data["bc_top"])       # (N, nx)
        self.bc_bottom = torch.from_numpy(data["bc_bottom"])
        self.bc_left = torch.from_numpy(data["bc_left"])
        self.bc_right = torch.from_numpy(data["bc_right"])

    def __len__(self):
        return self.fields.shape[0]

    def __getitem__(self, idx):
        cond = encode_conditioning(
            self.bc_top[idx], self.bc_bottom[idx],
            self.bc_left[idx], self.bc_right[idx],
        )
        target = self.fields[idx].unsqueeze(0)  # (1, nx, nx)
        return cond, target
```

**Step 7: Run all data tests**

```bash
pytest tests/test_conditioning.py tests/test_dataset.py -v
```

Expected: all passed

**Step 8: Commit**

```bash
git add src/diffphys/data/conditioning.py src/diffphys/data/dataset.py tests/test_conditioning.py tests/test_dataset.py
git commit -m "feat: 8-channel BC conditioning and PyTorch dataset

Conditioning tensor: 4 value channels (BC broadcast along perpendicular
axis) + 4 mask channels (all 1.0 for Phase 1). Dataset wraps .npz files
from generate.py."
```

---

### Task 3: U-Net Architecture

**Files:**
- Create: `src/diffphys/model/unet.py`
- Create: `tests/test_unet.py`

**Context:** The U-Net is the backbone for the deterministic regressor (in_ch=8), the ensemble members (in_ch=8), and the DDPM (in_ch=9 + time embedding). Architecture: encoder [64, 128, 256] with MaxPool downsampling, 256-channel bottleneck at 8x8, decoder with skip connections and bilinear upsampling. ~5M params. Optional sinusoidal time embedding for DDPM use.

**Step 1: Write tests**

Create `tests/test_unet.py`:

```python
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
        """Regressor should be ~5M params."""
        model = ConditionalUNet(in_ch=8, out_ch=1)
        n_params = sum(p.numel() for p in model.parameters())
        assert 4_000_000 < n_params < 6_000_000

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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_unet.py -v
```

Expected: `ERROR` -- `ModuleNotFoundError`

**Step 3: Write implementation**

Create `src/diffphys/model/unet.py`:

```python
"""Conditional U-Net for PDE field prediction.

Used as:
- Deterministic regressor: in_ch=8, out_ch=1 (conditioning -> field)
- DDPM noise predictor: in_ch=9, out_ch=1, time_emb_dim=256
  (noisy_field + conditioning -> predicted_noise)

Architecture: encoder [64, 128, 256], bottleneck 256 @ 8x8,
decoder with skip connections. ~5M params for regressor.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=1)


class ResBlock(nn.Module):
    """Residual block with optional time embedding injection."""

    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.time_proj = nn.Linear(time_emb_dim, out_ch) if time_emb_dim else None

    def forward(self, x, t_emb=None):
        h = F.gelu(self.bn1(self.conv1(x)))
        if self.time_proj is not None and t_emb is not None:
            h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.bn2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, time_emb_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t_emb=None):
        h = self.res1(x, t_emb)
        h = self.res2(h, t_emb)
        return h, self.pool(h)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.res1 = ResBlock(in_ch + skip_ch, out_ch, time_emb_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_emb_dim)

    def forward(self, x, skip, t_emb=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        h = torch.cat([x, skip], dim=1)
        h = self.res1(h, t_emb)
        return self.res2(h, t_emb)


class ConditionalUNet(nn.Module):
    """U-Net with optional time conditioning for DDPM.

    Args:
        in_ch: Input channels (8 for regressor, 9 for DDPM).
        out_ch: Output channels (1 for field prediction).
        base_ch: Base channel count (multiplied at each level).
        ch_mult: Channel multipliers per encoder level.
        time_emb_dim: If set, enables sinusoidal time embedding for DDPM.
    """

    def __init__(self, in_ch=8, out_ch=1, base_ch=64, ch_mult=(1, 2, 4),
                 time_emb_dim=None):
        super().__init__()
        chs = [base_ch * m for m in ch_mult]  # [64, 128, 256]

        # Time embedding
        self.time_emb = None
        t_dim = None
        if time_emb_dim is not None:
            t_dim = time_emb_dim
            self.time_emb = nn.Sequential(
                SinusoidalTimeEmbedding(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.GELU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )

        # Encoder
        self.enc_in = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        self.down0 = DownBlock(chs[0], chs[0], t_dim)
        self.down1 = DownBlock(chs[0], chs[1], t_dim)
        self.down2 = DownBlock(chs[1], chs[2], t_dim)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(chs[2], chs[2], t_dim),
            ResBlock(chs[2], chs[2], t_dim),
        )
        self._bottleneck_t_dim = t_dim

        # Decoder
        self.up2 = UpBlock(chs[2], chs[2], chs[2], t_dim)
        self.up1 = UpBlock(chs[2], chs[1], chs[1], t_dim)
        self.up0 = UpBlock(chs[1], chs[0], chs[0], t_dim)

        # Output
        self.out_conv = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x, t=None):
        t_emb = None
        if self.time_emb is not None and t is not None:
            t_emb = self.time_emb(t)

        x = self.enc_in(x)

        skip0, x = self.down0(x, t_emb)
        skip1, x = self.down1(x, t_emb)
        skip2, x = self.down2(x, t_emb)

        for block in self.bottleneck:
            x = block(x, t_emb)

        x = self.up2(x, skip2, t_emb)
        x = self.up1(x, skip1, t_emb)
        x = self.up0(x, skip0, t_emb)

        return self.out_conv(x)
```

**Step 4: Run tests**

```bash
pytest tests/test_unet.py -v
```

Expected: all passed

**Step 5: Commit**

```bash
git add src/diffphys/model/unet.py tests/test_unet.py
git commit -m "feat: conditional U-Net with optional time embedding

Encoder [64, 128, 256], bottleneck 256@8x8, decoder with skip connections.
~5M params for regressor (in_ch=8). DDPM variant adds sinusoidal time
embedding (in_ch=9, time_emb_dim=256). ResBlocks with BN + GELU."
```

---

### Task 4: Training Infrastructure

**Files:**
- Create: `src/diffphys/model/trainer.py`
- Create: `configs/unet_regressor.yaml`
- Create: `scripts/train.py`
- Create: `tests/test_trainer.py`

**Context:** Generic training loop that handles regressor and DDPM training. Reads YAML config, constructs model/optimizer/scheduler, trains with checkpointing and validation. The trainer is a simple functional design (not class hierarchy) for clarity.

**Step 1: Write trainer tests**

Create `tests/test_trainer.py`:

```python
"""Tests for training infrastructure."""

import numpy as np
import torch
import pytest
import yaml
from pathlib import Path

from diffphys.model.trainer import (
    build_model,
    build_optimizer,
    train_one_epoch,
    validate,
    load_config,
)
from diffphys.model.unet import ConditionalUNet
from diffphys.data.dataset import LaplacePDEDataset


@pytest.fixture
def tiny_npz(tmp_path):
    n, nx = 16, 16
    rng = np.random.default_rng(42)
    for split in ("train", "val"):
        np.savez(
            tmp_path / f"{split}.npz",
            fields=rng.standard_normal((n, nx, nx)).astype(np.float32),
            bc_top=rng.standard_normal((n, nx)).astype(np.float32),
            bc_bottom=rng.standard_normal((n, nx)).astype(np.float32),
            bc_left=rng.standard_normal((n, nx)).astype(np.float32),
            bc_right=rng.standard_normal((n, nx)).astype(np.float32),
        )
    return tmp_path


@pytest.fixture
def config(tiny_npz):
    return {
        "model": {"type": "unet", "in_channels": 8, "out_channels": 1, "base_channels": 16, "channel_mult": [1, 2, 4]},
        "training": {"batch_size": 4, "lr": 1e-3, "epochs": 2, "weight_decay": 1e-4, "scheduler": "cosine"},
        "data": {"train": str(tiny_npz / "train.npz"), "val": str(tiny_npz / "val.npz")},
        "logging": {"log_dir": str(tiny_npz / "logs"), "save_every": 1},
    }


class TestBuildModel:
    def test_builds_unet(self, config):
        model = build_model(config["model"])
        assert isinstance(model, ConditionalUNet)

    def test_respects_channel_config(self, config):
        config["model"]["base_channels"] = 32
        model = build_model(config["model"])
        # First conv should have 32 output channels
        assert model.enc_in.out_channels == 32


class TestTrainOneEpoch:
    def test_returns_finite_loss(self, config, tiny_npz):
        model = build_model(config["model"])
        optimizer = build_optimizer(model, config["training"])
        ds = LaplacePDEDataset(config["data"]["train"])
        loader = torch.utils.data.DataLoader(ds, batch_size=4)
        avg_loss = train_one_epoch(model, loader, optimizer, "cpu")
        assert np.isfinite(avg_loss)
        assert avg_loss > 0

    def test_loss_decreases_over_epochs(self, config, tiny_npz):
        model = build_model(config["model"])
        optimizer = build_optimizer(model, config["training"])
        ds = LaplacePDEDataset(config["data"]["train"])
        loader = torch.utils.data.DataLoader(ds, batch_size=4)
        loss1 = train_one_epoch(model, loader, optimizer, "cpu")
        loss2 = train_one_epoch(model, loader, optimizer, "cpu")
        loss3 = train_one_epoch(model, loader, optimizer, "cpu")
        # After a few epochs on tiny data, loss should decrease
        assert loss3 < loss1


class TestValidate:
    def test_returns_finite_loss(self, config, tiny_npz):
        model = build_model(config["model"])
        ds = LaplacePDEDataset(config["data"]["val"])
        loader = torch.utils.data.DataLoader(ds, batch_size=4)
        val_loss = validate(model, loader, "cpu")
        assert np.isfinite(val_loss)


class TestLoadConfig:
    def test_loads_yaml(self, tmp_path, config):
        path = tmp_path / "test_config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)
        loaded = load_config(str(path))
        assert loaded["model"]["type"] == "unet"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_trainer.py -v
```

Expected: `ERROR` -- `ModuleNotFoundError`

**Step 3: Write trainer implementation**

Create `src/diffphys/model/trainer.py`:

```python
"""Training infrastructure for regressor and DDPM models."""

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from .unet import ConditionalUNet


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(model_cfg):
    """Construct model from config dict."""
    mtype = model_cfg["type"]
    if mtype == "unet":
        return ConditionalUNet(
            in_ch=model_cfg.get("in_channels", 8),
            out_ch=model_cfg.get("out_channels", 1),
            base_ch=model_cfg.get("base_channels", 64),
            ch_mult=tuple(model_cfg.get("channel_mult", [1, 2, 4])),
            time_emb_dim=model_cfg.get("time_emb_dim", None),
        )
    elif mtype == "fno":
        from .fno import FNO2d
        return FNO2d(
            in_ch=model_cfg.get("in_channels", 8),
            out_ch=model_cfg.get("out_channels", 1),
            width=model_cfg.get("width", 40),
            modes=model_cfg.get("modes", 12),
            n_layers=model_cfg.get("n_layers", 4),
        )
    else:
        raise ValueError(f"Unknown model type: {mtype}")


def build_optimizer(model, train_cfg):
    return torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0),
    )


def build_scheduler(optimizer, train_cfg):
    if train_cfg.get("scheduler") == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"]
        )
    return None


def train_one_epoch(model, loader, optimizer, device):
    """Train for one epoch. Returns average MSE loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for cond, target in loader:
        cond, target = cond.to(device), target.to(device)
        pred = model(cond)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def validate(model, loader, device):
    """Compute average MSE on validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for cond, target in loader:
            cond, target = cond.to(device), target.to(device)
            pred = model(cond)
            total_loss += F.mse_loss(pred, target).item()
            n_batches += 1
    return total_loss / n_batches


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
    }, path)


def train(config, device="cpu"):
    """Full training loop from config dict."""
    from ..data.dataset import LaplacePDEDataset

    log_dir = Path(config["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config["model"]).to(device)
    optimizer = build_optimizer(model, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])

    train_ds = LaplacePDEDataset(config["data"]["train"])
    val_ds = LaplacePDEDataset(config["data"]["val"])
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["training"]["batch_size"],
        shuffle=True, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["training"]["batch_size"],
        num_workers=0,
    )

    best_val = float("inf")
    history = []

    for epoch in range(config["training"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        if scheduler:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={lr:.2e}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, log_dir / "best.pt")

        if (epoch + 1) % config["logging"].get("save_every", 10) == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, log_dir / f"epoch_{epoch+1}.pt")

    with open(log_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, history
```

**Step 4: Write config**

Create `configs/unet_regressor.yaml`:

```yaml
model:
  type: unet
  in_channels: 8
  out_channels: 1
  base_channels: 64
  channel_mult: [1, 2, 4]

training:
  batch_size: 64
  lr: 1.0e-3
  epochs: 50
  weight_decay: 1.0e-4
  scheduler: cosine

data:
  train: data/train.npz
  val: data/val.npz

logging:
  log_dir: experiments/unet_regressor
  save_every: 10
```

**Step 5: Write training script**

Create `scripts/train.py`:

```python
"""Train a model from a YAML config file.

Usage:
    python scripts/train.py --config configs/unet_regressor.yaml
    python scripts/train.py --config configs/unet_regressor.yaml --device cuda
"""

import argparse

import torch

from diffphys.model.trainer import load_config, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", default=None, help="Device (cpu/cuda/mps)")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Training {config['model']['type']} on {device}")
    train(config, device=device)


if __name__ == "__main__":
    main()
```

**Step 6: Run tests**

```bash
pytest tests/test_trainer.py -v
```

Expected: all passed

**Step 7: Commit**

```bash
git add src/diffphys/model/trainer.py tests/test_trainer.py configs/unet_regressor.yaml scripts/train.py
git commit -m "feat: training loop with YAML config and checkpointing

Supports regressor and DDPM training. Adam + cosine annealing.
Saves best checkpoint by val loss. scripts/train.py as CLI entry point."
```

---

### Task 5: FNO Architecture

**Files:**
- Create: `src/diffphys/model/fno.py`
- Create: `configs/fno.yaml`
- Create: `tests/test_fno.py`

**Context:** Fourier Neural Operator with spectral convolutions in Fourier space. 4 Fourier layers, width=40, modes=12. ~2M params. Input: (B, 8, 64, 64) conditioning + 2 grid coordinate channels lifted to width. Output: (B, 1, 64, 64) predicted field.

**Step 1: Write tests**

Create `tests/test_fno.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_fno.py -v
```

Expected: `ERROR` -- `ModuleNotFoundError`

**Step 3: Write implementation**

Create `src/diffphys/model/fno.py`:

```python
"""Fourier Neural Operator for 2D PDE fields.

4 Fourier layers with spectral convolutions. Input is the 8-channel
conditioning tensor + 2 grid coordinate channels, lifted to a hidden
width, processed in Fourier space, then projected to output.
~2M params with width=40, modes=12.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Spectral convolution via truncated FFT."""

    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_ch * out_ch)
        self.weights1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    def _compl_mul2d(self, x, weights):
        # (B, in_ch, M1, M2) x (in_ch, out_ch, M1, M2) -> (B, out_ch, M1, M2)
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(B, self.out_ch, x.size(2), x.size(3) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2] = \
            self._compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self._compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        return torch.fft.irfft2(out_ft, s=(x.size(2), x.size(3)))


class FNOBlock(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes, modes)
        self.linear = nn.Conv2d(width, width, 1)
        self.norm = nn.InstanceNorm2d(width)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.linear(x)))


class FNO2d(nn.Module):
    """Fourier Neural Operator for 2D fields.

    Args:
        in_ch: Input channels (8 for BC conditioning).
        out_ch: Output channels (1 for field prediction).
        width: Hidden channel width.
        modes: Number of Fourier modes to keep per dimension.
        n_layers: Number of Fourier layers.
    """

    def __init__(self, in_ch=8, out_ch=1, width=40, modes=12, n_layers=4):
        super().__init__()
        self.width = width

        # +2 for grid coordinates (x, y)
        self.lift = nn.Linear(in_ch + 2, width)

        self.layers = nn.ModuleList([
            FNOBlock(width, modes) for _ in range(n_layers)
        ])

        self.project = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, out_ch),
        )

    def _get_grid(self, x):
        B, _, H, W = x.shape
        gx = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        gy = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        return torch.cat([gx, gy], dim=1)

    def forward(self, x):
        grid = self._get_grid(x)
        x = torch.cat([x, grid], dim=1)  # (B, in_ch+2, H, W)

        # Lift: channel-wise linear (permute to last dim)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.lift(x)
        x = x.permute(0, 3, 1, 2)  # (B, width, H, W)

        for layer in self.layers:
            x = layer(x)

        # Project: channel-wise linear
        x = x.permute(0, 2, 3, 1)
        x = self.project(x)
        return x.permute(0, 3, 1, 2)  # (B, out_ch, H, W)
```

**Step 4: Write FNO config**

Create `configs/fno.yaml`:

```yaml
model:
  type: fno
  in_channels: 8
  out_channels: 1
  width: 40
  modes: 12
  n_layers: 4

training:
  batch_size: 64
  lr: 1.0e-3
  epochs: 50
  weight_decay: 1.0e-4
  scheduler: cosine

data:
  train: data/train.npz
  val: data/val.npz

logging:
  log_dir: experiments/fno
  save_every: 10
```

**Step 5: Run tests**

```bash
pytest tests/test_fno.py -v
```

Expected: all passed

**Step 6: Commit**

```bash
git add src/diffphys/model/fno.py tests/test_fno.py configs/fno.yaml
git commit -m "feat: Fourier Neural Operator (2M params)

4 Fourier layers, width=40, modes=12. Spectral convolutions via
truncated FFT. Grid coordinates appended as 2 extra input channels."
```

---

### Task 6: Physics-Aware Evaluation Metrics

**Files:**
- Create: `src/diffphys/evaluation/metrics.py`
- Create: `src/diffphys/evaluation/evaluate.py`
- Create: `scripts/evaluate.py`
- Create: `tests/test_metrics.py`

**Context:** Phase 1 metrics: relative L2 error, PDE residual norm, BC error, maximum principle violations, energy functional error. The evaluate module loads a trained model, runs inference on test splits, and computes all metrics.

**Step 1: Write metrics tests**

Create `tests/test_metrics.py`:

```python
"""Tests for physics-aware evaluation metrics."""

import torch
import numpy as np
import pytest
from diffphys.evaluation.metrics import (
    relative_l2_error,
    pde_residual_norm,
    bc_error,
    max_principle_violations,
    energy_functional,
)


class TestRelativeL2Error:
    def test_identical_is_zero(self):
        x = torch.randn(4, 1, 64, 64)
        err = relative_l2_error(x, x)
        assert err.shape == (4,)
        torch.testing.assert_close(err, torch.zeros(4), atol=1e-7, rtol=0)

    def test_known_value(self):
        pred = torch.ones(1, 1, 4, 4) * 2.0
        true = torch.ones(1, 1, 4, 4) * 1.0
        err = relative_l2_error(pred, true)
        # ||pred - true|| / ||true|| = ||1|| / ||1|| = 1.0
        assert err.item() == pytest.approx(1.0, abs=1e-6)

    def test_batch_dimension(self):
        pred = torch.randn(8, 1, 16, 16)
        true = torch.randn(8, 1, 16, 16)
        err = relative_l2_error(pred, true)
        assert err.shape == (8,)


class TestPDEResidualNorm:
    def test_constant_field_is_zero(self):
        """Laplacian of a constant field is zero."""
        field = torch.ones(1, 1, 16, 16) * 3.0
        res = pde_residual_norm(field, h=1.0 / 15)
        assert res.item() == pytest.approx(0.0, abs=1e-6)

    def test_linear_field_is_zero(self):
        """Laplacian of a linear field (ax + by + c) is zero."""
        x = torch.linspace(0, 1, 16)
        y = torch.linspace(0, 1, 16)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        field = (2 * X + 3 * Y + 1).unsqueeze(0).unsqueeze(0)
        res = pde_residual_norm(field, h=1.0 / 15)
        assert res.item() == pytest.approx(0.0, abs=1e-4)

    def test_nonzero_for_nonharmonic(self):
        """Laplacian of x^2 + y^2 is 4 (not zero)."""
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        field = (X ** 2 + Y ** 2).unsqueeze(0).unsqueeze(0)
        res = pde_residual_norm(field, h=1.0 / 63)
        assert res.item() > 0.1

    def test_batch_dimension(self):
        field = torch.randn(4, 1, 16, 16)
        res = pde_residual_norm(field, h=1.0 / 15)
        assert res.shape == (4,)


class TestBCError:
    def test_exact_bcs_zero(self):
        pred = torch.randn(2, 1, 16, 16)
        true = pred.clone()
        err = bc_error(pred, true)
        torch.testing.assert_close(err, torch.zeros(2), atol=1e-7, rtol=0)

    def test_measures_boundary_difference(self):
        pred = torch.zeros(1, 1, 8, 8)
        true = torch.ones(1, 1, 8, 8)
        err = bc_error(pred, true)
        assert err.item() > 0


class TestMaxPrincipleViolations:
    def test_harmonic_field_no_violations(self):
        """Harmonic field satisfies max principle."""
        field = torch.ones(1, 1, 16, 16) * 0.5
        field[:, :, 0, :] = 0.0   # top
        field[:, :, -1, :] = 1.0  # bottom
        n_viol = max_principle_violations(field)
        assert n_viol.item() == 0

    def test_detects_violations(self):
        """Interior exceeding boundary extremes is a violation."""
        field = torch.zeros(1, 1, 8, 8)
        field[:, :, 4, 4] = 2.0  # interior > max boundary (0)
        n_viol = max_principle_violations(field)
        assert n_viol.item() > 0


class TestEnergyFunctional:
    def test_constant_field_zero_energy(self):
        """Constant field has zero gradient energy."""
        field = torch.ones(1, 1, 16, 16) * 5.0
        E = energy_functional(field, h=1.0 / 15)
        assert E.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive(self):
        field = torch.randn(2, 1, 16, 16)
        E = energy_functional(field, h=1.0 / 15)
        assert (E >= 0).all()

    def test_batch_dimension(self):
        field = torch.randn(4, 1, 16, 16)
        E = energy_functional(field, h=1.0 / 15)
        assert E.shape == (4,)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_metrics.py -v
```

Expected: `ERROR` -- `ModuleNotFoundError`

**Step 3: Write metrics implementation**

Create `src/diffphys/evaluation/metrics.py`:

```python
"""Physics-aware evaluation metrics for PDE field predictions.

All functions expect tensors of shape (B, 1, H, W) and return
per-sample metrics of shape (B,).
"""

import torch


def relative_l2_error(pred, true):
    """||pred - true||_2 / ||true||_2, per sample."""
    diff = (pred - true).reshape(pred.shape[0], -1)
    true_flat = true.reshape(true.shape[0], -1)
    return diff.norm(dim=1) / true_flat.norm(dim=1).clamp(min=1e-8)


def pde_residual_norm(field, h=1.0 / 63):
    """RMS of discrete Laplacian on interior points.

    For a Laplace solution, this should be near zero.
    """
    f = field[:, 0]  # (B, H, W)
    lap = (
        f[:, :-2, 1:-1] + f[:, 2:, 1:-1]
        + f[:, 1:-1, :-2] + f[:, 1:-1, 2:]
        - 4 * f[:, 1:-1, 1:-1]
    ) / (h ** 2)
    return lap.reshape(field.shape[0], -1).pow(2).mean(dim=1).sqrt()


def bc_error(pred, true):
    """Mean absolute error on boundary pixels."""
    B = pred.shape[0]
    errors = []
    for f_pred, f_true in zip(pred[:, 0], true[:, 0]):
        top = (f_pred[0, :] - f_true[0, :]).abs().mean()
        bot = (f_pred[-1, :] - f_true[-1, :]).abs().mean()
        left = (f_pred[:, 0] - f_true[:, 0]).abs().mean()
        right = (f_pred[:, -1] - f_true[:, -1]).abs().mean()
        errors.append((top + bot + left + right) / 4)
    return torch.stack(errors)


def max_principle_violations(field):
    """Count interior pixels violating the discrete maximum principle."""
    f = field[:, 0]  # (B, H, W)
    interior = f[:, 1:-1, 1:-1]

    # Boundary extremes per sample
    top = f[:, 0, :]
    bot = f[:, -1, :]
    left = f[:, :, 0]
    right = f[:, :, -1]
    all_bc = torch.cat([top, bot, left, right], dim=1)
    bc_min = all_bc.min(dim=1, keepdim=True).values.unsqueeze(2)
    bc_max = all_bc.max(dim=1, keepdim=True).values.unsqueeze(2)

    violations = ((interior < bc_min - 1e-6) | (interior > bc_max + 1e-6))
    return violations.reshape(field.shape[0], -1).sum(dim=1)


def energy_functional(field, h=1.0 / 63):
    """Dirichlet energy: 0.5 * integral(|grad T|^2) dx dy."""
    f = field[:, 0]  # (B, H, W)
    dx = (f[:, :, 1:] - f[:, :, :-1]) / h
    dy = (f[:, 1:, :] - f[:, :-1, :]) / h
    # Average over the domain (trapezoidal-ish)
    E = 0.5 * h * h * (dx.pow(2).sum(dim=(1, 2)) + dy.pow(2).sum(dim=(1, 2)))
    return E
```

**Step 4: Write evaluate module**

Create `src/diffphys/evaluation/evaluate.py`:

```python
"""Model evaluation on test splits with physics-aware metrics."""

import json
from pathlib import Path

import torch
import numpy as np

from ..data.dataset import LaplacePDEDataset
from ..model.trainer import build_model, load_config
from .metrics import (
    relative_l2_error,
    pde_residual_norm,
    bc_error,
    max_principle_violations,
    energy_functional,
)


def evaluate_regressor(model, loader, device, h=1.0 / 63):
    """Evaluate a deterministic model on a dataset.

    Returns dict of metric_name -> list of per-sample values.
    """
    model.eval()
    results = {
        "rel_l2": [], "pde_residual": [], "bc_err": [],
        "max_viol": [], "energy_pred": [], "energy_true": [],
    }

    with torch.no_grad():
        for cond, target in loader:
            cond, target = cond.to(device), target.to(device)
            pred = model(cond)

            results["rel_l2"].append(relative_l2_error(pred, target))
            results["pde_residual"].append(pde_residual_norm(pred, h))
            results["bc_err"].append(bc_error(pred, target))
            results["max_viol"].append(max_principle_violations(pred))
            results["energy_pred"].append(energy_functional(pred, h))
            results["energy_true"].append(energy_functional(target, h))

    return {k: torch.cat(v).cpu().numpy().tolist() for k, v in results.items()}


def summarize_results(results):
    """Compute mean and std for each metric."""
    summary = {}
    for k, vals in results.items():
        arr = np.array(vals)
        summary[k] = {"mean": float(arr.mean()), "std": float(arr.std())}
    # Energy error as relative difference
    e_pred = np.array(results["energy_pred"])
    e_true = np.array(results["energy_true"])
    rel_energy = np.abs(e_pred - e_true) / np.maximum(e_true, 1e-8)
    summary["rel_energy_err"] = {"mean": float(rel_energy.mean()), "std": float(rel_energy.std())}
    return summary


def run_evaluation(config_path, checkpoint_path, test_npz_paths, device="cpu"):
    """Full evaluation pipeline."""
    config = load_config(config_path)
    model = build_model(config["model"]).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    all_results = {}
    for split_name, npz_path in test_npz_paths.items():
        ds = LaplacePDEDataset(npz_path)
        loader = torch.utils.data.DataLoader(ds, batch_size=64)
        raw = evaluate_regressor(model, loader, device)
        all_results[split_name] = summarize_results(raw)

    return all_results
```

**Step 5: Write evaluation script**

Create `scripts/evaluate.py`:

```python
"""Evaluate a trained model on test splits.

Usage:
    python scripts/evaluate.py --config configs/unet_regressor.yaml \
        --checkpoint experiments/unet_regressor/best.pt
"""

import argparse
import json

from diffphys.evaluation.evaluate import run_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args()

    test_splits = {
        "test_in": "data/test_in.npz",
        "test_ood": "data/test_ood.npz",
    }

    results = run_evaluation(args.config, args.checkpoint, test_splits, args.device)

    for split, metrics in results.items():
        print(f"\n=== {split} ===")
        for name, stats in metrics.items():
            print(f"  {name:20s}: {stats['mean']:.6f} +/- {stats['std']:.6f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
```

**Step 6: Run tests**

```bash
pytest tests/test_metrics.py -v
```

Expected: all passed

**Step 7: Commit**

```bash
git add src/diffphys/evaluation/metrics.py src/diffphys/evaluation/evaluate.py scripts/evaluate.py tests/test_metrics.py
git commit -m "feat: physics-aware evaluation metrics and evaluation pipeline

Relative L2, PDE residual norm, BC error, max principle violations,
energy functional. Evaluation script runs on test_in and test_ood splits."
```

---

### Task 7: Deep Ensemble

**Files:**
- Create: `src/diffphys/model/ensemble.py`
- Create: `configs/ensemble.yaml`
- Create: `tests/test_ensemble.py`

**Context:** 5 independent U-Net regressors trained with different seeds. At inference, run all 5 and compute pixel-wise mean and variance. This is the non-generative probabilistic baseline.

**Step 1: Write tests**

Create `tests/test_ensemble.py`:

```python
"""Tests for deep ensemble inference."""

import torch
import pytest
from diffphys.model.ensemble import EnsemblePredictor
from diffphys.model.unet import ConditionalUNet


class TestEnsemblePredictor:
    @pytest.fixture
    def ensemble(self):
        """3 tiny U-Nets (instead of 5) for fast testing."""
        models = []
        for seed in range(3):
            torch.manual_seed(seed)
            models.append(ConditionalUNet(in_ch=8, out_ch=1, base_ch=8, ch_mult=(1, 2)))
        return EnsemblePredictor(models)

    def test_mean_shape(self, ensemble):
        x = torch.randn(2, 8, 16, 16)
        mean, var = ensemble.predict(x)
        assert mean.shape == (2, 1, 16, 16)

    def test_variance_shape(self, ensemble):
        x = torch.randn(2, 8, 16, 16)
        mean, var = ensemble.predict(x)
        assert var.shape == (2, 1, 16, 16)

    def test_variance_non_negative(self, ensemble):
        x = torch.randn(2, 8, 16, 16)
        _, var = ensemble.predict(x)
        assert (var >= 0).all()

    def test_mean_is_average_of_members(self, ensemble):
        x = torch.randn(1, 8, 16, 16)
        mean, _ = ensemble.predict(x)
        # Manually compute
        preds = [m(x) for m in ensemble.models]
        manual_mean = torch.stack(preds).mean(dim=0)
        torch.testing.assert_close(mean, manual_mean)

    def test_single_model_zero_variance(self):
        """Ensemble of 1 should have zero variance."""
        model = ConditionalUNet(in_ch=8, out_ch=1, base_ch=8, ch_mult=(1, 2))
        ens = EnsemblePredictor([model])
        x = torch.randn(1, 8, 16, 16)
        _, var = ens.predict(x)
        torch.testing.assert_close(var, torch.zeros_like(var), atol=1e-7, rtol=0)

    def test_get_all_predictions(self, ensemble):
        x = torch.randn(2, 8, 16, 16)
        all_preds = ensemble.predict_all(x)
        assert all_preds.shape == (3, 2, 1, 16, 16)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ensemble.py -v
```

**Step 3: Write implementation**

Create `src/diffphys/model/ensemble.py`:

```python
"""Deep ensemble of U-Net regressors.

Train K independent U-Nets with different seeds, then aggregate
predictions at inference via mean and variance.
"""

import torch


class EnsemblePredictor:
    """Wraps K trained models for ensemble prediction.

    Args:
        models: List of trained nn.Module instances.
    """

    def __init__(self, models):
        self.models = models
        for m in self.models:
            m.eval()

    @torch.no_grad()
    def predict_all(self, x):
        """Return all member predictions. Shape: (K, B, C, H, W)."""
        preds = [m(x) for m in self.models]
        return torch.stack(preds, dim=0)

    @torch.no_grad()
    def predict(self, x):
        """Return (mean, variance) over ensemble members."""
        all_preds = self.predict_all(x)  # (K, B, C, H, W)
        mean = all_preds.mean(dim=0)
        var = all_preds.var(dim=0)
        return mean, var
```

**Step 4: Write ensemble config**

Create `configs/ensemble.yaml`:

```yaml
model:
  type: unet
  in_channels: 8
  out_channels: 1
  base_channels: 64
  channel_mult: [1, 2, 4]

ensemble:
  n_members: 5
  seeds: [100, 101, 102, 103, 104]

training:
  batch_size: 64
  lr: 1.0e-3
  epochs: 50
  weight_decay: 1.0e-4
  scheduler: cosine

data:
  train: data/train.npz
  val: data/val.npz

logging:
  log_dir: experiments/ensemble
  save_every: 10
```

**Step 5: Run tests**

```bash
pytest tests/test_ensemble.py -v
```

Expected: all passed

**Step 6: Commit**

```bash
git add src/diffphys/model/ensemble.py tests/test_ensemble.py configs/ensemble.yaml
git commit -m "feat: deep ensemble with mean/variance aggregation

EnsemblePredictor wraps K trained U-Nets. Returns pixel-wise mean
and variance for uncertainty quantification. Config for 5-member ensemble."
```

---

### Task 8: Conditional DDPM

**Files:**
- Create: `src/diffphys/model/ddpm.py`
- Create: `configs/ddpm.yaml`
- Create: `tests/test_ddpm.py`

**Context:** Denoising Diffusion Probabilistic Model with T=200, linear beta schedule (1e-4 to 0.02). Training: epsilon prediction with MSE loss. Sampling: standard reverse process. The U-Net from Task 3 serves as the noise predictor with in_ch=9 (1 noisy + 8 conditioning) and time_emb_dim=256.

**Step 1: Write tests**

Create `tests/test_ddpm.py`:

```python
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
        assert schedule.alpha_bars[200] < 0.1  # close to 0 at t=T

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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ddpm.py -v
```

**Step 3: Write implementation**

Create `src/diffphys/model/ddpm.py`:

```python
"""Conditional Denoising Diffusion Probabilistic Model.

T=200, linear beta schedule (1e-4 to 0.02), epsilon prediction.
Uses ConditionalUNet(in_ch=9, time_emb_dim=256) as noise predictor.
Input to U-Net: concatenation of noisy field (1 ch) and conditioning (8 ch).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseSchedule:
    """Linear beta schedule for DDPM.

    Indexing convention: betas[0] is unused; betas[1] through betas[T]
    are the T noise levels.
    """

    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02):
        self.T = T
        betas = torch.zeros(T + 1)
        betas[1:] = torch.linspace(beta_start, beta_end, T)
        self.betas = betas

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def add_noise(self, x0, noise, t):
        """q(x_t | x_0) = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise."""
        ab = self.alpha_bars[t]
        while ab.dim() < x0.dim():
            ab = ab.unsqueeze(-1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


class DDPM(nn.Module):
    """Conditional DDPM wrapper.

    Args:
        model: ConditionalUNet(in_ch=9, out_ch=1, time_emb_dim=...).
        T: Number of diffusion timesteps.
        beta_start, beta_end: Linear schedule endpoints.
    """

    def __init__(self, model, T=200, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.schedule = NoiseSchedule(T, beta_start, beta_end)
        self.T = T

    def training_step(self, conditioning, target):
        """Compute epsilon-prediction MSE loss.

        Args:
            conditioning: (B, 8, H, W) BC conditioning tensor.
            target: (B, 1, H, W) clean solution field.

        Returns:
            Scalar loss.
        """
        device = target.device
        self.schedule.to(device)

        B = target.shape[0]
        t = torch.randint(1, self.T + 1, (B,), device=device)
        noise = torch.randn_like(target)
        x_t = self.schedule.add_noise(target, noise, t)

        model_input = torch.cat([x_t, conditioning], dim=1)  # (B, 9, H, W)
        pred_noise = self.model(model_input, t)

        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, conditioning, n_samples=1):
        """Generate samples via reverse diffusion.

        Args:
            conditioning: (B, 8, H, W).
            n_samples: Number of samples per conditioning input.

        Returns:
            (n_samples, B, 1, H, W) tensor.
        """
        device = conditioning.device
        self.schedule.to(device)
        B, _, H, W = conditioning.shape

        # Expand conditioning for n_samples
        cond = conditioning.unsqueeze(0).expand(n_samples, -1, -1, -1, -1)
        cond = cond.reshape(n_samples * B, 8, H, W)

        x = torch.randn(n_samples * B, 1, H, W, device=device)

        for t in reversed(range(1, self.T + 1)):
            t_batch = torch.full((n_samples * B,), t, device=device, dtype=torch.long)

            model_input = torch.cat([x, cond], dim=1)
            pred_noise = self.model(model_input, t_batch)

            alpha_t = self.schedule.alphas[t]
            alpha_bar_t = self.schedule.alpha_bars[t]
            beta_t = self.schedule.betas[t]

            mean = (1.0 / alpha_t.sqrt()) * (
                x - (beta_t / (1 - alpha_bar_t).sqrt()) * pred_noise
            )

            if t > 1:
                z = torch.randn_like(x)
                x = mean + beta_t.sqrt() * z
            else:
                x = mean

        return x.reshape(n_samples, B, 1, H, W)
```

**Step 4: Write DDPM config**

Create `configs/ddpm.yaml`:

```yaml
model:
  type: unet
  in_channels: 9
  out_channels: 1
  base_channels: 64
  channel_mult: [1, 2, 4]
  time_emb_dim: 256

ddpm:
  T: 200
  beta_start: 1.0e-4
  beta_end: 0.02

training:
  batch_size: 64
  lr: 2.0e-4
  epochs: 100
  weight_decay: 0.0
  scheduler: cosine

data:
  train: data/train.npz
  val: data/val.npz

logging:
  log_dir: experiments/ddpm
  save_every: 20
```

**Step 5: Run tests**

```bash
pytest tests/test_ddpm.py -v
```

Expected: all passed

**Step 6: Add DDPM training support to trainer**

Modify `src/diffphys/model/trainer.py` -- add `train_ddpm_one_epoch` and `train_ddpm` functions:

Append to `src/diffphys/model/trainer.py`:

```python
def train_ddpm_one_epoch(ddpm, loader, optimizer, device):
    """Train DDPM for one epoch. Returns average loss."""
    ddpm.train()
    total_loss = 0.0
    n_batches = 0
    for cond, target in loader:
        cond, target = cond.to(device), target.to(device)
        loss = ddpm.training_step(cond, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def validate_ddpm(ddpm, loader, device):
    """Compute average DDPM loss on validation set."""
    ddpm.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for cond, target in loader:
            cond, target = cond.to(device), target.to(device)
            loss = ddpm.training_step(cond, target)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches


def train_ddpm(config, device="cpu"):
    """Full DDPM training loop from config dict."""
    from ..data.dataset import LaplacePDEDataset
    from .ddpm import DDPM

    log_dir = Path(config["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config["model"]).to(device)
    ddpm_cfg = config["ddpm"]
    ddpm = DDPM(model, T=ddpm_cfg["T"],
                beta_start=ddpm_cfg["beta_start"],
                beta_end=ddpm_cfg["beta_end"]).to(device)

    optimizer = build_optimizer(ddpm, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])

    train_ds = LaplacePDEDataset(config["data"]["train"])
    val_ds = LaplacePDEDataset(config["data"]["val"])
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["training"]["batch_size"],
        shuffle=True, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["training"]["batch_size"], num_workers=0,
    )

    best_val = float("inf")
    history = []

    for epoch in range(config["training"]["epochs"]):
        train_loss = train_ddpm_one_epoch(ddpm, train_loader, optimizer, device)
        val_loss = validate_ddpm(ddpm, val_loader, device)
        if scheduler:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={lr:.2e}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ddpm, optimizer, epoch, val_loss, log_dir / "best.pt")

        if (epoch + 1) % config["logging"].get("save_every", 10) == 0:
            save_checkpoint(ddpm, optimizer, epoch, val_loss, log_dir / f"epoch_{epoch+1}.pt")

    with open(log_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return ddpm, history
```

Update `scripts/train.py` to dispatch DDPM:

```python
# In main(), after loading config:
    if "ddpm" in config:
        print(f"Training DDPM on {device}")
        from diffphys.model.trainer import train_ddpm
        train_ddpm(config, device=device)
    else:
        print(f"Training {config['model']['type']} on {device}")
        train(config, device=device)
```

**Step 7: Commit**

```bash
git add src/diffphys/model/ddpm.py tests/test_ddpm.py configs/ddpm.yaml src/diffphys/model/trainer.py scripts/train.py
git commit -m "feat: conditional DDPM with epsilon prediction

T=200 linear beta schedule, reverse-process sampling.
NoiseSchedule handles forward process. DDPM wraps U-Net(in_ch=9)
for noise prediction. Training loop added to trainer module."
```

---

### Task 9: Observation Regimes

**Files:**
- Create: `src/diffphys/data/observation.py`
- Create: `tests/test_observation.py`

**Context:** Phase 2 introduces noisy and sparse BC observations. 5 regimes: exact (Phase 1 baseline), dense-noisy (all 64 points, sigma=0.1), sparse-clean (16 points, no noise), sparse-noisy (16 points, sigma=0.1), very-sparse (8 points, sigma=0.2). Sparse regimes interpolate between observed points and set masks to 1.0 at observed positions, 0.0 elsewhere.

**Step 1: Write tests**

Create `tests/test_observation.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_observation.py -v
```

**Step 3: Write implementation**

Create `src/diffphys/data/observation.py`:

```python
"""Observation regime transformations for Phase 2.

Regimes corrupt boundary conditions with noise and/or sparsity,
returning (observed_bc, mask) pairs for conditioning.
"""

import torch


REGIMES = ["exact", "dense-noisy", "sparse-clean", "sparse-noisy", "very-sparse"]

REGIME_CONFIG = {
    "exact":        {"n_points": 64, "noise_sigma": 0.0},
    "dense-noisy":  {"n_points": 64, "noise_sigma": 0.1},
    "sparse-clean": {"n_points": 16, "noise_sigma": 0.0},
    "sparse-noisy": {"n_points": 16, "noise_sigma": 0.1},
    "very-sparse":  {"n_points": 8,  "noise_sigma": 0.2},
}


def apply_observation_regime(bc, regime, rng=None):
    """Apply observation regime to a single BC edge.

    Args:
        bc: (nx,) tensor of true boundary values.
        regime: One of REGIMES.
        rng: Optional torch.Generator for reproducible noise.

    Returns:
        (observed_bc, mask): both (nx,) tensors.
        observed_bc has interpolated values between observed points.
        mask is 1.0 at observed positions, 0.0 elsewhere.
    """
    if regime not in REGIME_CONFIG:
        raise ValueError(f"Unknown regime: {regime}")

    nx = bc.shape[0]
    cfg = REGIME_CONFIG[regime]
    n_points = cfg["n_points"]
    sigma = cfg["noise_sigma"]

    if n_points >= nx:
        # All points observed
        mask = torch.ones(nx)
        observed = bc.clone()
        if sigma > 0:
            noise = torch.randn(nx, generator=rng) * sigma
            observed = observed + noise
        return observed, mask

    # Sparse: select evenly-spaced points including endpoints
    indices = torch.linspace(0, nx - 1, n_points).long()
    mask = torch.zeros(nx)
    mask[indices] = 1.0

    # Get observed values (with optional noise)
    obs_values = bc[indices].clone()
    if sigma > 0:
        noise = torch.randn(n_points, generator=rng) * sigma
        obs_values = obs_values + noise

    # Interpolate between observed points
    x_obs = indices.float()
    x_all = torch.arange(nx, dtype=torch.float32)
    observed = _linear_interp(x_obs, obs_values, x_all)

    return observed, mask


def _linear_interp(x_obs, y_obs, x_query):
    """Piecewise linear interpolation."""
    result = torch.zeros_like(x_query)
    for i in range(len(x_obs) - 1):
        x0, x1 = x_obs[i], x_obs[i + 1]
        y0, y1 = y_obs[i], y_obs[i + 1]
        mask = (x_query >= x0) & (x_query <= x1)
        t = (x_query[mask] - x0) / (x1 - x0)
        result[mask] = y0 + t * (y1 - y0)
    return result
```

**Step 4: Run tests**

```bash
pytest tests/test_observation.py -v
```

Expected: all passed

**Step 5: Commit**

```bash
git add src/diffphys/data/observation.py tests/test_observation.py
git commit -m "feat: observation regime transforms for Phase 2

5 regimes: exact, dense-noisy, sparse-clean, sparse-noisy, very-sparse.
Sparse regimes interpolate between evenly-spaced observed points.
Returns (observed_bc, mask) for conditioning tensor encoding."
```

---

### Task 10: UQ Metrics

**Files:**
- Create: `src/diffphys/evaluation/uq_metrics.py`
- Create: `tests/test_uq_metrics.py`

**Context:** Phase 2 uncertainty quantification metrics: pixel-wise coverage at nominal levels (50%, 90%, 95%), CRPS, calibration error, and sharpness. These apply to both ensemble and DDPM outputs, which provide (mean, variance) or (samples) respectively.

**Step 1: Write tests**

Create `tests/test_uq_metrics.py`:

```python
"""Tests for uncertainty quantification metrics."""

import torch
import pytest
from diffphys.evaluation.uq_metrics import (
    pixelwise_coverage,
    crps_gaussian,
    calibration_error,
    sharpness,
)


class TestPixelwiseCoverage:
    def test_perfect_coverage_at_wide_interval(self):
        """Very wide intervals should give ~100% coverage."""
        true = torch.zeros(10, 1, 8, 8)
        mean = torch.zeros(10, 1, 8, 8)
        std = torch.ones(10, 1, 8, 8) * 100  # huge std
        cov = pixelwise_coverage(true, mean, std, level=0.95)
        assert cov > 0.99

    def test_zero_coverage_at_narrow_interval(self):
        """Extremely narrow intervals on offset predictions -> low coverage."""
        true = torch.ones(10, 1, 8, 8)
        mean = torch.zeros(10, 1, 8, 8)
        std = torch.ones(10, 1, 8, 8) * 1e-6
        cov = pixelwise_coverage(true, mean, std, level=0.95)
        assert cov < 0.01

    def test_returns_scalar(self):
        true = torch.randn(4, 1, 8, 8)
        mean = torch.randn(4, 1, 8, 8)
        std = torch.ones(4, 1, 8, 8)
        cov = pixelwise_coverage(true, mean, std, level=0.90)
        assert cov.dim() == 0


class TestCRPSGaussian:
    def test_perfect_prediction_near_zero(self):
        """CRPS should be near zero when prediction matches truth exactly."""
        true = torch.zeros(4, 1, 8, 8)
        mean = torch.zeros(4, 1, 8, 8)
        std = torch.ones(4, 1, 8, 8) * 0.01
        crps = crps_gaussian(true, mean, std)
        assert crps.mean() < 0.01

    def test_increases_with_error(self):
        true = torch.zeros(4, 1, 8, 8)
        std = torch.ones(4, 1, 8, 8) * 0.5
        crps_close = crps_gaussian(true, torch.zeros(4, 1, 8, 8), std)
        crps_far = crps_gaussian(true, torch.ones(4, 1, 8, 8) * 5.0, std)
        assert crps_far.mean() > crps_close.mean()

    def test_output_shape(self):
        true = torch.randn(4, 1, 8, 8)
        mean = torch.randn(4, 1, 8, 8)
        std = torch.ones(4, 1, 8, 8)
        crps = crps_gaussian(true, mean, std)
        assert crps.shape == (4,)


class TestCalibrationError:
    def test_well_calibrated_low_error(self):
        """Standard normal samples should be well-calibrated."""
        torch.manual_seed(42)
        N = 1000
        true = torch.randn(N, 1, 1, 1)
        mean = torch.zeros(N, 1, 1, 1)
        std = torch.ones(N, 1, 1, 1)
        err = calibration_error(true, mean, std)
        assert err < 0.1

    def test_miscalibrated_high_error(self):
        """Overconfident predictions should have high calibration error."""
        torch.manual_seed(42)
        N = 1000
        true = torch.randn(N, 1, 1, 1) * 5
        mean = torch.zeros(N, 1, 1, 1)
        std = torch.ones(N, 1, 1, 1) * 0.1  # way too confident
        err = calibration_error(true, mean, std)
        assert err > 0.3


class TestSharpness:
    def test_narrow_is_sharper(self):
        std_narrow = torch.ones(4, 1, 8, 8) * 0.1
        std_wide = torch.ones(4, 1, 8, 8) * 10.0
        assert sharpness(std_narrow) < sharpness(std_wide)

    def test_returns_scalar(self):
        std = torch.ones(4, 1, 8, 8)
        s = sharpness(std)
        assert s.dim() == 0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_uq_metrics.py -v
```

**Step 3: Write implementation**

Create `src/diffphys/evaluation/uq_metrics.py`:

```python
"""Uncertainty quantification metrics for Phase 2 evaluation.

All functions expect (B, 1, H, W) tensors for true, mean, std.
"""

import math

import torch


def pixelwise_coverage(true, mean, std, level=0.95):
    """Fraction of pixels where true value falls within the prediction interval.

    Uses Gaussian assumption: interval = mean +/- z * std.
    """
    z = torch.distributions.Normal(0, 1).icdf(torch.tensor(0.5 + level / 2))
    lower = mean - z * std
    upper = mean + z * std
    covered = ((true >= lower) & (true <= upper)).float()
    return covered.mean()


def crps_gaussian(true, mean, std):
    """Continuous Ranked Probability Score for Gaussian predictive distribution.

    CRPS(N(mu, sigma^2), y) = sigma * [z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]
    where z = (y - mu) / sigma, Phi = CDF, phi = PDF.

    Returns per-sample mean CRPS, shape (B,).
    """
    z = (true - mean) / std.clamp(min=1e-8)
    normal = torch.distributions.Normal(0, 1)
    crps_pixel = std * (
        z * (2 * normal.cdf(z) - 1)
        + 2 * normal.log_prob(z).exp()
        - 1.0 / math.sqrt(math.pi)
    )
    return crps_pixel.reshape(true.shape[0], -1).mean(dim=1)


def calibration_error(true, mean, std, n_bins=10):
    """Mean absolute calibration error across quantile levels.

    For each nominal level p in [0.1, ..., 0.9], compute empirical
    coverage and return mean |empirical - nominal|.
    """
    levels = torch.linspace(0.1, 0.9, n_bins)
    errors = []
    for p in levels:
        empirical = pixelwise_coverage(true, mean, std, level=p.item())
        errors.append((empirical - p).abs())
    return torch.stack(errors).mean()


def sharpness(std):
    """Mean prediction interval width (proportional to mean std)."""
    return std.mean()
```

**Step 4: Run tests**

```bash
pytest tests/test_uq_metrics.py -v
```

Expected: all passed

**Step 5: Commit**

```bash
git add src/diffphys/evaluation/uq_metrics.py tests/test_uq_metrics.py
git commit -m "feat: UQ metrics — coverage, CRPS, calibration, sharpness

Pixel-wise Gaussian coverage at arbitrary levels. Closed-form Gaussian
CRPS. Calibration error as mean |empirical - nominal| across quantile
bins. Sharpness as mean prediction std."
```

---

### Task 11: Phase 2 Dataset + Evaluation Integration

**Files:**
- Modify: `src/diffphys/data/dataset.py`
- Create: `configs/ensemble_phase2.yaml`
- Create: `configs/ddpm_phase2.yaml`
- Create: `src/diffphys/evaluation/evaluate_uq.py`
- Create: `scripts/evaluate_phase2.py`

**Context:** Phase 2 training uses observation regime augmentation -- at each __getitem__, a regime is randomly applied to the BCs. Phase 2 evaluation runs ensemble and DDPM on test splits under each specific regime and computes UQ metrics. Matched-sample comparison: 5 ensemble members vs 5 DDPM samples.

**Step 1: Add regime support to dataset**

Modify `src/diffphys/data/dataset.py` to add an optional `regime` parameter:

```python
"""PyTorch dataset for Laplace PDE solution fields."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .conditioning import encode_conditioning
from .observation import apply_observation_regime, REGIMES


class LaplacePDEDataset(Dataset):
    """Loads .npz files produced by diffphys.pde.generate.

    Each sample returns (conditioning, target) where:
      conditioning: (8, nx, nx) float32 tensor
      target: (1, nx, nx) float32 tensor

    Args:
        npz_path: Path to .npz file.
        regime: Observation regime. "exact" for Phase 1,
            "mixed" for random regime per sample (Phase 2 training),
            or a specific regime name for evaluation.
    """

    def __init__(self, npz_path, regime="exact"):
        data = np.load(npz_path)
        self.fields = torch.from_numpy(data["fields"])
        self.bc_top = torch.from_numpy(data["bc_top"])
        self.bc_bottom = torch.from_numpy(data["bc_bottom"])
        self.bc_left = torch.from_numpy(data["bc_left"])
        self.bc_right = torch.from_numpy(data["bc_right"])
        self.regime = regime

    def __len__(self):
        return self.fields.shape[0]

    def __getitem__(self, idx):
        regime = self.regime
        if regime == "mixed":
            regime = REGIMES[torch.randint(len(REGIMES), (1,)).item()]

        bcs = [self.bc_top[idx], self.bc_bottom[idx],
               self.bc_left[idx], self.bc_right[idx]]

        if regime == "exact":
            cond = encode_conditioning(*bcs)
        else:
            obs_bcs = []
            masks = []
            for bc in bcs:
                obs, mask = apply_observation_regime(bc, regime)
                obs_bcs.append(obs)
                masks.append(mask)
            cond = encode_conditioning(*obs_bcs, *masks)

        target = self.fields[idx].unsqueeze(0)
        return cond, target
```

**Step 2: Write Phase 2 configs**

Create `configs/ensemble_phase2.yaml`:

```yaml
model:
  type: unet
  in_channels: 8
  out_channels: 1
  base_channels: 64
  channel_mult: [1, 2, 4]

ensemble:
  n_members: 5
  seeds: [200, 201, 202, 203, 204]

training:
  batch_size: 64
  lr: 1.0e-3
  epochs: 50
  weight_decay: 1.0e-4
  scheduler: cosine
  regime: mixed

data:
  train: data/train.npz
  val: data/val.npz

logging:
  log_dir: experiments/ensemble_phase2
  save_every: 10
```

Create `configs/ddpm_phase2.yaml`:

```yaml
model:
  type: unet
  in_channels: 9
  out_channels: 1
  base_channels: 64
  channel_mult: [1, 2, 4]
  time_emb_dim: 256

ddpm:
  T: 200
  beta_start: 1.0e-4
  beta_end: 0.02

training:
  batch_size: 64
  lr: 2.0e-4
  epochs: 100
  weight_decay: 0.0
  scheduler: cosine
  regime: mixed

data:
  train: data/train.npz
  val: data/val.npz

logging:
  log_dir: experiments/ddpm_phase2
  save_every: 20
```

**Step 3: Write UQ evaluation module**

Create `src/diffphys/evaluation/evaluate_uq.py`:

```python
"""Phase 2 UQ evaluation: ensemble vs DDPM under observation regimes."""

import json
from pathlib import Path

import torch
import numpy as np

from ..data.dataset import LaplacePDEDataset
from ..data.observation import REGIMES
from ..model.trainer import build_model, load_config
from ..model.ensemble import EnsemblePredictor
from ..model.ddpm import DDPM
from .uq_metrics import pixelwise_coverage, crps_gaussian, calibration_error, sharpness


def evaluate_ensemble_uq(ensemble, loader, device):
    """Evaluate ensemble UQ metrics."""
    all_true, all_mean, all_std = [], [], []

    for cond, target in loader:
        cond, target = cond.to(device), target.to(device)
        mean, var = ensemble.predict(cond)
        std = var.sqrt().clamp(min=1e-8)
        all_true.append(target.cpu())
        all_mean.append(mean.cpu())
        all_std.append(std.cpu())

    true = torch.cat(all_true)
    mean = torch.cat(all_mean)
    std = torch.cat(all_std)

    return _compute_uq_summary(true, mean, std)


def evaluate_ddpm_uq(ddpm, loader, device, n_samples=5):
    """Evaluate DDPM UQ metrics using sample mean/std."""
    all_true, all_mean, all_std = [], [], []

    for cond, target in loader:
        cond, target = cond.to(device), target.to(device)
        samples = ddpm.sample(cond, n_samples=n_samples)  # (K, B, 1, H, W)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0).clamp(min=1e-8)
        all_true.append(target.cpu())
        all_mean.append(mean.cpu())
        all_std.append(std.cpu())

    true = torch.cat(all_true)
    mean = torch.cat(all_mean)
    std = torch.cat(all_std)

    return _compute_uq_summary(true, mean, std)


def _compute_uq_summary(true, mean, std):
    return {
        "coverage_50": pixelwise_coverage(true, mean, std, 0.50).item(),
        "coverage_90": pixelwise_coverage(true, mean, std, 0.90).item(),
        "coverage_95": pixelwise_coverage(true, mean, std, 0.95).item(),
        "crps": crps_gaussian(true, mean, std).mean().item(),
        "calibration_error": calibration_error(true, mean, std).item(),
        "sharpness": sharpness(std).item(),
    }


def run_phase2_evaluation(model_type, config_path, checkpoint_paths,
                          test_npz, device="cpu"):
    """Evaluate a model under all observation regimes.

    Args:
        model_type: "ensemble" or "ddpm".
        config_path: YAML config path.
        checkpoint_paths: List of paths (ensemble) or single path (ddpm).
        test_npz: Path to test .npz file.
        device: Compute device.

    Returns:
        Dict mapping regime -> UQ metrics.
    """
    config = load_config(config_path)
    results = {}

    for regime in REGIMES:
        ds = LaplacePDEDataset(test_npz, regime=regime)
        loader = torch.utils.data.DataLoader(ds, batch_size=32)

        if model_type == "ensemble":
            models = []
            for cp in checkpoint_paths:
                m = build_model(config["model"]).to(device)
                ckpt = torch.load(cp, map_location=device, weights_only=True)
                m.load_state_dict(ckpt["model_state_dict"])
                models.append(m)
            ensemble = EnsemblePredictor(models)
            results[regime] = evaluate_ensemble_uq(ensemble, loader, device)

        elif model_type == "ddpm":
            model = build_model(config["model"]).to(device)
            ddpm = DDPM(model, **config["ddpm"]).to(device)
            ckpt = torch.load(checkpoint_paths[0], map_location=device, weights_only=True)
            ddpm.load_state_dict(ckpt["model_state_dict"])
            results[regime] = evaluate_ddpm_uq(ddpm, loader, device, n_samples=5)

    return results
```

**Step 4: Write Phase 2 evaluation script**

Create `scripts/evaluate_phase2.py`:

```python
"""Evaluate Phase 2 models (ensemble and DDPM) under all observation regimes.

Usage:
    python scripts/evaluate_phase2.py --model-type ensemble \
        --config configs/ensemble_phase2.yaml \
        --checkpoints experiments/ensemble_phase2/member_*/best.pt

    python scripts/evaluate_phase2.py --model-type ddpm \
        --config configs/ddpm_phase2.yaml \
        --checkpoints experiments/ddpm_phase2/best.pt
"""

import argparse
import json

from diffphys.evaluation.evaluate_uq import run_phase2_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["ensemble", "ddpm"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--test-npz", default="data/test_in.npz")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = run_phase2_evaluation(
        args.model_type, args.config, args.checkpoints,
        args.test_npz, args.device,
    )

    for regime, metrics in results.items():
        print(f"\n=== {regime} ===")
        for name, val in metrics.items():
            print(f"  {name:20s}: {val:.6f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
```

**Step 5: Commit**

```bash
git add src/diffphys/data/dataset.py src/diffphys/data/observation.py \
    src/diffphys/evaluation/evaluate_uq.py scripts/evaluate_phase2.py \
    configs/ensemble_phase2.yaml configs/ddpm_phase2.yaml
git commit -m "feat: Phase 2 observation regime training and UQ evaluation

Dataset supports regime='mixed' for Phase 2 training augmentation.
UQ evaluation runs ensemble/DDPM under all 5 observation regimes.
Matched-sample comparison: 5 ensemble members vs 5 DDPM samples."
```

---

### Task 12: Physics-Informed DDPM (Stretch Goal)

**Files:**
- Create: `src/diffphys/model/physics_ddpm.py`
- Create: `configs/physics_ddpm.yaml`
- Create: `tests/test_physics_ddpm.py`

**Context:** Stretch goal. Adds PDE-residual regularization to the DDPM loss. At each training step, we compute the Laplacian of the denoised estimate (x_0 hat from the epsilon prediction) and penalize non-zero residuals. The residual weight is timestep-dependent: lower at high-noise timesteps (where the estimate is unreliable) and higher at low-noise timesteps.

**Step 1: Write tests**

Create `tests/test_physics_ddpm.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_physics_ddpm.py -v
```

**Step 3: Write implementation**

Create `src/diffphys/model/physics_ddpm.py`:

```python
"""Physics-informed DDPM with PDE-residual regularization.

Adds a Laplacian residual penalty to the standard epsilon-prediction
loss. The penalty is weighted by a timestep-dependent factor:
  w(t) = residual_weight * alpha_bar_t
so that the physics loss is strongest when the denoised estimate
is most reliable (low noise, t near 0).
"""

import torch
import torch.nn.functional as F

from .ddpm import DDPM


class PhysicsDDPM(DDPM):
    """DDPM with PDE-residual loss.

    Args:
        model: ConditionalUNet with time embedding.
        T: Number of diffusion timesteps.
        residual_weight: Base weight for PDE residual loss.
        beta_start, beta_end: Schedule endpoints.
    """

    def __init__(self, model, T=200, residual_weight=0.1,
                 beta_start=1e-4, beta_end=0.02):
        super().__init__(model, T, beta_start, beta_end)
        self.residual_weight = residual_weight

    def _compute_x0_hat(self, x_t, pred_noise, t):
        """Estimate x_0 from x_t and predicted noise."""
        ab = self.schedule.alpha_bars[t]
        while ab.dim() < x_t.dim():
            ab = ab.unsqueeze(-1)
        return (x_t - (1 - ab).sqrt() * pred_noise) / ab.sqrt().clamp(min=1e-8)

    def _laplacian_residual(self, field, h=1.0 / 63):
        """RMS of discrete Laplacian on interior."""
        f = field[:, 0]  # (B, H, W)
        lap = (
            f[:, :-2, 1:-1] + f[:, 2:, 1:-1]
            + f[:, 1:-1, :-2] + f[:, 1:-1, 2:]
            - 4 * f[:, 1:-1, 1:-1]
        ) / (h ** 2)
        return lap.pow(2).mean(dim=(1, 2))

    def training_step(self, conditioning, target):
        """Compute MSE + PDE residual loss.

        Returns dict with "total", "mse", "residual" losses.
        """
        device = target.device
        self.schedule.to(device)

        B = target.shape[0]
        t = torch.randint(1, self.T + 1, (B,), device=device)
        noise = torch.randn_like(target)
        x_t = self.schedule.add_noise(target, noise, t)

        model_input = torch.cat([x_t, conditioning], dim=1)
        pred_noise = self.model(model_input, t)

        mse_loss = F.mse_loss(pred_noise, noise)

        # Denoised estimate for physics loss
        x0_hat = self._compute_x0_hat(x_t, pred_noise, t)
        residual = self._laplacian_residual(x0_hat)

        # Timestep-dependent weight: stronger at low noise
        ab = self.schedule.alpha_bars[t]
        weighted_residual = (ab * residual).mean()
        residual_loss = self.residual_weight * weighted_residual

        total = mse_loss + residual_loss

        return {"total": total, "mse": mse_loss, "residual": residual_loss}
```

**Step 4: Write config**

Create `configs/physics_ddpm.yaml`:

```yaml
model:
  type: unet
  in_channels: 9
  out_channels: 1
  base_channels: 64
  channel_mult: [1, 2, 4]
  time_emb_dim: 256

ddpm:
  T: 200
  beta_start: 1.0e-4
  beta_end: 0.02

physics:
  residual_weight: 0.1

training:
  batch_size: 64
  lr: 2.0e-4
  epochs: 100
  weight_decay: 0.0
  scheduler: cosine

data:
  train: data/train.npz
  val: data/val.npz

logging:
  log_dir: experiments/physics_ddpm
  save_every: 20
```

**Step 5: Run tests**

```bash
pytest tests/test_physics_ddpm.py -v
```

Expected: all passed

**Step 6: Commit**

```bash
git add src/diffphys/model/physics_ddpm.py tests/test_physics_ddpm.py configs/physics_ddpm.yaml
git commit -m "feat: physics-informed DDPM with PDE-residual regularization

Adds Laplacian residual penalty to epsilon-prediction loss. Weight is
timestep-dependent (alpha_bar_t) so physics loss is strongest at low
noise. Stretch goal for Phase 3."
```

---

### Task 13: Modal Deployment

**Files:**
- Create: `modal/train_remote.py`
- Create: `modal/requirements.txt`

**Context:** Training runs on Modal A100 GPUs. The Modal script wraps the training functions with a Modal App, mounts the codebase, and stores checkpoints in a Modal Volume. Budget: ~9 hours A100 total.

**Step 1: Write Modal training script**

Create `modal/requirements.txt`:

```
torch>=2.0
numpy
scipy
pyyaml
```

Create `modal/train_remote.py`:

```python
"""Remote training on Modal A100 GPUs.

Usage:
    modal run modal/train_remote.py --config configs/unet_regressor.yaml
    modal run modal/train_remote.py --config configs/ddpm.yaml
"""

import modal

app = modal.App("diffphys-training")

volume = modal.Volume.from_name("diffphys-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("modal/requirements.txt")
    .copy_local_dir("src", "/root/src")
    .copy_local_dir("configs", "/root/configs")
    .run_commands("cd /root && pip install -e .")
)


@app.function(
    image=image,
    gpu="A100",
    timeout=3600 * 4,
    volumes={"/data": volume},
)
def train_model(config_path: str):
    import torch
    from diffphys.model.trainer import load_config, train, train_ddpm

    config = load_config(config_path)

    # Point data paths to the volume mount
    for key in ("train", "val"):
        if key in config.get("data", {}):
            original = config["data"][key]
            config["data"][key] = original.replace("data/", "/data/")

    config["logging"]["log_dir"] = config["logging"]["log_dir"].replace(
        "experiments/", "/data/experiments/"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "ddpm" in config:
        train_ddpm(config, device=device)
    else:
        train(config, device=device)

    volume.commit()


@app.function(
    image=image,
    gpu="A100",
    timeout=3600 * 6,
    volumes={"/data": volume},
)
def train_ensemble(config_path: str):
    import torch
    from diffphys.model.trainer import load_config, train

    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for key in ("train", "val"):
        config["data"][key] = config["data"][key].replace("data/", "/data/")

    seeds = config["ensemble"]["seeds"]
    for i, seed in enumerate(seeds):
        print(f"\n=== Training ensemble member {i+1}/{len(seeds)} (seed={seed}) ===")
        torch.manual_seed(seed)

        member_config = {**config}
        member_config["logging"] = {
            **config["logging"],
            "log_dir": config["logging"]["log_dir"].replace(
                "experiments/", "/data/experiments/"
            ) + f"/member_{i}",
        }

        train(member_config, device=device)

    volume.commit()


@app.local_entrypoint()
def main(config: str):
    import yaml
    with open(config) as f:
        cfg = yaml.safe_load(f)

    if "ensemble" in cfg:
        train_ensemble.remote(config)
    else:
        train_model.remote(config)
```

**Step 2: Upload data to Modal volume**

```bash
# One-time data upload
modal volume put diffphys-data data/train.npz /train.npz
modal volume put diffphys-data data/val.npz /val.npz
modal volume put diffphys-data data/test_in.npz /test_in.npz
modal volume put diffphys-data data/test_ood.npz /test_ood.npz
```

**Step 3: Commit**

```bash
git add modal/train_remote.py modal/requirements.txt
git commit -m "feat: Modal A100 deployment for remote training

Wraps training functions with Modal App. Data stored in Modal Volume.
Supports both single-model and ensemble training. Budget: ~9hr A100."
```

---

### Task 14: Final Integration and Verification

**Files:**
- Create: `scripts/run_all_experiments.sh`

**Context:** End-to-end script that trains all models and runs evaluation. Also serves as documentation of the full experiment pipeline.

**Step 1: Write orchestration script**

Create `scripts/run_all_experiments.sh`:

```bash
#!/bin/bash
set -euo pipefail

echo "=== Phase 1: Deterministic Baselines ==="

echo "Training U-Net regressor..."
python scripts/train.py --config configs/unet_regressor.yaml --device cuda

echo "Training FNO..."
python scripts/train.py --config configs/fno.yaml --device cuda

echo "Evaluating Phase 1 models..."
python scripts/evaluate.py --config configs/unet_regressor.yaml \
    --checkpoint experiments/unet_regressor/best.pt \
    --output experiments/unet_regressor/results.json --device cuda

python scripts/evaluate.py --config configs/fno.yaml \
    --checkpoint experiments/fno/best.pt \
    --output experiments/fno/results.json --device cuda

echo "=== Phase 2: Probabilistic Models ==="

echo "Training ensemble (5 members)..."
for i in 0 1 2 3 4; do
    seed=$((100 + i))
    log_dir="experiments/ensemble/member_${i}"
    python scripts/train.py --config configs/ensemble.yaml --device cuda
done

echo "Training DDPM..."
python scripts/train.py --config configs/ddpm.yaml --device cuda

echo "Phase 2 UQ evaluation..."
python scripts/evaluate_phase2.py --model-type ensemble \
    --config configs/ensemble.yaml \
    --checkpoints experiments/ensemble/member_*/best.pt \
    --output experiments/ensemble/uq_results.json --device cuda

python scripts/evaluate_phase2.py --model-type ddpm \
    --config configs/ddpm.yaml \
    --checkpoints experiments/ddpm/best.pt \
    --output experiments/ddpm/uq_results.json --device cuda

echo "=== Done ==="
```

**Step 2: Run all tests**

```bash
pytest tests/ -v
```

Expected: 120+ tests passing (54 existing + ~70 new from Tasks 2-12)

**Step 3: Commit**

```bash
git add scripts/run_all_experiments.sh
git commit -m "feat: orchestration script for full experiment pipeline

Trains U-Net, FNO, ensemble (5 members), and DDPM. Evaluates Phase 1
with physics metrics and Phase 2 with UQ metrics under 5 observation
regimes. Matched-sample comparison: 5 ensemble vs 5 DDPM samples."
```

---

## Execution Order Summary

| Task | Component | New Tests | Dependencies |
|------|-----------|-----------|-------------|
| 2 | Dataset + Conditioning | ~18 | -- |
| 3 | U-Net Architecture | ~9 | -- |
| 4 | Training Infrastructure | ~6 | Tasks 2, 3 |
| 5 | FNO Architecture | ~5 | Task 4 |
| 6 | Physics-Aware Metrics | ~14 | Task 2 |
| 7 | Deep Ensemble | ~6 | Task 3 |
| 8 | DDPM | ~11 | Tasks 3, 4 |
| 9 | Observation Regimes | ~11 | -- |
| 10 | UQ Metrics | ~10 | -- |
| 11 | Phase 2 Integration | ~0 (integration) | Tasks 7-10 |
| 12 | Physics DDPM (stretch) | ~5 | Task 8 |
| 13 | Modal Deployment | ~0 (infra) | Task 4 |
| 14 | Final Integration | ~0 (orchestration) | All |

**Parallelizable tasks:** Tasks 2, 3, 9, 10 have no dependencies on each other.
