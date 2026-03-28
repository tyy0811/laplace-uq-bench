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
        if self.time_emb is not None:
            if t is None:
                raise ValueError(
                    "ConditionalUNet has time_emb_dim set but t was not passed. "
                    "DDPM mode requires a timestep tensor."
                )
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
