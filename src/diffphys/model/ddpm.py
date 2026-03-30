"""Conditional Denoising Diffusion Probabilistic Model.

T=200, linear beta schedule (1e-4 to 0.02), epsilon prediction.
Uses ConditionalUNet(in_ch=9, time_emb_dim=256) as noise predictor.
Input to U-Net: concatenation of noisy field (1 ch) and conditioning (8 ch).

Improved DDPM extensions:
- Cosine noise schedule with zero-terminal SNR (Nichol & Dhariwal 2021)
- v-prediction parameterisation (Salimans & Ho 2022)
- Min-SNR-gamma loss weighting (Hang et al. 2023)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def cosine_beta_schedule(T, s=0.008):
    """Cosine beta schedule with zero-terminal SNR.

    Implements the schedule from Nichol & Dhariwal (2021).
    Returns a (T+1,) tensor where index 0 is unused and indices 1..T
    hold the T beta values, matching the NoiseSchedule indexing convention.
    """
    steps = torch.arange(T + 1, dtype=torch.float64)  # 0 .. T
    # f(t) = cos^2( (t/T + s) / (1+s) * pi/2 )
    f = torch.cos(((steps / T) + s) / (1 + s) * (math.pi / 2)) ** 2
    # alpha_bar_t = f(t) / f(0)
    alpha_bars = f / f[0]
    # Enforce zero-terminal SNR: alpha_bar_T -> 0
    alpha_bars[-1] = 0.0
    # Derive betas: beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
    betas = torch.zeros(T + 1, dtype=torch.float64)
    betas[1:] = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
    betas = betas.clamp(0.0, 0.999)
    return betas.float()


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

class NoiseSchedule:
    """Beta schedule for DDPM (linear or cosine).

    Indexing convention: betas[0] is unused; betas[1] through betas[T]
    are the T noise levels.
    """

    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02, schedule="linear"):
        self.T = T
        if schedule == "cosine":
            betas = cosine_beta_schedule(T)
        elif schedule == "linear":
            betas = torch.zeros(T + 1)
            betas[1:] = torch.linspace(beta_start, beta_end, T)
        else:
            raise ValueError(f"Unknown schedule: {schedule!r}")
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


# ---------------------------------------------------------------------------
# Min-SNR-gamma loss weighting
# ---------------------------------------------------------------------------

def compute_min_snr_weight(alpha_bars, t, gamma=5.0):
    """Min-SNR-gamma weight for each sample in a batch.

    SNR(t) = alpha_bar_t / (1 - alpha_bar_t).
    Weight = min(SNR(t), gamma) / gamma, clamped to (0, 1].

    Args:
        alpha_bars: Full alpha_bar schedule tensor (T+1,).
        t: (B,) integer timestep indices.
        gamma: SNR clipping threshold.

    Returns:
        (B,) weight tensor.
    """
    ab = alpha_bars[t]
    snr = ab / (1.0 - ab).clamp(min=1e-8)
    return snr.clamp(max=gamma) / gamma


# ---------------------------------------------------------------------------
# v-prediction helpers
# ---------------------------------------------------------------------------

def compute_v_target(x_0, noise, alpha_bar_t):
    """Compute v-prediction target: v = sqrt(alpha_bar) * noise - sqrt(1-alpha_bar) * x_0."""
    return alpha_bar_t.sqrt() * noise - (1.0 - alpha_bar_t).sqrt() * x_0


def recover_x0_from_v(x_t, v_pred, alpha_bar_t):
    """Recover x_0 from v-prediction: x_0 = sqrt(alpha_bar) * x_t - sqrt(1-alpha_bar) * v."""
    return alpha_bar_t.sqrt() * x_t - (1.0 - alpha_bar_t).sqrt() * v_pred


# ---------------------------------------------------------------------------
# Improved DDPM
# ---------------------------------------------------------------------------

class ImprovedDDPM(DDPM):
    """Improved conditional DDPM with cosine schedule, v-prediction, and Min-SNR.

    Args:
        model: ConditionalUNet(in_ch=9, out_ch=1, time_emb_dim=...).
        T: Number of diffusion timesteps.
        prediction: "v" for v-prediction, "epsilon" for standard epsilon prediction.
        min_snr_gamma: Gamma for Min-SNR-gamma weighting. None to disable.
        beta_start, beta_end: Only used when schedule="linear".
        schedule: "cosine" (default) or "linear".
    """

    def __init__(self, model, T=200, prediction="v", min_snr_gamma=5.0,
                 beta_start=1e-4, beta_end=0.02, schedule="cosine"):
        # Bypass DDPM.__init__ to build our own schedule
        nn.Module.__init__(self)
        self.model = model
        self.T = T
        self.prediction = prediction
        self.min_snr_gamma = min_snr_gamma
        self.schedule = NoiseSchedule(T, beta_start, beta_end, schedule=schedule)

    def training_step(self, conditioning, target):
        """Compute loss with v-prediction (or epsilon) and Min-SNR weighting.

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
        pred = self.model(model_input, t)

        # Build target and per-sample loss
        if self.prediction == "v":
            ab = self.schedule.alpha_bars[t]
            while ab.dim() < target.dim():
                ab = ab.unsqueeze(-1)
            v_target = compute_v_target(target, noise, ab)
            per_sample_loss = (pred - v_target).pow(2).mean(dim=(1, 2, 3))
        elif self.prediction == "epsilon":
            per_sample_loss = (pred - noise).pow(2).mean(dim=(1, 2, 3))
        else:
            raise ValueError(f"Unknown prediction mode: {self.prediction!r}")

        # Min-SNR weighting
        if self.min_snr_gamma is not None:
            weights = compute_min_snr_weight(
                self.schedule.alpha_bars, t, gamma=self.min_snr_gamma
            )
            per_sample_loss = per_sample_loss * weights

        return per_sample_loss.mean()

    @torch.no_grad()
    def sample(self, conditioning, n_samples=1):
        """Generate samples via reverse diffusion with v-prediction or epsilon.

        Args:
            conditioning: (B, 8, H, W).
            n_samples: Number of samples per conditioning input.

        Returns:
            (n_samples, B, 1, H, W) tensor.
        """
        if self.prediction == "epsilon":
            return super().sample(conditioning, n_samples=n_samples)

        # v-prediction reverse process
        device = conditioning.device
        self.schedule.to(device)
        B, _, H, W = conditioning.shape

        cond = conditioning.unsqueeze(0).expand(n_samples, -1, -1, -1, -1)
        cond = cond.reshape(n_samples * B, 8, H, W)

        x = torch.randn(n_samples * B, 1, H, W, device=device)

        for t in reversed(range(1, self.T + 1)):
            t_batch = torch.full((n_samples * B,), t, device=device, dtype=torch.long)

            model_input = torch.cat([x, cond], dim=1)
            v_pred = self.model(model_input, t_batch)

            alpha_t = self.schedule.alphas[t]
            alpha_bar_t = self.schedule.alpha_bars[t]
            beta_t = self.schedule.betas[t]

            # Recover predicted x_0 and epsilon from v
            ab = alpha_bar_t
            while ab.dim() < x.dim():
                ab = ab.unsqueeze(-1)
            pred_x0 = recover_x0_from_v(x, v_pred, ab)
            pred_noise = ab.sqrt() * v_pred + (1.0 - ab).sqrt() * x

            # Standard DDPM reverse step using predicted epsilon
            mean = (1.0 / alpha_t.sqrt()) * (
                x - (beta_t / (1 - alpha_bar_t).sqrt()) * pred_noise
            )

            if t > 1:
                z = torch.randn_like(x)
                x = mean + beta_t.sqrt() * z
            else:
                x = mean

        return x.reshape(n_samples, B, 1, H, W)
