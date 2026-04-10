"""Unconditional Denoising Diffusion Probabilistic Model.

Learns a prior p(T) over Laplace solution fields without boundary condition
input. Used as the backbone for DPS (§10) — observations are incorporated
via gradient guidance at inference time, not through input conditioning.

Architecture: same ConditionalUNet backbone as the conditional DDPM, but
with in_channels=1 (field only, no conditioning channels).

Training improvements match the conditional ImprovedDDPM:
- Cosine noise schedule with zero-terminal SNR
- v-prediction parameterisation
- Min-SNR-gamma loss weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ddpm import (
    NoiseSchedule,
    compute_min_snr_weight,
    compute_v_target,
    recover_x0_from_v,
)


class UnconditionalDDPM(nn.Module):
    """Unconditional improved DDPM over Laplace solution fields.

    Args:
        model: ConditionalUNet(in_ch=1, out_ch=1, time_emb_dim=256).
        T: Number of diffusion timesteps.
        prediction: "v" for v-prediction, "epsilon" for standard.
        min_snr_gamma: Gamma for Min-SNR-gamma weighting. None to disable.
        schedule: "cosine" (default) or "linear".
    """

    def __init__(self, model, T=200, prediction="v", min_snr_gamma=5.0,
                 beta_start=1e-4, beta_end=0.02, schedule="cosine"):
        super().__init__()
        self.model = model
        self.T = T
        self.prediction = prediction
        self.min_snr_gamma = min_snr_gamma
        self.schedule = NoiseSchedule(T, beta_start, beta_end, schedule=schedule)

    def training_step(self, target):
        """Compute loss with v-prediction and Min-SNR weighting.

        Args:
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

        pred = self.model(x_t, t)

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

        if self.min_snr_gamma is not None:
            weights = compute_min_snr_weight(
                self.schedule.alpha_bars, t, gamma=self.min_snr_gamma
            )
            per_sample_loss = per_sample_loss * weights

        return per_sample_loss.mean()

    @torch.no_grad()
    def sample(self, n_samples, H=64, W=64, device=None):
        """Generate unconditional samples from the prior.

        Args:
            n_samples: Number of samples to generate.
            H, W: Spatial dimensions.
            device: Compute device.

        Returns:
            (n_samples, 1, H, W) tensor.
        """
        if device is None:
            device = next(self.parameters()).device
        self.schedule.to(device)

        x = torch.randn(n_samples, 1, H, W, device=device)

        for t in reversed(range(1, self.T + 1)):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)

            pred = self.model(x, t_batch)

            alpha_t = self.schedule.alphas[t]
            alpha_bar_t = self.schedule.alpha_bars[t]
            beta_t = self.schedule.betas[t]

            if self.prediction == "v":
                ab = alpha_bar_t
                while ab.dim() < x.dim():
                    ab = ab.unsqueeze(-1)
                pred_noise = ab.sqrt() * pred + (1.0 - ab).sqrt() * x
            else:
                pred_noise = pred

            mean = (1.0 / alpha_t.sqrt()) * (
                x - (beta_t / (1 - alpha_bar_t).sqrt()) * pred_noise
            )

            if t > 1:
                z = torch.randn_like(x)
                x = mean + beta_t.sqrt() * z
            else:
                x = mean

        return x
