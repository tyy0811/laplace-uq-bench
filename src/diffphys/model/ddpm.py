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
