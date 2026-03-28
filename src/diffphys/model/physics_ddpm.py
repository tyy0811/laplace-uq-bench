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
