"""Diffusion Posterior Sampling with measurement + physics guidance.

Implements Chung et al. (ICLR 2023) with the DiffusionPDE (Huang et al.,
NeurIPS 2024) dual-guidance extension. See theory.md §10.

Uses a trained UnconditionalDDPM as the prior and incorporates observations
via gradient guidance at each reverse step.
"""

import torch
from typing import Callable, Optional, Tuple

from .ddpm import recover_x0_from_v


class DPSSampler:
    """Diffusion Posterior Sampling with measurement + physics guidance.

    Args:
        model: Trained UnconditionalDDPM.
        zeta_obs: Measurement guidance strength.
        zeta_pde: Physics guidance strength.
        guidance_schedule: Callable(t, T) -> (zeta_obs_t, zeta_pde_t).
            If None, uses linear anneal (full at t=T, 10% at t=1).
    """

    def __init__(self, model, zeta_obs=1.0, zeta_pde=0.1,
                 guidance_schedule: Optional[Callable] = None,
                 grad_clip: Optional[float] = 1.0):
        self.model = model
        self.zeta_obs = zeta_obs
        self.zeta_pde = zeta_pde
        self.guidance_schedule = guidance_schedule or self._default_schedule
        self.grad_clip = grad_clip

    def sample(self, y_obs: torch.Tensor, obs_operator: Callable,
               n_samples: int = 5, H: int = 64, W: int = 64
               ) -> torch.Tensor:
        """Sample from p(x | y_obs) using DPS.

        Args:
            y_obs: Noisy boundary observations (any shape; must match
                obs_operator output).
            obs_operator: Callable mapping (n_samples, 1, H, W) -> predicted
                observations (same shape as y_obs, broadcastable over batch).
            n_samples: Number of posterior samples.
            H, W: Spatial dimensions.

        Returns:
            (n_samples, 1, H, W) posterior samples.
        """
        device = next(self.model.parameters()).device
        self.model.schedule.to(device)
        y_obs = y_obs.to(device)
        T = self.model.T

        x_t = torch.randn(n_samples, 1, H, W, device=device)

        for t in reversed(range(1, T + 1)):
            x_t = self._dps_step(x_t, t, y_obs, obs_operator)

        return x_t

    def _dps_step(self, x_t: torch.Tensor, t: int,
                  y_obs: torch.Tensor,
                  obs_operator: Callable) -> torch.Tensor:
        """Single DPS reverse step with dual guidance.

        Implements equation (10.5) from theory.md.
        """
        n_samples = x_t.shape[0]
        device = x_t.device
        schedule = self.model.schedule

        # Enable gradient tracking for guidance
        x_t = x_t.detach().requires_grad_(True)

        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)

        # Forward pass through the model
        pred = self.model.model(x_t, t_batch)

        # Get noise schedule values
        alpha_bar_t = schedule.alpha_bars[t]
        alpha_t = schedule.alphas[t]
        beta_t = schedule.betas[t]

        # Compute Tweedie denoised estimate x0_hat
        ab = alpha_bar_t
        while ab.dim() < x_t.dim():
            ab = ab.unsqueeze(-1)

        if self.model.prediction == "v":
            x0_hat = recover_x0_from_v(x_t, pred, ab)
            pred_noise = ab.sqrt() * pred + (1.0 - ab).sqrt() * x_t
        else:
            pred_noise = pred
            x0_hat = (x_t - (1 - ab).sqrt() * pred_noise) / ab.sqrt()

        # Measurement guidance: ||y_obs - H(x0_hat)||^2 (mean-normalized)
        y_pred = obs_operator(x0_hat)
        loss_obs = (y_obs - y_pred).pow(2).mean()

        # Physics guidance: ||laplacian(x0_hat)||^2 (mean-normalized)
        loss_pde = self._laplacian_loss(x0_hat)

        # Compute guidance gradients
        zeta_obs_t, zeta_pde_t = self.guidance_schedule(t, self.model.T)
        total_loss = zeta_obs_t * loss_obs + zeta_pde_t * loss_pde
        guidance = torch.autograd.grad(total_loss, x_t)[0]

        # Clip gradient norm to prevent divergence
        if self.grad_clip is not None:
            grad_norm = guidance.norm()
            if grad_norm > self.grad_clip:
                guidance = guidance * (self.grad_clip / grad_norm)

        # Standard DDPM reverse step minus guidance
        with torch.no_grad():
            mean = (1.0 / alpha_t.sqrt()) * (
                x_t - (beta_t / (1 - alpha_bar_t).sqrt()) * pred_noise
            )

            if t > 1:
                z = torch.randn_like(x_t)
                x_t_prev = mean + beta_t.sqrt() * z - guidance
            else:
                x_t_prev = mean - guidance

        return x_t_prev.detach()

    @staticmethod
    def _laplacian_loss(x: torch.Tensor, h: float = 1.0 / 63) -> torch.Tensor:
        """Sum of squared discrete Laplacian on interior (differentiable).

        Args:
            x: (B, 1, H, W) field tensor with gradients enabled.

        Returns:
            Scalar loss.
        """
        f = x[:, 0]  # (B, H, W)
        lap = (
            f[:, :-2, 1:-1] + f[:, 2:, 1:-1]
            + f[:, 1:-1, :-2] + f[:, 1:-1, 2:]
            - 4 * f[:, 1:-1, 1:-1]
        ) / (h ** 2)
        return lap.pow(2).mean()

    def _default_schedule(self, t: int, T: int
                          ) -> Tuple[float, float]:
        """Linear anneal: full strength at t=T, 10% at t=1."""
        scale = 0.1 + 0.9 * (t / T)
        return self.zeta_obs * scale, self.zeta_pde * scale
