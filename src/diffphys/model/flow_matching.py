"""Conditional Flow Matching with Optimal Transport.

Learns a velocity field v_theta(x_t, t) transporting samples from
N(0,I) (t=0) to data (t=1) along straight-line paths.

Interpolant: x_t = (1-t)*x_0 + t*x_1, x_0 ~ N(0,I), x_1 = data
Target: u_t = x_1 - x_0 (constant velocity)
Loss: ||v_theta(x_t, t, cond) - u_t||^2

OT-CFM: Hungarian OT coupling within mini-batch for straighter flows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatchingSchedule:
    """Manages interpolation and velocity targets for CFM."""

    def __init__(self, sigma_min=1e-4):
        self.sigma_min = sigma_min

    def sample_time(self, batch_size, device):
        """Sample t ~ Uniform(0, 1)."""
        return torch.rand(batch_size, device=device)

    def interpolate(self, x_0, x_1, t):
        """x_t = (1-t)*x_0 + t*x_1."""
        t = t[:, None, None, None]
        return (1 - t) * x_0 + t * x_1

    def compute_target(self, x_0, x_1):
        """Velocity target u_t = x_1 - x_0."""
        return x_1 - x_0


class OTCouplingMatcher:
    """Mini-batch optimal transport coupling for straighter flows."""

    @torch.no_grad()
    def find_coupling(self, x_0, x_1):
        """Find OT permutation within mini-batch.

        Uses squared-Euclidean cost with Hungarian algorithm to extract
        a true permutation (no duplicates).
        """
        from scipy.optimize import linear_sum_assignment

        B = x_0.shape[0]
        x0_flat = x_0.reshape(B, -1)
        x1_flat = x_1.reshape(B, -1)

        C = torch.cdist(x0_flat, x1_flat, p=2).pow(2)
        # For each target j, find the best source i (transpose so rows=targets)
        _, perm = linear_sum_assignment(C.T.cpu().numpy())
        perm = torch.tensor(perm, device=x_0.device, dtype=torch.long)
        return x_0[perm]


class ConditionalFlowMatcher(nn.Module):
    """Conditional flow matching wrapper.

    Args:
        model: ConditionalUNet(in_ch=9, out_ch=1, time_emb_dim=256).
        use_ot: Whether to use OT coupling.
        n_sample_steps: Euler steps for ODE sampling.
    """

    def __init__(self, model, use_ot=True, n_sample_steps=50):
        super().__init__()
        self.model = model
        self.schedule = FlowMatchingSchedule()
        self.ot_matcher = OTCouplingMatcher() if use_ot else None
        self.n_sample_steps = n_sample_steps

    def _predict_velocity(self, x_t, conditioning, t):
        """Forward pass: cat inputs and scale time."""
        model_input = torch.cat([x_t, conditioning], dim=1)
        t_scaled = t * 1000.0
        return self.model(model_input, t_scaled)

    def training_step(self, conditioning, target):
        """Compute flow matching MSE loss.

        Args:
            conditioning: (B, 8, H, W) BC conditioning tensor.
            target: (B, 1, H, W) clean solution field (x_1).

        Returns:
            Scalar loss.
        """
        B = target.shape[0]
        device = target.device

        x_0 = torch.randn_like(target)

        if self.ot_matcher is not None:
            x_0 = self.ot_matcher.find_coupling(x_0, target)

        t = self.schedule.sample_time(B, device)
        x_t = self.schedule.interpolate(x_0, target, t)
        velocity_target = self.schedule.compute_target(x_0, target)

        v_pred = self._predict_velocity(x_t, conditioning, t)
        return F.mse_loss(v_pred, velocity_target)

    @torch.no_grad()
    def sample(self, conditioning, n_samples=1):
        """Generate samples via Euler ODE integration.

        Args:
            conditioning: (B, 8, H, W).
            n_samples: Number of samples per conditioning input.

        Returns:
            (n_samples, B, 1, H, W) tensor.
        """
        device = conditioning.device
        B, _, H, W = conditioning.shape
        dt = 1.0 / self.n_sample_steps

        cond = conditioning.unsqueeze(0).expand(n_samples, -1, -1, -1, -1)
        cond = cond.reshape(n_samples * B, 8, H, W)

        x_t = torch.randn(n_samples * B, 1, H, W, device=device)

        for i in range(self.n_sample_steps):
            t = torch.full((n_samples * B,), i / self.n_sample_steps, device=device)
            v = self._predict_velocity(x_t, cond, t)
            x_t = x_t + v * dt

        return x_t.reshape(n_samples, B, 1, H, W)
