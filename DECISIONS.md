# Design Decisions

## DPS Guidance Configuration (Phase C, April 2026)

### Decision: zeta_obs=100, zeta_pde=0, no gradient clipping

### Context

DPS (§10) uses dual guidance — measurement (zeta_obs) and physics (zeta_pde) — to steer unconditional samples toward consistency with observations and PDE structure. The guidance strengths were tuned on 20 validation examples from the sparse-noisy regime (16 observation points per edge, sigma_obs=0.1).

### Round 1: Initial grid search

Searched zeta_obs in {0.1, 0.5, 1.0, 5.0, 10.0} x zeta_pde in {0.0, 0.01, 0.1, 1.0} with grad_clip=1.0.

| zeta_obs | zeta_pde | Rel L2 | Obs RMSE | PDE Residual |
|----------|----------|--------|----------|--------------|
| 0.1 | 0.0 | 1.14 | 0.73 | 4.15 |
| 0.5 | 0.0 | 1.13 | 0.73 | 4.16 |
| 1.0 | 0.0 | 0.93 | 0.63 | 4.16 |
| 5.0 | 0.0 | 0.54 | 0.39 | 4.15 |
| 10.0 | 0.0 | 0.30 | 0.24 | 4.19 |
| 10.0 | 0.01 | 1.02 | 0.65 | 16.1 |
| 10.0 | 0.1 | 1.09 | 0.70 | 55.9 |
| 10.0 | 1.0 | 1.12 | 0.72 | 40.2 |

Finding: zeta_pde=0 consistently best at every zeta_obs level. Any nonzero physics weight degraded performance. Stronger measurement guidance monotonically improved accuracy with no plateau at zeta_obs=10.

### Round 2: Extended grid (higher zeta_obs + relaxed clipping)

Searched zeta_obs in {20, 50, 100} x grad_clip in {1.0, 5.0, None}, all with zeta_pde=0.

| zeta_obs | grad_clip | Rel L2 |
|----------|-----------|--------|
| 10.0 | 1.0 | 0.304 |
| 20.0 | 1.0 | 0.150 |
| 20.0 | 5.0 | 0.148 |
| 50.0 | 1.0 | 0.088 |
| 50.0 | 5.0 | 0.088 |
| 50.0 | None | 0.088 |
| 100.0 | 1.0 | 0.069 |
| 100.0 | 5.0 | 0.069 |
| 100.0 | None | 0.069 |
| 10.0 | 5.0 | 0.304 |
| 10.0 | None | 0.304 |

Finding: Accuracy improves monotonically up to zeta_obs=100. Gradient clipping has no effect at these strengths — the linear anneal schedule keeps gradients stable without clipping. No NaN at any configuration.

### Pre-flight: Fine zeta_pde sweep at zeta_obs=100

Tested zeta_pde in {0.0, 0.001, 0.005, 0.01, 0.05} at zeta_obs=100, grad_clip=None.

| zeta_pde | Rel L2 | Status |
|----------|--------|--------|
| 0.0 | 0.069 | Stable |
| 0.001 | NaN | Diverged |
| 0.005 | NaN | Diverged |
| 0.01 | NaN | Diverged |
| 0.05 | NaN | Diverged |

Finding: Every nonzero zeta_pde causes sampling to diverge to NaN when gradient clipping is disabled. This is not "physics guidance is unnecessary" — it is "physics guidance is actively destabilizing." The discrete Laplacian gradient through the Tweedie denoised mean amplifies noise at intermediate diffusion steps, consistent with the Jensen's Gap caveat in §6.1. The unconditional prior alone encodes enough elliptic structure that explicit PDE-residual gradients become redundant and harmful.

### Pre-flight: Per-example distribution at best settings

zeta_obs=100, zeta_pde=0, grad_clip=None, 20 validation examples (sparse-noisy).

| Metric | Value |
|--------|-------|
| Mean rel L2 | 0.069 |
| Median rel L2 | 0.061 |
| Std | 0.027 |
| IQR | [0.051, 0.085] |
| Min | 0.026 |
| Max | 0.135 |

Distribution is tight and unimodal with slight right skew. No outliers above 0.15. Worst-case example (0.135) is still a meaningful reconstruction.

### Pre-flight: Observation noise floor analysis

| Metric | Value |
|--------|-------|
| sigma_obs (sparse-noisy) | 0.100 |
| Mean obs RMSE | 0.095 |
| Mean obs_rmse / sigma_obs | 0.947 |
| Median ratio | 0.931 |
| Range | [0.816, 1.212] |

DPS reaches within 5% of the observation noise floor on average. 19 of 20 examples have ratio in [0.82, 1.06]. One outlier at 1.21 (likely an example with unusual boundary structure the prior doesn't represent well — a finite-data effect). The sub-unity mean ratio (0.95) suggests the prior provides a small amount of denoising beyond what the observations alone contain.

### Interpretation for §10.5

The 30x gap to conditional DDPM (rel L2 0.069 vs 0.002) is the cost of observation-model independence, not a failure of the method. Conditional DDPM was trained against the exact observation operator; DPS uses only a prior over solution fields. The relevant finding is that DPS saturates the information content of the observations — further improvement requires either denoising the observations or providing additional data, not stronger guidance.

### Why dual guidance (measurement + physics) vs measurement only

The theory in §10.3 motivates dual guidance following DiffusionPDE (Huang et al., NeurIPS 2024). On this benchmark, the physics term is actively harmful: every nonzero zeta_pde causes divergence without gradient clipping. The unconditional prior already learns the elliptic solution manifold from data — explicit PDE-residual gradients create a competing optimization signal. For problems where the prior is less well-matched to the PDE structure (e.g., turbulent flows, nonlinear PDEs), the physics term may become essential.

### Why exact autograd through x0_hat vs stop-gradient approximation

The DPS guidance gradient must flow through the Tweedie denoised estimate x0_hat to x_t. We use exact autograd (torch.autograd.grad) rather than a stop-gradient approximation. This is validated by test_dps.py::test_gradient_through_x0_hat_is_nonzero. The stop-gradient variant would zero out the guidance entirely, since the observation operator acts on x0_hat, not x_t directly.

### Why linear anneal for the guidance schedule

The guidance schedule scales linearly from 10% at t=1 to 100% at t=T. Stronger guidance early (high t, noisy samples) steers the broad distribution toward observation-consistent modes; weaker guidance late (low t, near-clean samples) avoids overriding fine details. The anneal also explains why gradient clipping is unnecessary — the natural decay of guidance strength at low t prevents late-step instability.

### Why reuse conditional DDPM backbone with in_channels=1

The unconditional model uses the same ConditionalUNet architecture with in_ch=1 instead of 9. This ensures capacity is identical — any performance difference between conditional and unconditional is due to the conditioning signal, not architecture. The ~4.5M parameter count is sufficient for 64x64 Laplace solutions.

### What we would change at larger scale

1. Search zeta_obs > 100 to check for further improvement or eventual instability
2. Adaptive guidance scheduling (e.g., based on per-step observation RMSE)
3. Langevin-corrected DPS (Song et al., 2023) for tighter posterior approximation
4. Multi-resolution unconditional prior for higher-resolution fields
