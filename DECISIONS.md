# Design Decisions

## Matched K=5 as Primary Comparison (Phase B, April 2026)

### Decision: All pixel-level UQ comparisons use matched K=5 samples

### Context

The original benchmark used K=20 samples for generative models (FM, DDPM) vs K=5 ensemble members for pixel-level coverage and interval width metrics. This gave generative models a statistical advantage in mean/std estimation. Functional CRPS (§11.3) already used matched K=5.

### Why matched K=5

The ensemble has exactly 5 members — that's a fixed architectural constraint, not a tunable parameter. Comparing K=20 generative vs K=5 ensemble conflates "generative models are better calibrated" with "more samples give better quantile estimates." The matched comparison isolates the former.

### Why Option B reporting (matched headline, K=20 in appendix)

Three options were considered: (A) matched only, (B) matched primary with K=20 secondary, (C) side-by-side. Option B preserves the information that generative models do improve with more samples (relevant for practitioners with compute budget) while keeping the headline comparison honest. The K=20 numbers are in Appendix Table A1 of benchmark_results.md.

### Why parameterize via Modal entrypoint argument (Option C)

The `n_samples_generative` parameter is threaded from `main()` through all `.spawn()`/`.remote()` calls. This allows reproducing K=20 numbers with a single flag (`--n-samples-generative 20`) without code changes. Alternative approaches (hardcoding, module-level constant) were rejected because they require editing code to switch between matched and unmatched comparisons.

### Why checkpoint validation with run_params

After parameterizing sample counts, stale K=20 partial checkpoints on the Modal volume could silently merge with new K=5 results during resume. All evaluation functions now save `n_samples_generative` in their checkpoint metadata and validate on resume via `_load_partial_with_validation`. Incompatible partials raise `ValueError` with "Use --fresh to start over."

---

## DPS Guidance Configuration (Phase C, April 2026)

### Decision: zeta_obs=100, zeta_pde=0, no gradient clipping

### Context

DPS (§10) uses dual guidance — measurement (zeta_obs) and physics (zeta_pde) — to steer unconditional samples toward consistency with observations and PDE structure. The guidance strengths were tuned on 20 validation examples from the sparse-noisy regime (16 observation points per edge, sigma_obs=0.1).

### Round 1: Initial grid search

Searched zeta_obs in {0.1, 0.5, 1.0, 5.0, 10.0} x zeta_pde in {0.0, 0.01, 0.1, 1.0} with grad_clip=1.0.

| zeta_obs | zeta_pde | Rel L2 | Obs RMSE | PDE Residual |
|----------|----------|--------|----------|--------------|
| 0.1 | 0.0 | 1.140 | 0.73 | 4.15 |
| 0.5 | 0.0 | 1.130 | 0.73 | 4.16 |
| 1.0 | 0.0 | 0.930 | 0.63 | 4.16 |
| 5.0 | 0.0 | 0.540 | 0.39 | 4.15 |
| 10.0 | 0.0 | 0.304 | 0.24 | 4.19 |
| 10.0 | 0.01 | 1.020 | 0.65 | 16.1 |
| 10.0 | 0.1 | 1.090 | 0.70 | 55.9 |
| 10.0 | 1.0 | 1.120 | 0.72 | 40.2 |

Finding: zeta_pde=0 consistently best at every zeta_obs level. Any nonzero physics weight degraded performance. Stronger measurement guidance monotonically improved accuracy with no plateau at zeta_obs=10. This monotonic improvement motivated the Round 2 extended grid up to zeta_obs=100; Round 1 alone would have led to under-tuning.

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

Distribution is tight and unimodal with slight right skew. No outliers above 0.15. Worst-case example (0.135) is still a meaningful reconstruction. (This pre-flight distribution is on 20 validation examples; the full §10.5 evaluation in benchmark_results.md Table 7 reports median rel L2 = 0.056 on the full test set.)

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

Note that since the BCs fully determine T via the FD solver, training on exact BCs gives the prior the marginal p(T) induced by the BC generation distribution (§2.4) — not a "free" prior over arbitrary harmonic functions. This is appropriate for the benchmark but means the prior's coverage of solution fields inherits the structure of the BC family.

### Why train unconditional prior on exact BCs only

The unconditional model trains on clean Laplace solutions (regime: exact in the config). It never sees noisy or sparse boundary conditions during training — unlike the conditional models which train on mixed regimes (§5.4). This is intentional: the unconditional prior should learn p(T), the distribution over solution fields, not p(T|observations). Observation information enters only at inference via the guidance gradient. Training on mixed-regime data would conflate the prior with the observation model, defeating the purpose of the unconditional approach.

### Why mean-normalized guidance losses

The measurement loss uses `.mean()` over spatial dimensions: `(y_obs - y_pred).pow(2).mean()`. This makes the guidance strength (zeta_obs=100) invariant to the number of observation points. If we used `.sum()`, doubling the observation count would double the effective guidance, requiring re-tuning zeta_obs for each regime. Mean normalization decouples the guidance strength from observation density, which is why a single zeta_obs works across all 5 regimes.

### What we would change at larger scale

1. Search zeta_obs > 100 to check for further improvement or eventual instability
2. Adaptive guidance scheduling (e.g., based on per-step observation RMSE)
3. Langevin-corrected DPS (Song et al., 2023) for tighter posterior approximation
4. Multi-resolution unconditional prior for higher-resolution fields
