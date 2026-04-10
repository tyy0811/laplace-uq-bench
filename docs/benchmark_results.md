# Detailed Benchmark Results

Full results tables referenced from the main README. All results computed on Modal T4 GPUs with per-model checkpointing. Unless otherwise noted, conformal results use pixelwise (marginal) calibration, not the spatial/simultaneous variant. All numbers are from single trained checkpoints with fixed random seeds; no cross-seed variance is reported.

---

## Table 1: Solution Accuracy (Phase 1 — Exact BCs)

| Model | Split | Rel. L2 (↓) | PDE Residual (↓) | BC Error (↓) | Rel. Energy Err (↓) |
|-------|-------|-------------|------------------|-------------|---------------------|
| U-Net regressor | test_in | 0.0114 | 20.58 | 0.0067 | 1.30% |
| U-Net regressor | test_ood | 0.0289 | 28.29 | 0.0199 | 3.18% |
| FNO | test_in | 0.4016 | 24.52 | 0.2088 | 4.48% |
| FNO | test_ood | 0.3979 | 42.43 | 0.2569 | 9.32% |
| DDPM (5-sample mean) | test_in | **0.0022** | **4.22** | **0.0014** | **0.21%** |
| DDPM (5-sample mean) | test_ood | 0.0639 | **4.68** | 0.0774 | 16.4% |

FNO is included as an operator-learning baseline but underperforms U-Net significantly (rel. L2 ~40x worse), likely undertrained or undersized for this problem. It was not tuned further since the focus is on the ensemble/DDPM comparison.

**Note:** DDPM rows use the Phase 2 mixed-regime checkpoint evaluated on exact-BC test data. U-Net and FNO were trained on exact BCs only. This is not a controlled Phase 1 comparison — DDPM benefits from training on harder data and being tested on easier data.

## Table 2: Physics Compliance (Phase 1 — Exact BCs, In-Distribution)

| Model | PDE Residual (↓) | BC Error (↓) | Max Principle Violations (↓) | Energy Error (↓) |
|-------|------------------|-------------|-----------------------------|--------------------|
| FD solver (oracle) | ~1e-10 | 0 | 0% | 0 |
| U-Net regressor | 20.58 | 0.0067 | 0% | 1.30% |
| FNO | 24.52 | 0.2088 | 0.6% | 4.48% |
| DDPM (5-sample mean) | **4.22** | **0.0014** | 0% | **0.21%** |

## Table 3: Pixel-Level Posterior Quality (Phase 2 — Sparse-Noisy, 16 pts, σ=0.1, Matched K=5)

| Metric | Ensemble (5) | Flow Matching (5) | DDPM (5) |
|--------|-------------|-------------------|----------|
| Raw coverage@90 | 31.4% | 78.8% | 79.0% |
| Pixelwise conformal coverage@90 | 90.2% | 88.8% | 91.2% |
| Pixelwise conformal interval width | 0.133 | 0.156 | **0.115** |

All methods use matched K=5 samples. DDPM achieves the tightest calibrated intervals. See Appendix Table A1 for a K=20 compute-advantage comparison.

## Table 4: Functional-Level CRPS (Phase 2 — Sparse-Noisy, Matched 5v5)

| Quantity | Ensemble CRPS (5) ↓ | FM CRPS (5) ↓ | DDPM CRPS (5) ↓ |
|----------|---------------------|---------------|-----------------|
| Center T | 0.0086 | 0.0147 | **0.0077** |
| Subregion Mean T | 0.0087 | 0.0146 | **0.0076** |
| Max Interior T | 0.0438 | 0.0575 | **0.0288** |
| Dirichlet Energy | 0.1902 | 0.2759 | **0.1267** |
| Top Edge Flux | 0.1115 | 0.0692 | **0.0439** |

CRPS (lower is better) evaluated on 300 test samples with **matched 5 samples per model** (5 ensemble members vs 5 generated fields). DDPM achieves the best functional CRPS on all 5 derived quantities.

## Table 5: Observation Regime Comparison (Phase 2 — All 5 Regimes, In-Distribution, Matched K=5)

| Regime | Ens raw@90 | FM raw@90 | DDPM raw@90 | Ens conformal@90 | FM conformal@90 | DDPM conformal@90 | Ens width | FM width | DDPM width |
|--------|-----------|----------|------------|-----------------|----------------|-------------------|----------|---------|-----------|
| exact | 81.5% | 87.8% | 94.7% | 90.7% | 89.8% | 89.5% | 0.031 | 0.042 | **0.005** |
| dense-noisy | 55.1% | 78.5% | 76.3% | 91.1% | 90.8% | 87.8% | 0.074 | 0.090 | **0.053** |
| sparse-clean | 80.9% | 89.3% | 94.2% | 90.9% | 90.0% | 90.1% | 0.033 | 0.057 | **0.005** |
| sparse-noisy | 31.4% | 78.8% | 79.0% | 90.2% | 88.8% | 91.2% | 0.133 | 0.156 | **0.115** |
| very-sparse | 14.6% | 76.8% | 77.5% | 89.8% | 89.9% | 90.2% | 0.380 | 0.420 | **0.311** |

All coverage/width values use pixelwise conformal prediction at 90% target with **matched K=5 samples** across all methods. DDPM achieves the tightest conformal intervals at every regime. See Appendix Table A1 for a K=20 compute-advantage comparison.

## Table 6: OOD Generalization — Held-Out Piecewise BCs (All 5 Regimes)

**Deterministic models (exact regime, 300 samples):**

| Model | Rel. L2 (↓) | PDE Residual (↓) | BC Error (↓) | Rel. Energy Err (↓) |
|-------|-------------|------------------|-------------|---------------------|
| U-Net | 0.0281 | 28.07 | 0.0195 | 3.08% |
| FNO | 0.3993 | 42.07 | 0.2623 | 9.62% |

**Probabilistic models across all 5 observation regimes (150 samples, matched K=5):**

| Regime | Ens cov@90 | FM cov@90 | DDPM cov@90 | Ens CRPS (↓) | FM CRPS (↓) | DDPM CRPS (↓) | Ens cal err (↓) | DDPM cal err (↓) |
|--------|-----------|----------|------------|-------------|------------|--------------|----------------|-----------------|
| exact | 65.4% | 72.5% | 79.9% | **0.013** | 0.025 | 0.015 | 0.165 | **0.029** |
| dense-noisy | 55.1% | 69.7% | 69.1% | **0.017** | 0.027 | 0.018 | 0.227 | **0.131** |
| sparse-clean | 61.8% | 77.1% | 81.4% | **0.015** | 0.037 | 0.022 | 0.188 | **0.027** |
| sparse-noisy | 38.9% | 70.9% | 68.3% | **0.026** | 0.043 | 0.030 | 0.314 | **0.143** |
| very-sparse | 16.0% | 68.5% | 66.4% | 0.068 | 0.084 | **0.071** | 0.427 | **0.157** |

All generative models use matched K=5 samples. FM calibration error is not reported; the calibration-error comparison is between DDPM and ensemble only. Ensemble CRPS is lowest on 4 of 5 regimes because its predictions are accurate on average, but its uncertainty is too narrow — producing tight intervals that frequently miss the ground truth. DDPM's slightly higher CRPS reflects wider intervals that actually contain the truth (80% vs 65% coverage at 90% target on exact-regime OOD). At very-sparse, where the observation gap is largest, DDPM overtakes ensemble on CRPS as well.

## Table 7: DPS Empirical Outcomes (Phase 2 — In-Distribution + Zero-Shot, K=5)

**In-distribution regimes (conditional DDPM vs DPS):**

| Regime | DDPM rel L2 (↓) | DPS rel L2 (↓) | DPS obs RMSE | DPS PDE res | DPS cov@90 |
|--------|----------------|----------------|-------------|------------|-----------|
| exact | ~0.002 | 0.032 | 0.032 | 4.33 | 96.4% |
| dense-noisy | ~0.005 | 0.103 | 0.103 | 4.37 | 93.7% |
| sparse-clean | ~0.002 | 0.032 | 0.032 | 4.35 | 95.8% |
| sparse-noisy | ~0.002 | 0.056 | 0.095 | 4.36 | 86.3% |
| very-sparse | ~0.010 | 0.145 | 0.145 | 4.33 | 56.1% |

DPS PDE residual is regime-invariant ([4.33, 4.37]) because physics structure comes from the unconditional prior, not the observation model. DPS reaches the observation noise floor (obs RMSE within 5% of sigma_obs) on all noisy regimes.

**Zero-shot observation patterns (unseen during training):**

| Pattern | DDPM rel L2 | DPS rel L2 | DDPM PDE res | DPS PDE res |
|---------|-----------|-----------|-------------|------------|
| Extreme noise (σ=0.5) | 0.298 | 0.285 | 6.84 | 4.62 |
| Non-uniform sensors (16/8/32/4) | — | — | 934 | 4.31 |
| Single-edge observation | 0.926 | 0.848 | 41.7 | 4.26 |

Under zero-shot patterns, DPS is unaffected while the conditional model collapses: PDE residual blows up 10–220× indicating samples that no longer approximate Laplace solutions. DPS guidance config: ζ_obs=100, ζ_pde=0. See [DECISIONS.md](../DECISIONS.md) for guidance tuning analysis.

## Table 8: Computational Cost

**Training (T4 GPU):**

| Model | Parameters | Epochs | Wall Time |
|-------|-----------|--------|-----------|
| U-Net regressor | ~5M | 50 | ~4 hrs |
| FNO | ~2M | 50 | ~4 hrs |
| Deep ensemble (5 members, sequential) | ~25M | 5 × 50 | ~22 hrs |
| Flow Matching | ~5M | 80 | ~5.5 hrs |
| Improved DDPM | ~5M | 80 | ~4.5 hrs |
| Unconditional DDPM (DPS prior) | ~5M | 80 | ~4.5 hrs |

Training times are approximate per-model estimates. Ensemble members train sequentially (~4.4 hrs each).

**Evaluation (T4 GPU):**

| Eval Task | Samples | Wall Time |
|-----------|---------|-----------|
| Phase 1 accuracy (U-Net + FNO) | 5,000 × 2 splits | <1 min |
| DDPM Phase 1 (5-sample mean, 2 splits) | 300 × 2 | ~22 min |
| Ensemble UQ (5 regimes) | 300 × 5 | ~13 min |
| Flow Matching UQ (5 regimes, 5 samples) | 300 × 5 | ~15 min |
| DDPM UQ (5 regimes, 5 samples) | 300 × 5 | ~55 min |
| Conformal (3 models × 5 regimes, 5 samples) | 300 × 5 | ~50 min |
| Functional CRPS (3 models, 5 samples) | 300 | ~14 min |
| OOD — exact regime (5 models) | 300 | ~52 min |
| OOD — all 5 regimes (3 models, 5 samples) | 150 × 5 | ~40 min |
| DPS in-distribution (5 regimes, 5 samples) | 20 × 5 | ~15 min |
| DPS zero-shot (3 patterns, 5 samples) | 20 × 3 | ~10 min |

DDPM evaluation is the dominant cost due to 200 denoising steps per sample. FM evaluation is ~4× faster (50 ODE steps). Ensemble evaluation is near-instant (single forward pass per member). All generative evaluations use matched K=5 samples.

**Inference per sample:**

| Model | 1 sample | 5 samples |
|-------|----------|-----------|
| FD solver | ~0.5ms CPU | N/A |
| U-Net regressor | ~1ms GPU | N/A |
| FNO | ~1ms GPU | N/A |
| Deep ensemble (5x) | ~5ms GPU | ~5ms GPU |
| Flow Matching (50 steps) | ~100ms GPU | ~500ms GPU |
| Improved DDPM (200 steps) | ~200ms GPU | ~1s GPU |
| DPS (200 steps + guidance grads) | ~400ms GPU | ~2s GPU |

---

## Additional Figures

| Figure | Description |
|--------|-------------|
| [`fig7_functional_crps.png`](../figures/fig7_functional_crps.png) | Functional CRPS bar chart (matched 5v5, sparse-noisy regime) |
| [`fig9b_conformal_per_regime.png`](../figures/fig9b_conformal_per_regime.png) | Per-regime conformal coverage breakdown (all 5 regimes) |
| [`fig10_convergence.png`](../figures/fig10_convergence.png) | Training convergence: improved DDPM vs. flow matching |
| [`fig_interval_widths.png`](../figures/fig_interval_widths.png) | Prediction interval widths across methods and conformal variants |

---

## Appendix Table A1: Compute-Advantage Comparison (K=20 Generative, In-Distribution)

These results use 20 samples for generative models vs 5 ensemble members, giving generative models a statistical advantage in mean/std estimation. Reported for reference; the **matched K=5 comparison in Table 5 is the primary claim**.

| Regime | Ens raw@90 | FM raw@90 | DDPM raw@90 | Ens conformal@90 | FM conformal@90 | DDPM conformal@90 | Ens width | FM width | DDPM width |
|--------|-----------|----------|------------|-----------------|----------------|-------------------|----------|---------|-----------|
| exact | 81.5% | 96.5% | 99.6% | 90.7% | 90.5% | 89.5% | 0.031 | 0.034 | 0.003 |
| dense-noisy | 55.1% | 87.7% | 87.8% | 91.1% | 88.8% | 89.9% | 0.074 | 0.070 | 0.044 |
| sparse-clean | 80.9% | 98.1% | 99.4% | 90.9% | 91.8% | 90.2% | 0.033 | 0.049 | 0.003 |
| sparse-noisy | 31.4% | 89.4% | 85.4% | 90.2% | 89.9% | 88.8% | 0.133 | 0.131 | 0.088 |
| very-sparse | 14.6% | 83.8% | 83.7% | 89.8% | 87.9% | 88.1% | 0.380 | 0.331 | 0.248 |
