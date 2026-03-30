# Revised Phase 2 Implementation Plan

**Context:** DDPM at 60 epochs shows 16.8% coverage (target: 90%) with wide-but-wrong samples.
Diagnostics confirm conditioning works (cross-BC std=0.287) but model is undertrained
(loss still decreasing at epoch 60 when cosine LR hit zero). Within-BC sample diversity
is low (std=0.009), producing near-identical inaccurate samples.

**Revised project story:**
> Benchmarked three approaches to uncertainty quantification for PDE surrogates under noisy
> boundary observations: conditional flow matching, improved DDPM, and deep ensembles with
> conformal prediction — showing that flow matching converges 3-5x faster than DDPM while
> conformal prediction provides distribution-free coverage guarantees at minimal cost.

**GPU budget:** ~$10-12 total on T4
**Timeline:** 5-6 working days for core, +2 days for DPS stretch

---

## 0. Diagnostics (COMPLETED)

- **Loss curve:** Still decreasing at epoch 60. Undertraining confirmed.
- **Conditioning check:** PASS. Cross-BC std=0.287, within-BC std=0.009.
- **Verdict:** No structural issues. Proceed with plan.

---

## 1. Conditional Flow Matching (OT-CFM)

### Files
- `src/diffphys/model/flow_matching.py` — FlowMatchingSchedule, OTCouplingMatcher
- `src/diffphys/model/trainer.py` — add train_flow_matching functions
- `configs/flow_matching.yaml`
- `tests/test_flow_matching.py` — 8+ tests

### Architecture
- Reuses ConditionalUNet(in_ch=9, out_ch=1, time_emb_dim=256) unchanged
- Input: cat([x_t, conditioning], dim=1) — same as DDPM
- Time embedding: scale t * 1000.0 for sinusoidal embedding (t in [0,1])
- Training target: velocity u_t = x_1 - x_0 (constant, simpler than noise prediction)
- Sampling: 50 Euler ODE steps (vs DDPM's 200 reverse SDE steps)
- OT coupling: Sinkhorn on (B,B) cost matrix before computing interpolants

### Config
- epochs: 80 (CFM converges ~3x faster)
- batch_size: 64 (OT coupling benefits from larger batches)
- lr: 1e-4, cosine schedule
- regime: mixed
- ot_coupling: true
- n_sample_steps: 50

### Estimated cost: ~$2.80 (4.7 hrs T4)

---

## 2. Improved DDPM

### Changes to existing code (~55 LOC total)
1. **Min-SNR-gamma weighting** (~15 LOC) — clips SNR to reduce high-noise gradient conflicts
2. **Cosine noise schedule + zero-terminal SNR** (~10 LOC) — preserves signal longer
3. **v-prediction** (~30 LOC) — stable across all noise levels, replaces epsilon prediction

### Config
- epochs: 80 (3-5x more effective per epoch with improvements)
- batch_size: 32 (unchanged — isolate improvements as only variable)
- lr: 1e-4, cosine schedule
- regime: mixed

### Estimated cost: ~$3.20 (5.3 hrs T4)

---

## 3. Conformal Prediction on Ensemble

### Files
- `src/diffphys/evaluation/conformal.py` — SpatialConformalPredictor, PixelwiseConformalPredictor
- `tests/test_conformal.py` — 6+ tests

### Approach
- Split val set: 2500 calibration, 2500 validation
- Compute nonconformity scores: max(|y - mu| / sigma) per sample (spatial)
- Find quantile q_hat for target coverage via finite-sample correction
- Prediction bands: [mu +/- q_hat * sigma]
- Guarantee: P(coverage) >= 1 - alpha for any new sample

### Estimated cost: $0 (post-hoc wrapper on existing ensemble)

---

## 4. Evaluation Plan

### Phase 1 (already done)
- U-Net regressor: rel_l2=0.011 (test_in), 0.029 (test_ood)
- FNO: rel_l2=0.402 (test_in), 0.398 (test_ood)

### Phase 2 UQ (20 samples for generative models)
- Ensemble (raw): already evaluated, 86.4% coverage at 95% level (exact regime)
- Ensemble + conformal: expected >=90% guaranteed
- Flow matching (20 samples): TBD
- Improved DDPM (20 samples): TBD

### Evaluation on all 5 observation regimes
- exact, dense-noisy, sparse-clean, sparse-noisy, very-sparse

---

## 5. Execution Order

```
Day 1: [1] Flow matching implementation + tests
       [2] DDPM improvements implementation
       Launch FM training overnight

Day 2: [3] Conformal prediction implementation + tests
       Launch improved DDPM training
       Evaluate FM if done

Day 3: Evaluate FM and improved DDPM
       Run conformal evaluation (CPU)
       Generate comparison figures

Day 4: Fill result tables
       Final test suite pass
       Documentation and writeup
```

---

## 6. Batch Size Rationale

- **Flow matching: 64** — OT coupling quality scales with batch size (64x64 cost matrix).
  U-Net activations ~2-3GB, well within T4's 16GB. Add OOM fallback to 48.
- **Improved DDPM: 32** — Matches existing setup. Three simultaneous changes (Min-SNR,
  cosine schedule, v-prediction) need batch size held constant for controlled comparison.
