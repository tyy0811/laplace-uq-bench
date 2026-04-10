# Matched 5v5 Pixel-Level Comparison

Comparison of pixel-level UQ metrics at matched sample count (K=5) vs the previous K=20 generative evaluation. All numbers from pixelwise conformal prediction at 90% target on 300 test samples (150 cal / 150 test). Ensemble uses 5 fixed members throughout.

**Why matched K=5.** Earlier evaluations of generative methods (FM, DDPM) used K=20 samples for mean/standard-deviation estimation while the ensemble baseline used its 5 fixed members. This gave generative methods a statistical advantage in the pixel-level coverage and interval-width metrics. The tables below report the corrected matched-sample comparison at K=5 across all methods; K=20 generative results are retained for reference. Functional CRPS (§11.3) was already reported at matched K=5 in earlier revisions and is unchanged.

Coverage values within ~2.5pp of the 90% target are within finite-sample noise (binomial SE on 150 test samples). Variation between 87.8% and 91.2% across regimes should not be over-interpreted.

## In-Distribution (test_in.npz)

### Exact BCs (64 pts, no noise)

| Method | K | Raw cov@90 | Conformal cov@90 | Conformal width |
|--------|---|-----------|-----------------|----------------|
| Ensemble | 5 | 81.5% | 90.7% | 0.0311 |
| Improved DDPM | 20 | 99.6% | 89.5% | 0.0029 |
| **Improved DDPM** | **5** | **94.7%** | **89.5%** | **0.0046** |
| Flow Matching | 20 | 96.5% | 90.5% | 0.0341 |
| **Flow Matching** | **5** | **87.8%** | **89.8%** | **0.0415** |

### Dense-Noisy (64 pts, sigma=0.1)

| Method | K | Raw cov@90 | Conformal cov@90 | Conformal width |
|--------|---|-----------|-----------------|----------------|
| Ensemble | 5 | 55.1% | 91.1% | 0.0737 |
| Improved DDPM | 20 | 87.8% | 89.9% | 0.0438 |
| **Improved DDPM** | **5** | **76.3%** | **87.8%** | **0.0533** |
| Flow Matching | 20 | 87.7% | 88.8% | 0.0701 |
| **Flow Matching** | **5** | **78.5%** | **90.8%** | **0.0897** |

### Sparse-Clean (16 pts, no noise)

| Method | K | Raw cov@90 | Conformal cov@90 | Conformal width |
|--------|---|-----------|-----------------|----------------|
| Ensemble | 5 | 80.9% | 90.9% | 0.0330 |
| Improved DDPM | 20 | 99.4% | 90.2% | 0.0034 |
| **Improved DDPM** | **5** | **94.2%** | **90.1%** | **0.0051** |
| Flow Matching | 20 | 98.1% | 91.8% | 0.0490 |
| **Flow Matching** | **5** | **89.3%** | **90.0%** | **0.0574** |

### Sparse-Noisy (16 pts, sigma=0.1) — Headline regime

| Method | K | Raw cov@90 | Conformal cov@90 | Conformal width |
|--------|---|-----------|-----------------|----------------|
| Ensemble | 5 | 31.4% | 90.2% | 0.1329 |
| Improved DDPM | 20 | 85.4% | 88.8% | 0.0883 |
| **Improved DDPM** | **5** | **79.0%** | **91.2%** | **0.1147** |
| Flow Matching | 20 | 89.4% | 89.9% | 0.1306 |
| **Flow Matching** | **5** | **78.8%** | **88.8%** | **0.1555** |

### Very-Sparse (8 pts, sigma=0.2)

| Method | K | Raw cov@90 | Conformal cov@90 | Conformal width |
|--------|---|-----------|-----------------|----------------|
| Ensemble | 5 | 14.6% | 89.8% | 0.3804 |
| Improved DDPM | 20 | 83.7% | 88.1% | 0.2476 |
| **Improved DDPM** | **5** | **77.5%** | **90.2%** | **0.3110** |
| Flow Matching | 20 | 83.8% | 87.9% | 0.3312 |
| **Flow Matching** | **5** | **76.8%** | **89.9%** | **0.4197** |

## Summary

At matched K=5, raw coverage for generative models drops by 5-12 percentage points compared to K=20 (e.g., DDPM sparse-noisy: 85.4% to 79.0%). After conformal calibration, all methods still achieve near-nominal 90% coverage. The key metric is **conformal interval width**, where DDPM retains the tightest intervals across all regimes at matched sample count:

| Regime | Ensemble width | DDPM width (K=5) | FM width (K=5) | Best |
|--------|---------------|-----------------|---------------|------|
| exact | 0.031 | **0.005** | 0.042 | DDPM |
| dense-noisy | 0.074 | **0.053** | 0.090 | DDPM |
| sparse-clean | 0.033 | **0.005** | 0.057 | DDPM |
| sparse-noisy | 0.133 | **0.115** | 0.156 | DDPM |
| very-sparse | 0.380 | **0.311** | 0.420 | DDPM |

DDPM produces the tightest calibrated intervals at every regime, even at matched sample count. The advantage over ensemble narrows from 1.5–10.7x (K=20) to 1.2–6.8x (K=5) — the headline regimes (sparse-noisy, very-sparse) were always modest and are now slightly more so; the exact and sparse-clean regimes retain a large multiplier because the true posterior is nearly degenerate there and DDPM correctly recognizes this.
