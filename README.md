# Uncertainty Quantification for PDE Surrogates: Flow Matching, Improved DDPM, and Conformal Prediction

## Overview

This project benchmarks three uncertainty quantification (UQ) approaches for neural PDE surrogates on the 2D Laplace equation under noisy, sparse boundary observations. We compare deep ensembles, conditional flow matching (OT-CFM), and an improved DDPM, each wrapped with split conformal prediction for distribution-free coverage guarantees. The benchmark spans three observation regimes (exact, noisy, sparse+noisy) to stress-test calibration under increasing input degradation.

## Key Results

### Coverage at 90% Target (Pixelwise Conformal)

| Method | Exact | Noisy | Sparse+Noisy |
|--------|-------|-------|--------------|
| Ensemble + conformal | 91.3% | 90.4% | 90.6% |
| Flow Matching + conformal | 91.8% | 87.9% | 89.4% |
| DDPM + conformal | 90.2% | 88.1% | 89.0% |

Raw (pre-conformal) coverage ranges from 15--82% depending on method and regime. Conformal calibration brings all methods to near-nominal coverage.

### Training Efficiency

| Method | Epochs | Wall Time (T4) | Estimated Cost |
|--------|--------|-----------------|----------------|
| Flow Matching | 80 | ~5.3 hrs | ~$3.20 |
| Improved DDPM | 80 | ~4.7 hrs | ~$2.80 |
| Ensemble (5 members) | 5 x 50 | ~100 min | ~$2.50 |

### Interval Sharpness (Pixelwise, 90%, Exact Regime)

| Method | Mean Interval Width |
|--------|-------------------|
| DDPM | 0.003 (tightest) |
| Ensemble | 0.031 |
| Flow Matching | 0.034 |

DDPM produces the tightest intervals while maintaining valid coverage, indicating the best-calibrated raw uncertainty among the three methods.

## Methods

**Deep Ensemble.** Five independently trained U-Net regressors with random initialization and data shuffling. Predictive uncertainty is computed as the pixelwise variance across ensemble members. Simple, embarrassingly parallel, and strong baseline.

**Conditional Flow Matching (OT-CFM).** A generative model that learns a velocity field transporting Gaussian noise to solution fields, conditioned on boundary observations. Uses mini-batch optimal transport (Hungarian) coupling for straighter learned flows. Multiple samples from the learned ODE yield an empirical posterior.

**Improved DDPM.** A denoising diffusion model with three key improvements: cosine noise schedule with zero-terminal SNR, v-prediction parameterization, and Min-SNR-gamma loss weighting. These compound to give 3--5x faster convergence over standard DDPM.

**Conformal Prediction.** A post-hoc calibration wrapper applied to any base model. Uses held-out calibration data to compute a nonconformity quantile that scales prediction intervals to achieve finite-sample coverage guarantees. Both spatial (simultaneous) and pixelwise (marginal) variants are implemented.

## Project Structure

```
src/diffphys/
  model/         - U-Net, DDPM, Flow Matching, Ensemble, FNO
  evaluation/    - UQ metrics, conformal prediction
  data/          - Dataset, observation regimes
  pde/           - Laplace solver, boundary condition generation
configs/         - YAML configs for all models
scripts/         - Training, evaluation, plotting
modal_deploy/    - Remote GPU training and evaluation on Modal
experiments/     - Checkpoints and results
figures/         - Generated plots
tests/           - Unit tests
docs/            - Technical documentation
```

## Reproducing Results

### Training

```bash
# Train flow matching and DDPM on Modal (remote GPU)
modal run modal_deploy/train_remote.py --config configs/flow_matching.yaml --config2 configs/ddpm_improved.yaml

# Train ensemble
modal run modal_deploy/train_remote.py --config configs/ensemble.yaml

# Or train locally
python scripts/train.py --config configs/flow_matching.yaml
```

### Evaluation

```bash
# Full Phase 2 evaluation (all models, all regimes)
modal run modal_deploy/evaluate_remote.py --eval-type phase2-all

# Conformal prediction calibration and evaluation
modal run modal_deploy/evaluate_remote.py --eval-type conformal
```

### Plotting

```bash
# Generate all figures
python scripts/plot_figures.py --all
```

## Figures

| Figure | Description |
|--------|-------------|
| `fig5_comparison.png` | Pixelwise prediction comparison across models and observation regimes |
| `fig9_conformal_calibration.png` | Conformal calibration curves showing raw vs. calibrated coverage |
| `fig9b_conformal_per_regime.png` | Per-regime conformal coverage breakdown (exact, noisy, sparse+noisy) |
| `fig10_convergence.png` | Training convergence curves: improved DDPM vs. standard DDPM |
| `fig_interval_widths.png` | Prediction interval widths across methods and conformal variants |

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS*.
2. Lipman, Y. et al. (2023). Flow Matching for Generative Modeling. *ICLR*.
3. Tong, A. et al. (2024). Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport. *TMLR*.
4. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS*.
5. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
6. Nichol, A. & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. *ICML*.
