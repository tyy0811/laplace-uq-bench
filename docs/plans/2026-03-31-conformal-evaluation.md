# Conformal Prediction Evaluation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Evaluate conformal prediction on all three models (ensemble, FM, DDPM) across 5 observation regimes, proving that conformal wrapping closes the gap to 90% coverage.

**Architecture:** Add a single `evaluate_conformal` Modal function that splits 300 test samples into 150 calibration + 150 evaluation. For each model × regime, collect raw predictions (mean, std), calibrate both spatial and pixelwise conformal predictors at 50/90/95% targets, then measure actual coverage and interval width on the eval split. Per-model checkpointing for resilience.

**Tech Stack:** numpy (conformal calibration), torch (model inference), Modal T4 GPU (FM/DDPM sampling)

---

### Task 1: Add `_collect_predictions` helper to `evaluate_uq.py`

The conformal evaluator needs raw per-sample (mean, std, truth) arrays, not aggregated metrics. Add helper functions that return these.

**Files:**
- Modify: `src/diffphys/evaluation/evaluate_uq.py`
- Test: `tests/test_conformal.py`

**Step 1: Write the failing test**

Add to `tests/test_conformal.py`:

```python
class TestCollectPredictions:
    def test_collect_ensemble_predictions_shapes(self):
        """Helper should return (N, H, W) numpy arrays."""
        from diffphys.evaluation.evaluate_uq import collect_ensemble_predictions
        import torch

        # Fake ensemble: 2 models that return constant values
        class FakeModel(torch.nn.Module):
            def __init__(self, val):
                super().__init__()
                self.val = val
            def forward(self, x):
                return torch.full((x.shape[0], 1, 8, 8), self.val)

        from diffphys.model.ensemble import EnsemblePredictor
        ensemble = EnsemblePredictor([FakeModel(1.0), FakeModel(3.0)])

        # Fake dataset: 10 samples
        conds = torch.randn(10, 8, 8, 8)
        targets = torch.randn(10, 1, 8, 8)
        ds = torch.utils.data.TensorDataset(conds, targets)
        loader = torch.utils.data.DataLoader(ds, batch_size=4)

        mean, std, truth = collect_ensemble_predictions(ensemble, loader, "cpu")
        assert mean.shape == (10, 8, 8)
        assert std.shape == (10, 8, 8)
        assert truth.shape == (10, 8, 8)
        # Ensemble mean of [1.0, 3.0] = 2.0
        assert np.allclose(mean, 2.0, atol=1e-5)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_conformal.py::TestCollectPredictions::test_collect_ensemble_predictions_shapes -v`
Expected: FAIL with `ImportError: cannot import name 'collect_ensemble_predictions'`

**Step 3: Write implementation**

Add to `src/diffphys/evaluation/evaluate_uq.py`:

```python
def collect_ensemble_predictions(ensemble, loader, device):
    """Collect raw ensemble predictions as numpy arrays.

    Returns:
        (mean, std, truth) each shaped (N, H, W) as numpy arrays.
    """
    all_true, all_mean, all_std = [], [], []
    for cond, target in loader:
        cond, target = cond.to(device), target.to(device)
        mean, var = ensemble.predict(cond)
        std = var.sqrt().clamp(min=1e-8)
        all_true.append(target[:, 0].cpu().numpy())
        all_mean.append(mean[:, 0].cpu().numpy())
        all_std.append(std[:, 0].cpu().numpy())
    return np.concatenate(all_mean), np.concatenate(all_std), np.concatenate(all_true)


def collect_generative_predictions(model, loader, device, n_samples=20):
    """Collect raw generative model predictions as numpy arrays.

    Works for both DDPM and CFM (any model with .sample(cond, n_samples)).

    Returns:
        (mean, std, truth) each shaped (N, H, W) as numpy arrays.
    """
    all_true, all_mean, all_std = [], [], []
    total_batches = len(loader)
    for batch_idx, (cond, target) in enumerate(loader):
        cond, target = cond.to(device), target.to(device)
        samples = model.sample(cond, n_samples=n_samples)  # (K, B, 1, H, W)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0).clamp(min=1e-8)
        all_true.append(target[:, 0].cpu().numpy())
        all_mean.append(mean[:, 0].cpu().numpy())
        all_std.append(std[:, 0].cpu().numpy())
        if (batch_idx + 1) % 2 == 0 or batch_idx == total_batches - 1:
            print(f"    Batch {batch_idx + 1}/{total_batches}")
    return np.concatenate(all_mean), np.concatenate(all_std), np.concatenate(all_true)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_conformal.py::TestCollectPredictions -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/diffphys/evaluation/evaluate_uq.py tests/test_conformal.py
git commit -m "feat: add collect_predictions helpers for conformal evaluation"
```

---

### Task 2: Add `evaluate_conformal_for_model` function to `evaluate_uq.py`

Core logic: given a model's (mean, std, truth) arrays on cal and test splits, run conformal prediction at multiple coverage targets and return metrics.

**Files:**
- Modify: `src/diffphys/evaluation/evaluate_uq.py`
- Test: `tests/test_conformal.py`

**Step 1: Write the failing test**

Add to `tests/test_conformal.py`:

```python
class TestConformalEvaluation:
    def test_evaluate_conformal_for_model_keys(self):
        """Should return spatial and pixelwise results at each target."""
        from diffphys.evaluation.evaluate_uq import evaluate_conformal_for_model

        rng = np.random.default_rng(42)
        N = 100
        cal_mean = rng.standard_normal((N, 8, 8))
        cal_std = np.abs(rng.standard_normal((N, 8, 8))) + 0.1
        cal_truth = cal_mean + 0.5 * rng.standard_normal((N, 8, 8))
        test_mean = rng.standard_normal((N, 8, 8))
        test_std = np.abs(rng.standard_normal((N, 8, 8))) + 0.1
        test_truth = test_mean + 0.5 * rng.standard_normal((N, 8, 8))

        results = evaluate_conformal_for_model(
            cal_mean, cal_std, cal_truth,
            test_mean, test_std, test_truth,
            targets=[0.50, 0.90, 0.95],
        )
        # Check structure
        assert "raw_coverage_90" in results
        assert "spatial_90_coverage" in results
        assert "spatial_90_q_hat" in results
        assert "spatial_90_mean_width" in results
        assert "pixelwise_90_coverage" in results
        assert "pixelwise_90_q_hat" in results

    def test_conformal_improves_coverage(self):
        """Conformal should achieve >= target coverage on well-behaved data."""
        from diffphys.evaluation.evaluate_uq import evaluate_conformal_for_model

        rng = np.random.default_rng(123)
        N = 500
        cal_mean = rng.standard_normal((N, 8, 8))
        cal_std = np.ones((N, 8, 8))
        cal_truth = cal_mean + rng.standard_normal((N, 8, 8))
        test_mean = rng.standard_normal((N, 8, 8))
        test_std = np.ones((N, 8, 8))
        test_truth = test_mean + rng.standard_normal((N, 8, 8))

        results = evaluate_conformal_for_model(
            cal_mean, cal_std, cal_truth,
            test_mean, test_std, test_truth,
            targets=[0.90],
        )
        # Pixelwise conformal should hit ~90% on iid Gaussian data
        assert results["pixelwise_90_coverage"] >= 0.88
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_conformal.py::TestConformalEvaluation -v`
Expected: FAIL with `ImportError`

**Step 3: Write implementation**

Add to `src/diffphys/evaluation/evaluate_uq.py`:

```python
def evaluate_conformal_for_model(
    cal_mean, cal_std, cal_truth,
    test_mean, test_std, test_truth,
    targets=(0.50, 0.90, 0.95),
):
    """Run conformal prediction evaluation on precomputed predictions.

    Args:
        cal_mean, cal_std, cal_truth: (N_cal, H, W) calibration arrays.
        test_mean, test_std, test_truth: (N_test, H, W) test arrays.
        targets: Coverage target levels.

    Returns:
        Dict with raw and conformal metrics.
    """
    from .conformal import SpatialConformalPredictor, PixelwiseConformalPredictor

    results = {}

    for target in targets:
        pct = int(target * 100)
        alpha = 1.0 - target

        # Raw coverage (no conformal) on test set
        z = scipy_norm_ppf((1 + target) / 2)  # e.g. 1.645 for 90%
        raw_lower = test_mean - z * test_std
        raw_upper = test_mean + z * test_std
        raw_covered = (test_truth >= raw_lower) & (test_truth <= raw_upper)
        results[f"raw_coverage_{pct}"] = float(raw_covered.mean())

        # Spatial conformal
        cp_s = SpatialConformalPredictor(alpha=alpha)
        cp_s.calibrate(cal_mean, cal_std, cal_truth)
        s_lower, s_upper = cp_s.predict_intervals(test_mean, test_std)
        s_covered = (test_truth >= s_lower) & (test_truth <= s_upper)
        results[f"spatial_{pct}_coverage"] = float(s_covered.all(axis=(1, 2)).mean())
        results[f"spatial_{pct}_pixelwise_coverage"] = float(s_covered.mean())
        results[f"spatial_{pct}_q_hat"] = float(cp_s.q_hat)
        results[f"spatial_{pct}_mean_width"] = float((s_upper - s_lower).mean())

        # Pixelwise conformal
        cp_p = PixelwiseConformalPredictor(alpha=alpha)
        cp_p.calibrate(cal_mean, cal_std, cal_truth)
        p_lower, p_upper = cp_p.predict_intervals(test_mean, test_std)
        p_covered = (test_truth >= p_lower) & (test_truth <= p_upper)
        results[f"pixelwise_{pct}_coverage"] = float(p_covered.mean())
        results[f"pixelwise_{pct}_q_hat"] = float(cp_p.q_hat)
        results[f"pixelwise_{pct}_mean_width"] = float((p_upper - p_lower).mean())

    return results
```

Also add at the top of `evaluate_uq.py`, near the existing imports:

```python
from scipy.stats import norm as _scipy_norm

def scipy_norm_ppf(q):
    """Quantile function for standard normal (lazy import)."""
    return float(_scipy_norm.ppf(q))
```

Note: if scipy is not available in the environment, replace with a hardcoded lookup:
```python
_NORM_QUANTILES = {0.75: 0.6745, 0.95: 1.6449, 0.975: 1.9600}
def scipy_norm_ppf(q):
    return _NORM_QUANTILES.get(q, 1.6449)
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_conformal.py::TestConformalEvaluation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/diffphys/evaluation/evaluate_uq.py tests/test_conformal.py
git commit -m "feat: add evaluate_conformal_for_model with spatial + pixelwise"
```

---

### Task 3: Add `evaluate_conformal` function to `evaluate_remote.py`

The Modal function that orchestrates the full conformal evaluation across all models and regimes.

**Files:**
- Modify: `modal_deploy/evaluate_remote.py`

**Step 1: Add the function**

Insert before the `diagnose_ddpm` function in `modal_deploy/evaluate_remote.py`:

```python
@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 24,
    volumes={"/data": volume},
)
def evaluate_conformal(max_samples: int = 300):
    """Evaluate conformal prediction on all models across all regimes.

    Splits max_samples into cal (first half) and test (second half).
    For each model x regime: collect predictions, calibrate conformal, measure coverage.
    Per-model checkpointing: saves after completing each model.
    """
    import json
    import time
    import os
    import numpy as np
    import torch
    from diffphys.data.dataset import LaplacePDEDataset
    from diffphys.data.observation import REGIMES
    from diffphys.model.trainer import build_model, load_config, _build_ddpm
    from diffphys.model.ensemble import EnsemblePredictor
    from diffphys.model.flow_matching import ConditionalFlowMatcher
    from diffphys.evaluation.evaluate_uq import (
        collect_ensemble_predictions,
        collect_generative_predictions,
        evaluate_conformal_for_model,
    )

    out_dir = "/data/experiments/conformal"
    os.makedirs(out_dir, exist_ok=True)
    partial_path = os.path.join(out_dir, "conformal_partial.json")

    # Resume from partial if available
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            state = json.load(f)
        results = state.get("results", {})
        total_time = state.get("eval_time_seconds", 0.0)
        print(f"Resuming: completed models = {list(results.keys())}")
    else:
        results = {}
        total_time = 0.0

    n_cal = max_samples // 2
    n_test = max_samples - n_cal

    # --- Model definitions ---
    model_configs = []

    # 1. Ensemble (fast — just forward passes)
    if "ensemble" not in results:
        config_ens = load_config("/root/configs/ensemble_phase2.yaml")
        models = []
        for i in range(5):
            m = build_model(config_ens["model"]).to("cuda")
            ckpt = torch.load(
                f"/data/experiments/ensemble_phase2/member_{i}/best.pt",
                map_location="cuda",
            )
            m.load_state_dict(ckpt["model_state_dict"])
            models.append(m)
        ensemble = EnsemblePredictor(models)
        model_configs.append(("ensemble", ensemble, "ensemble"))

    # 2. Flow Matching (moderate — 50 steps x 20 samples)
    if "flow_matching" not in results:
        config_fm = load_config("/root/configs/flow_matching.yaml")
        fm_model = build_model(config_fm["model"]).to("cuda")
        fm_cfg = config_fm["flow_matching"]
        cfm = ConditionalFlowMatcher(
            fm_model,
            use_ot=fm_cfg.get("use_ot", True),
            n_sample_steps=fm_cfg.get("n_sample_steps", 50),
        ).to("cuda")
        ckpt = torch.load(
            "/data/experiments/flow_matching/best.pt", map_location="cuda"
        )
        cfm.load_state_dict(ckpt["model_state_dict"])
        model_configs.append(("flow_matching", cfm, "generative"))

    # 3. Improved DDPM (slow — 200 steps x 20 samples)
    if "ddpm_improved" not in results:
        config_ddpm = load_config("/root/configs/ddpm_improved.yaml")
        ddpm_model = build_model(config_ddpm["model"]).to("cuda")
        ddpm = _build_ddpm(ddpm_model, config_ddpm["ddpm"]).to("cuda")
        ckpt = torch.load(
            "/data/experiments/ddpm_improved/best.pt", map_location="cuda"
        )
        ddpm.load_state_dict(ckpt["model_state_dict"])
        model_configs.append(("ddpm_improved", ddpm, "generative"))

    for model_name, model, model_type in model_configs:
        print(f"\n{'='*60}")
        print(f"=== Conformal evaluation: {model_name} ===")
        print(f"{'='*60}")
        t0 = time.time()
        model_results = {}

        for regime in REGIMES:
            print(f"\n  --- {regime} ---")
            ds = LaplacePDEDataset("/data/test_in.npz", regime=regime)

            cal_ds = torch.utils.data.Subset(ds, range(n_cal))
            test_ds = torch.utils.data.Subset(ds, range(n_cal, n_cal + n_test))
            cal_loader = torch.utils.data.DataLoader(cal_ds, batch_size=32)
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32)

            # Collect predictions
            if model_type == "ensemble":
                print("  Collecting ensemble predictions (cal)...")
                cal_mean, cal_std, cal_truth = collect_ensemble_predictions(
                    model, cal_loader, "cuda"
                )
                print("  Collecting ensemble predictions (test)...")
                test_mean, test_std, test_truth = collect_ensemble_predictions(
                    model, test_loader, "cuda"
                )
            else:
                print(f"  Collecting {model_name} predictions (cal, 150 samples)...")
                cal_mean, cal_std, cal_truth = collect_generative_predictions(
                    model, cal_loader, "cuda", n_samples=20
                )
                print(f"  Collecting {model_name} predictions (test, 150 samples)...")
                test_mean, test_std, test_truth = collect_generative_predictions(
                    model, test_loader, "cuda", n_samples=20
                )

            # Run conformal evaluation
            regime_results = evaluate_conformal_for_model(
                cal_mean, cal_std, cal_truth,
                test_mean, test_std, test_truth,
                targets=[0.50, 0.90, 0.95],
            )

            for k, v in regime_results.items():
                print(f"    {k:35s}: {v:.6f}")

            model_results[regime] = regime_results

        elapsed = time.time() - t0
        total_time += elapsed
        print(f"\n  {model_name} completed in {elapsed:.1f}s")

        results[model_name] = model_results

        # Checkpoint after each model
        state = {"results": results, "eval_time_seconds": total_time}
        with open(partial_path, "w") as f:
            json.dump(state, f, indent=2)
        volume.commit()
        print(f"  Checkpoint saved ({len(results)}/3 models)")

    # Write final results
    out_path = os.path.join(out_dir, "conformal_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "eval_time_seconds": total_time}, f, indent=2)
    volume.commit()
    print(f"\nAll done! Saved to {out_path}")
    return {"results": results, "eval_time_seconds": total_time}
```

**Step 2: Wire into entrypoint**

In the `main()` function, add the `"conformal"` case:

```python
    elif eval_type == "conformal":
        if fresh:
            _clear_partial("experiments/conformal/conformal_partial.json")
        evaluate_conformal.remote(max_samples=max_samples)
```

Also update the docstring at the top of the file to include:
```
    modal run modal_deploy/evaluate_remote.py --eval-type conformal
```

And update the error message:
```python
"Use: phase1, ensemble-uq, ddpm-uq, fm-uq, phase2-all, conformal, diagnose"
```

**Step 3: Commit**

```bash
git add modal_deploy/evaluate_remote.py
git commit -m "feat: add conformal evaluation for all models across regimes"
```

---

### Task 4: Run conformal evaluation on Modal

**Step 1: Launch**

```bash
modal run --detach modal_deploy/evaluate_remote.py --eval-type conformal --max-samples 300
```

**Step 2: Verify startup**

Check logs show "=== Conformal evaluation: ensemble ===" and batch progress within ~30s.

**Step 3: Monitor completion**

- Ensemble: ~2 min (just forward passes)
- Flow Matching: ~30 min (20 × 50 steps × 150 cal + 150 test samples)
- DDPM: ~2 hrs (20 × 200 steps × 150 cal + 150 test samples)
- Total: ~2.5 hrs

**Step 4: Download results**

```bash
modal volume get diffphys-data experiments/conformal/conformal_results.json experiments/conformal/conformal_results.json
```

**Step 5: Commit results**

```bash
git add experiments/conformal/conformal_results.json
git commit -m "data: conformal prediction evaluation results across all models and regimes"
```

---

### Expected output structure

```json
{
  "results": {
    "ensemble": {
      "exact": {
        "raw_coverage_50": 0.47,
        "raw_coverage_90": 0.81,
        "raw_coverage_95": 0.86,
        "spatial_90_coverage": 0.92,
        "spatial_90_pixelwise_coverage": 0.99,
        "spatial_90_q_hat": 3.5,
        "spatial_90_mean_width": 0.045,
        "pixelwise_90_coverage": 0.91,
        "pixelwise_90_q_hat": 1.8,
        "pixelwise_90_mean_width": 0.023
      },
      "dense-noisy": { ... },
      ...
    },
    "flow_matching": { ... },
    "ddpm_improved": { ... }
  },
  "eval_time_seconds": 9000
}
```

### Key validation checks after results arrive

1. **Spatial conformal coverage at 90% target should be >= 90%** for all models × regimes (this is the mathematical guarantee)
2. **Pixelwise conformal coverage at 90% should be >= 90%** (marginally)
3. **Spatial intervals should be wider than pixelwise** (spatial is conservative)
4. **Ensemble conformal should have tightest intervals** (smallest sharpness → least inflation needed)
5. **q_hat values should increase with regime difficulty** (more noise → harder to cover)
