"""Remote evaluation on Modal T4 GPU.

Usage:
    modal run modal_deploy/evaluate_remote.py --eval-type phase1
    modal run modal_deploy/evaluate_remote.py --eval-type ensemble-uq
    modal run modal_deploy/evaluate_remote.py --eval-type ddpm-uq
    modal run modal_deploy/evaluate_remote.py --eval-type fm-uq
    modal run modal_deploy/evaluate_remote.py --eval-type phase2-all
    modal run modal_deploy/evaluate_remote.py --eval-type functional-crps
    modal run modal_deploy/evaluate_remote.py --eval-type ood
    modal run modal_deploy/evaluate_remote.py --eval-type conformal
"""

import modal
import time

app = modal.App("diffphys-eval")

volume = modal.Volume.from_name("diffphys-data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("modal_deploy/requirements.txt")
    .add_local_dir("src", "/root/src", copy=True)
    .add_local_dir("configs", "/root/configs", copy=True)
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .run_commands("cd /root && pip install -e .")
)


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 4,
    volumes={"/data": volume},
)
def evaluate_phase1():
    """Evaluate U-Net and FNO on test splits."""
    import json
    from diffphys.evaluation.evaluate import run_evaluation

    results = {}
    for model_name, config_path in [
        ("unet_regressor", "/root/configs/unet_regressor.yaml"),
        ("fno", "/root/configs/fno.yaml"),
    ]:
        checkpoint = f"/data/experiments/{model_name}/best.pt"
        test_splits = {
            "test_in": "/data/test_in.npz",
            "test_ood": "/data/test_ood.npz",
        }
        print(f"\n=== Evaluating {model_name} ===")
        t0 = time.time()
        res = run_evaluation(config_path, checkpoint, test_splits, device="cuda")
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        for split, metrics in res.items():
            print(f"\n  {split}:")
            for name, stats in metrics.items():
                print(f"    {name:20s}: {stats['mean']:.6f} +/- {stats['std']:.6f}")

        results[model_name] = {"metrics": res, "eval_time_seconds": elapsed}

    out_path = "/data/experiments/phase1_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    volume.commit()
    return results


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 4,
    volumes={"/data": volume},
)
def evaluate_ensemble_uq():
    """Evaluate ensemble under all observation regimes."""
    import json
    import time
    from diffphys.evaluation.evaluate_uq import run_phase2_evaluation

    checkpoint_paths = [f"/data/experiments/ensemble_phase2/member_{i}/best.pt" for i in range(5)]

    print("=== Evaluating Ensemble UQ ===")
    t0 = time.time()
    results = run_phase2_evaluation(
        "ensemble", "/root/configs/ensemble_phase2.yaml",
        checkpoint_paths, "/data/test_in.npz", device="cuda",
    )
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")

    for regime, metrics in results.items():
        print(f"\n  {regime}:")
        for name, val in metrics.items():
            print(f"    {name:20s}: {val:.6f}")

    out = {"metrics": results, "eval_time_seconds": elapsed}
    out_path = "/data/experiments/ensemble_phase2/uq_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")
    volume.commit()
    return out


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 24,
    volumes={"/data": volume},
)
def evaluate_ddpm_uq(max_samples: int = 300):
    """Evaluate improved DDPM under all observation regimes, with per-regime checkpointing."""
    import json
    import time
    import os
    import torch
    from diffphys.data.dataset import LaplacePDEDataset
    from diffphys.data.observation import REGIMES
    from diffphys.model.trainer import build_model, load_config, _build_ddpm
    from diffphys.evaluation.evaluate_uq import evaluate_ddpm_uq as _eval_ddpm

    out_dir = "/data/experiments/ddpm_improved"
    os.makedirs(out_dir, exist_ok=True)
    partial_path = os.path.join(out_dir, "uq_partial.json")

    # Load existing partial results for warm start
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            state = json.load(f)
        results = state.get("metrics", {})
        total_time = state.get("eval_time_seconds", 0.0)
        print(f"Resuming from {len(results)} completed regimes: {list(results.keys())}")
    else:
        results = {}
        total_time = 0.0

    config = load_config("/root/configs/ddpm_improved.yaml")
    model = build_model(config["model"]).to("cuda")
    ddpm = _build_ddpm(model, config["ddpm"]).to("cuda")
    ckpt = torch.load("/data/experiments/ddpm_improved/best.pt", map_location="cuda")
    ddpm.load_state_dict(ckpt["model_state_dict"])
    ddpm.eval()

    for regime in REGIMES:
        if regime in results:
            print(f"\n  Skipping {regime} (already done)")
            continue

        print(f"\n=== Evaluating Improved DDPM UQ: {regime} ===")
        ds = LaplacePDEDataset("/data/test_in.npz", regime=regime)
        if max_samples and max_samples < len(ds):
            ds = torch.utils.data.Subset(ds, range(max_samples))
        print(f"  Using {len(ds)} samples, 20 samples each")
        loader = torch.utils.data.DataLoader(ds, batch_size=32)

        t0 = time.time()
        regime_results = _eval_ddpm(ddpm, loader, "cuda", n_samples=20)
        elapsed = time.time() - t0
        total_time += elapsed
        print(f"  Completed {regime} in {elapsed:.1f}s")

        for name, val in regime_results.items():
            print(f"    {name:20s}: {val:.6f}")

        results[regime] = regime_results

        # Save after each regime
        state = {"metrics": results, "eval_time_seconds": total_time}
        with open(partial_path, "w") as f:
            json.dump(state, f, indent=2)
        volume.commit()
        print(f"  Checkpoint saved ({len(results)}/{len(REGIMES)} regimes)")

    # Write final results
    out_path = os.path.join(out_dir, "uq_results.json")
    with open(out_path, "w") as f:
        json.dump({"metrics": results, "eval_time_seconds": total_time}, f, indent=2)
    volume.commit()
    print(f"\nAll done! Saved to {out_path}")
    return {"metrics": results, "eval_time_seconds": total_time}


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 24,
    volumes={"/data": volume},
)
def evaluate_fm_uq(max_samples: int = 300):
    """Evaluate flow matching under all observation regimes, with per-regime checkpointing."""
    import json
    import time
    import os
    import torch
    from diffphys.data.dataset import LaplacePDEDataset
    from diffphys.data.observation import REGIMES
    from diffphys.model.trainer import build_model, load_config
    from diffphys.model.flow_matching import ConditionalFlowMatcher
    from diffphys.evaluation.evaluate_uq import evaluate_cfm_uq

    out_dir = "/data/experiments/flow_matching"
    os.makedirs(out_dir, exist_ok=True)
    partial_path = os.path.join(out_dir, "uq_partial.json")

    if os.path.exists(partial_path):
        with open(partial_path) as f:
            state = json.load(f)
        results = state.get("metrics", {})
        total_time = state.get("eval_time_seconds", 0.0)
        print(f"Resuming from {len(results)} completed regimes: {list(results.keys())}")
    else:
        results = {}
        total_time = 0.0

    config = load_config("/root/configs/flow_matching.yaml")
    model = build_model(config["model"]).to("cuda")
    fm_cfg = config["flow_matching"]
    cfm = ConditionalFlowMatcher(
        model,
        use_ot=fm_cfg.get("use_ot", True),
        n_sample_steps=fm_cfg.get("n_sample_steps", 50),
    ).to("cuda")
    ckpt = torch.load("/data/experiments/flow_matching/best.pt", map_location="cuda")
    cfm.load_state_dict(ckpt["model_state_dict"])
    cfm.eval()

    for regime in REGIMES:
        if regime in results:
            print(f"\n  Skipping {regime} (already done)")
            continue

        print(f"\n=== Evaluating Flow Matching UQ: {regime} ===")
        ds = LaplacePDEDataset("/data/test_in.npz", regime=regime)
        if max_samples and max_samples < len(ds):
            ds = torch.utils.data.Subset(ds, range(max_samples))
        print(f"  Using {len(ds)} samples, 20 samples each")
        loader = torch.utils.data.DataLoader(ds, batch_size=32)

        t0 = time.time()
        regime_results = evaluate_cfm_uq(cfm, loader, "cuda", n_samples=20)
        elapsed = time.time() - t0
        total_time += elapsed
        print(f"  Completed {regime} in {elapsed:.1f}s")

        for name, val in regime_results.items():
            print(f"    {name:20s}: {val:.6f}")

        results[regime] = regime_results

        state = {"metrics": results, "eval_time_seconds": total_time}
        with open(partial_path, "w") as f:
            json.dump(state, f, indent=2)
        volume.commit()
        print(f"  Checkpoint saved ({len(results)}/{len(REGIMES)} regimes)")

    out_path = os.path.join(out_dir, "uq_results.json")
    with open(out_path, "w") as f:
        json.dump({"metrics": results, "eval_time_seconds": total_time}, f, indent=2)
    volume.commit()
    print(f"\nAll done! Saved to {out_path}")
    return {"metrics": results, "eval_time_seconds": total_time}


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

    if max_samples < 4:
        raise ValueError(f"max_samples must be >= 4 for meaningful cal/test split, got {max_samples}")
    n_cal = max_samples // 2
    n_test = max_samples - n_cal

    # --- Model definitions ---
    model_configs = []

    # 1. Ensemble (fast - just forward passes)
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

    # 2. Flow Matching (moderate - 50 steps x 20 samples)
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
        cfm.eval()
        model_configs.append(("flow_matching", cfm, "generative"))

    # 3. Improved DDPM (slow - 200 steps x 20 samples)
    if "ddpm_improved" not in results:
        config_ddpm = load_config("/root/configs/ddpm_improved.yaml")
        ddpm_model = build_model(config_ddpm["model"]).to("cuda")
        ddpm = _build_ddpm(ddpm_model, config_ddpm["ddpm"]).to("cuda")
        ckpt = torch.load(
            "/data/experiments/ddpm_improved/best.pt", map_location="cuda"
        )
        ddpm.load_state_dict(ckpt["model_state_dict"])
        ddpm.eval()
        model_configs.append(("ddpm_improved", ddpm, "generative"))

    for model_name, model, model_type in model_configs:
        print(f"\n{'='*60}")
        print(f"=== Conformal evaluation: {model_name} ===")
        print(f"{'='*60}")
        t0 = time.time()

        # Resume partial regime results for this model (DDPM can be slow)
        partial_key = f"_partial_{model_name}"
        model_results = results.pop(partial_key, {})
        if model_results:
            print(f"  Resuming {model_name} from {len(model_results)}/{len(REGIMES)} regimes")

        for regime in REGIMES:
            if regime in model_results:
                print(f"\n  --- {regime} (already done, skipping) ---")
                continue

            print(f"\n  --- {regime} ---")
            ds = LaplacePDEDataset("/data/test_in.npz", regime=regime)
            actual_cal = min(n_cal, len(ds) // 2)
            actual_test = min(n_test, len(ds) - actual_cal)

            cal_ds = torch.utils.data.Subset(ds, range(actual_cal))
            test_ds = torch.utils.data.Subset(ds, range(actual_cal, actual_cal + actual_test))
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
                print(f"  Collecting {model_name} predictions (cal, {n_cal} samples)...")
                cal_mean, cal_std, cal_truth = collect_generative_predictions(
                    model, cal_loader, "cuda", n_samples=20
                )
                print(f"  Collecting {model_name} predictions (test, {n_test} samples)...")
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

            # Per-regime checkpoint for slow models (DDPM)
            if model_name == "ddpm_improved":
                results[partial_key] = model_results
                state = {"results": results, "eval_time_seconds": total_time}
                with open(partial_path, "w") as f:
                    json.dump(state, f, indent=2)
                volume.commit()
                print(f"  Regime checkpoint saved ({len(model_results)}/{len(REGIMES)})")

        elapsed = time.time() - t0
        total_time += elapsed
        print(f"\n  {model_name} completed in {elapsed:.1f}s")

        # Remove partial key, store final results for this model
        results.pop(partial_key, None)
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


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    volumes={"/data": volume},
)
def generate_fig5_data(test_indices=(0, 100, 200)):
    """Generate predictions from all models for Figure 5 visual comparison.

    Saves .npz with ground truth, regressor, ensemble mean/std, FM samples, DDPM samples.
    """
    import numpy as np
    import torch
    from diffphys.data.dataset import LaplacePDEDataset
    from diffphys.model.trainer import build_model, load_config, _build_ddpm
    from diffphys.model.ensemble import EnsemblePredictor
    from diffphys.model.flow_matching import ConditionalFlowMatcher

    ds = LaplacePDEDataset("/data/test_in.npz", regime="sparse-noisy")
    results = {}

    for idx in test_indices:
        cond, target = ds[idx]
        cond_gpu = cond.unsqueeze(0).to("cuda")
        gt = target[0].cpu().numpy()  # (64, 64)

        # Regressor
        cfg_reg = load_config("/root/configs/unet_regressor.yaml")
        reg = build_model(cfg_reg["model"]).to("cuda")
        ckpt = torch.load("/data/experiments/unet_regressor/best.pt", map_location="cuda")
        reg.load_state_dict(ckpt["model_state_dict"])
        reg.eval()
        with torch.no_grad():
            reg_pred = reg(cond_gpu[:, :8])[0, 0].cpu().numpy()

        # Ensemble
        cfg_ens = load_config("/root/configs/ensemble_phase2.yaml")
        models = []
        for i in range(5):
            m = build_model(cfg_ens["model"]).to("cuda")
            ck = torch.load(f"/data/experiments/ensemble_phase2/member_{i}/best.pt", map_location="cuda")
            m.load_state_dict(ck["model_state_dict"])
            models.append(m)
        ens = EnsemblePredictor(models)
        ens_mean, ens_var = ens.predict(cond_gpu[:, :8])
        ens_mean_np = ens_mean[0, 0].cpu().numpy()
        ens_std_np = ens_var.sqrt()[0, 0].cpu().numpy()

        # Flow Matching (5 samples)
        cfg_fm = load_config("/root/configs/flow_matching.yaml")
        fm_model = build_model(cfg_fm["model"]).to("cuda")
        fm_cfg = cfg_fm["flow_matching"]
        cfm = ConditionalFlowMatcher(
            fm_model, use_ot=fm_cfg.get("use_ot", True),
            n_sample_steps=fm_cfg.get("n_sample_steps", 50),
        ).to("cuda")
        ckpt = torch.load("/data/experiments/flow_matching/best.pt", map_location="cuda")
        cfm.load_state_dict(ckpt["model_state_dict"])
        cfm.eval()
        fm_samples = cfm.sample(cond_gpu, n_samples=5)[:, 0, 0].cpu().numpy()  # (5, 64, 64)

        # DDPM (5 samples)
        cfg_ddpm = load_config("/root/configs/ddpm_improved.yaml")
        ddpm_model = build_model(cfg_ddpm["model"]).to("cuda")
        ddpm = _build_ddpm(ddpm_model, cfg_ddpm["ddpm"]).to("cuda")
        ckpt = torch.load("/data/experiments/ddpm_improved/best.pt", map_location="cuda")
        ddpm.load_state_dict(ckpt["model_state_dict"])
        ddpm.eval()
        ddpm_samples = ddpm.sample(cond_gpu, n_samples=5)[:, 0, 0].cpu().numpy()  # (5, 64, 64)

        results[f"idx_{idx}"] = {
            "ground_truth": gt,
            "regressor": reg_pred,
            "ensemble_mean": ens_mean_np,
            "ensemble_std": ens_std_np,
            "fm_samples": fm_samples,
            "ddpm_samples": ddpm_samples,
            "conditioning": cond.numpy(),
        }
        print(f"  Done idx={idx}")

    # Save as npz
    flat = {}
    for key, d in results.items():
        for name, arr in d.items():
            flat[f"{key}_{name}"] = arr

    out_path = "/data/experiments/fig5_data.npz"
    np.savez(out_path, **flat)
    volume.commit()
    print(f"Saved to {out_path}")


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 24,
    volumes={"/data": volume},
)
def evaluate_ood(max_samples: int = 300):
    """Evaluate all models on held-out piecewise BC family (test_ood.npz).

    Runs deterministic models (regressor, FNO) + generative models (ensemble, FM, DDPM)
    with per-model checkpointing.
    """
    import json
    import time
    import os
    import torch
    from diffphys.data.dataset import LaplacePDEDataset
    from diffphys.model.trainer import build_model, load_config, _build_ddpm
    from diffphys.model.ensemble import EnsemblePredictor
    from diffphys.model.flow_matching import ConditionalFlowMatcher
    from diffphys.evaluation.evaluate import evaluate_regressor, summarize_results
    from diffphys.evaluation.evaluate_uq import (
        evaluate_ensemble_uq,
        evaluate_ddpm_uq,
        evaluate_cfm_uq,
    )

    out_dir = "/data/experiments/ood"
    os.makedirs(out_dir, exist_ok=True)
    partial_path = os.path.join(out_dir, "ood_partial.json")

    run_params = {"max_samples": max_samples, "dataset": "test_ood.npz", "regime": "exact", "eval_version": 2}
    results, total_time = _load_partial_with_validation(partial_path, run_params)
    if results:
        print(f"Resuming: {list(results.keys())}")

    ds = LaplacePDEDataset("/data/test_ood.npz", regime="exact")
    if max_samples and max_samples < len(ds):
        ds = torch.utils.data.Subset(ds, range(max_samples))
    print(f"Using {len(ds)} OOD samples")
    loader = torch.utils.data.DataLoader(ds, batch_size=32)

    # 1. U-Net regressor (deterministic)
    if "unet_regressor" not in results:
        print("\n=== OOD: U-Net Regressor ===")
        t0 = time.time()
        cfg = load_config("/root/configs/unet_regressor.yaml")
        model = build_model(cfg["model"]).to("cuda")
        ckpt = torch.load("/data/experiments/unet_regressor/best.pt", map_location="cuda")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        raw = evaluate_regressor(model, loader, "cuda")
        results["unet_regressor"] = summarize_results(raw)
        elapsed = time.time() - t0
        total_time += elapsed
        print(f"  Done in {elapsed:.1f}s")
        _save_partial(partial_path, results, total_time, run_params)
        volume.commit()

    # 2. FNO (deterministic)
    if "fno" not in results:
        print("\n=== OOD: FNO ===")
        t0 = time.time()
        cfg = load_config("/root/configs/fno.yaml")
        model = build_model(cfg["model"]).to("cuda")
        ckpt = torch.load("/data/experiments/fno/best.pt", map_location="cuda")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        raw = evaluate_regressor(model, loader, "cuda")
        results["fno"] = summarize_results(raw)
        elapsed = time.time() - t0
        total_time += elapsed
        print(f"  Done in {elapsed:.1f}s")
        _save_partial(partial_path, results, total_time, run_params)
        volume.commit()

    # 3. Ensemble
    if "ensemble" not in results:
        print("\n=== OOD: Ensemble ===")
        t0 = time.time()
        cfg = load_config("/root/configs/ensemble_phase2.yaml")
        models = []
        for i in range(5):
            m = build_model(cfg["model"]).to("cuda")
            ck = torch.load(f"/data/experiments/ensemble_phase2/member_{i}/best.pt", map_location="cuda")
            m.load_state_dict(ck["model_state_dict"])
            models.append(m)
        ens = EnsemblePredictor(models)
        results["ensemble"] = evaluate_ensemble_uq(ens, loader, "cuda")
        elapsed = time.time() - t0
        total_time += elapsed
        print(f"  Done in {elapsed:.1f}s")
        _save_partial(partial_path, results, total_time, run_params)
        volume.commit()

    # 4. Flow Matching
    if "flow_matching" not in results:
        print("\n=== OOD: Flow Matching ===")
        t0 = time.time()
        cfg = load_config("/root/configs/flow_matching.yaml")
        fm_model = build_model(cfg["model"]).to("cuda")
        fm_cfg = cfg["flow_matching"]
        cfm = ConditionalFlowMatcher(
            fm_model, use_ot=fm_cfg.get("use_ot", True),
            n_sample_steps=fm_cfg.get("n_sample_steps", 50),
        ).to("cuda")
        ckpt = torch.load("/data/experiments/flow_matching/best.pt", map_location="cuda")
        cfm.load_state_dict(ckpt["model_state_dict"])
        cfm.eval()
        results["flow_matching"] = evaluate_cfm_uq(cfm, loader, "cuda", n_samples=20)
        elapsed = time.time() - t0
        total_time += elapsed
        print(f"  Done in {elapsed:.1f}s")
        _save_partial(partial_path, results, total_time, run_params)
        volume.commit()

    # 5. Improved DDPM
    if "ddpm_improved" not in results:
        print("\n=== OOD: Improved DDPM ===")
        t0 = time.time()
        cfg = load_config("/root/configs/ddpm_improved.yaml")
        ddpm_model = build_model(cfg["model"]).to("cuda")
        ddpm = _build_ddpm(ddpm_model, cfg["ddpm"]).to("cuda")
        ckpt = torch.load("/data/experiments/ddpm_improved/best.pt", map_location="cuda")
        ddpm.load_state_dict(ckpt["model_state_dict"])
        ddpm.eval()
        results["ddpm_improved"] = evaluate_ddpm_uq(ddpm, loader, "cuda", n_samples=20)
        elapsed = time.time() - t0
        total_time += elapsed
        print(f"  Done in {elapsed:.1f}s")
        _save_partial(partial_path, results, total_time, run_params)
        volume.commit()

    # Write final
    out_path = os.path.join(out_dir, "ood_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "eval_time_seconds": total_time, "run_params": run_params}, f, indent=2)
    volume.commit()
    print(f"\nAll done! Saved to {out_path}")
    return {"results": results, "eval_time_seconds": total_time}


def _save_partial(path, results, total_time, run_params):
    import json
    with open(path, "w") as f:
        json.dump({"results": results, "eval_time_seconds": total_time, "run_params": run_params}, f, indent=2)


def _load_partial_with_validation(partial_path, run_params):
    """Load partial checkpoint and validate run_params match.

    Returns (results, total_time) if valid, raises ValueError on mismatch.
    Missing run_params in the checkpoint (legacy files) are treated as incompatible.
    """
    import json
    import os
    if not os.path.exists(partial_path):
        return {}, 0.0
    with open(partial_path) as f:
        state = json.load(f)
    saved_params = state.get("run_params")
    if saved_params != run_params:
        raise ValueError(
            f"Partial checkpoint has run_params={saved_params} but current run "
            f"uses {run_params}. Use --fresh to start over."
        )
    return state.get("results", {}), state.get("eval_time_seconds", 0.0)


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 24,
    volumes={"/data": volume},
)
def evaluate_functional_crps(max_samples: int = 300):
    """Compute functional-level CRPS on 5 derived quantities for all models.

    For each model, generates 5 sample fields per test input (sparse-noisy regime),
    computes derived quantities (center T, mean T, max T, energy, flux),
    and evaluates CRPS. Per-model checkpointing.
    """
    import json
    import time
    import os
    import numpy as np
    import torch
    from diffphys.data.dataset import LaplacePDEDataset
    from diffphys.model.trainer import build_model, load_config, _build_ddpm
    from diffphys.model.ensemble import EnsemblePredictor
    from diffphys.model.flow_matching import ConditionalFlowMatcher
    from diffphys.evaluation.functionals import compute_functional_crps

    out_dir = "/data/experiments/functional_crps"
    os.makedirs(out_dir, exist_ok=True)
    partial_path = os.path.join(out_dir, "functional_partial.json")

    run_params = {"max_samples": max_samples, "dataset": "test_in.npz", "regime": "sparse-noisy", "n_gen_samples": 5}
    results, total_time = _load_partial_with_validation(partial_path, run_params)
    if results:
        print(f"Resuming: {list(results.keys())}")

    # Load sparse-noisy test data
    ds = LaplacePDEDataset("/data/test_in.npz", regime="sparse-noisy")
    if max_samples and max_samples < len(ds):
        ds = torch.utils.data.Subset(ds, range(max_samples))
    print(f"Using {len(ds)} samples, sparse-noisy regime")

    n_gen_samples = 5  # matched sample count

    def _collect_and_crps(model_name, sample_fn):
        """Collect 5 sample fields per input and compute functional CRPS."""
        all_crps = []
        loader = torch.utils.data.DataLoader(ds, batch_size=1)
        total = len(loader)
        for idx, (cond, target) in enumerate(loader):
            cond = cond.to("cuda")
            truth = target[0, 0].numpy()  # (64, 64)

            sample_fields = sample_fn(cond)  # (K, 64, 64) numpy
            crps_dict = compute_functional_crps(sample_fields, truth)
            all_crps.append(crps_dict)

            if (idx + 1) % 50 == 0 or idx == total - 1:
                print(f"    {model_name}: {idx + 1}/{total}")

        # Average CRPS across all test samples
        avg = {}
        for key in all_crps[0]:
            vals = [d[key] for d in all_crps]
            avg[f"mean_{key}"] = float(np.mean(vals))
            avg[f"std_{key}"] = float(np.std(vals))
        return avg

    # 1. Ensemble
    if "ensemble" not in results:
        print("\n=== Functional CRPS: Ensemble ===")
        t0 = time.time()
        cfg = load_config("/root/configs/ensemble_phase2.yaml")
        models = []
        for i in range(5):
            m = build_model(cfg["model"]).to("cuda")
            ck = torch.load(f"/data/experiments/ensemble_phase2/member_{i}/best.pt", map_location="cuda")
            m.load_state_dict(ck["model_state_dict"])
            models.append(m)
        ens = EnsemblePredictor(models)

        def ens_sample(cond):
            preds = ens.predict_all(cond[:, :8])  # (5, 1, 1, 64, 64)
            return preds[:, 0, 0].cpu().numpy()

        results["ensemble"] = _collect_and_crps("ensemble", ens_sample)
        total_time += time.time() - t0
        _save_partial(partial_path, results, total_time, run_params)
        volume.commit()
        print(f"  Checkpoint saved (ensemble done)")

    # 2. Flow Matching
    if "flow_matching" not in results:
        print("\n=== Functional CRPS: Flow Matching ===")
        t0 = time.time()
        cfg = load_config("/root/configs/flow_matching.yaml")
        fm_model = build_model(cfg["model"]).to("cuda")
        fm_cfg = cfg["flow_matching"]
        cfm = ConditionalFlowMatcher(
            fm_model, use_ot=fm_cfg.get("use_ot", True),
            n_sample_steps=fm_cfg.get("n_sample_steps", 50),
        ).to("cuda")
        ckpt = torch.load("/data/experiments/flow_matching/best.pt", map_location="cuda")
        cfm.load_state_dict(ckpt["model_state_dict"])
        cfm.eval()

        def fm_sample(cond):
            samples = cfm.sample(cond, n_samples=n_gen_samples)  # (5, 1, 1, 64, 64)
            return samples[:, 0, 0].cpu().numpy()

        results["flow_matching"] = _collect_and_crps("flow_matching", fm_sample)
        total_time += time.time() - t0
        _save_partial(partial_path, results, total_time, run_params)
        volume.commit()
        print(f"  Checkpoint saved (flow_matching done)")

    # 3. Improved DDPM
    if "ddpm_improved" not in results:
        print("\n=== Functional CRPS: Improved DDPM ===")
        t0 = time.time()
        cfg = load_config("/root/configs/ddpm_improved.yaml")
        ddpm_model = build_model(cfg["model"]).to("cuda")
        ddpm = _build_ddpm(ddpm_model, cfg["ddpm"]).to("cuda")
        ckpt = torch.load("/data/experiments/ddpm_improved/best.pt", map_location="cuda")
        ddpm.load_state_dict(ckpt["model_state_dict"])
        ddpm.eval()

        def ddpm_sample(cond):
            samples = ddpm.sample(cond, n_samples=n_gen_samples)  # (5, 1, 1, 64, 64)
            return samples[:, 0, 0].cpu().numpy()

        results["ddpm_improved"] = _collect_and_crps("ddpm_improved", ddpm_sample)
        total_time += time.time() - t0
        _save_partial(partial_path, results, total_time, run_params)
        volume.commit()
        print(f"  Checkpoint saved (ddpm_improved done)")

    # Final
    out_path = os.path.join(out_dir, "functional_crps_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "eval_time_seconds": total_time, "run_params": run_params}, f, indent=2)
    volume.commit()
    print(f"\nAll done! Saved to {out_path}")
    return {"results": results, "eval_time_seconds": total_time}


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 4,
    volumes={"/data": volume},
)
def evaluate_ddpm_phase1(max_samples: int = 300, n_samples: int = 5):
    """Evaluate DDPM on Phase 1 exact BCs — accuracy and physics metrics from sample mean.

    Generates n_samples fields per test input, computes mean prediction,
    then evaluates accuracy/physics metrics (rel_l2, pde_residual, bc_error, energy).
    """
    import json
    import time
    import numpy as np
    import torch
    from diffphys.data.dataset import LaplacePDEDataset
    from diffphys.model.trainer import build_model, load_config, _build_ddpm
    from diffphys.evaluation.evaluate import evaluate_regressor, summarize_results

    config = load_config("/root/configs/ddpm_improved.yaml")
    model = build_model(config["model"]).to("cuda")
    ddpm = _build_ddpm(model, config["ddpm"]).to("cuda")
    ckpt = torch.load("/data/experiments/ddpm_improved/best.pt", map_location="cuda")
    ddpm.load_state_dict(ckpt["model_state_dict"])
    ddpm.eval()

    results = {}
    for split_name, split_path in [("test_in", "/data/test_in.npz"), ("test_ood", "/data/test_ood.npz")]:
        print(f"\n=== DDPM Phase 1: {split_name} ===")
        ds = LaplacePDEDataset(split_path, regime="exact")
        if max_samples and max_samples < len(ds):
            ds = torch.utils.data.Subset(ds, range(max_samples))
        print(f"  {len(ds)} samples, {n_samples} DDPM samples each")

        # Generate sample means and evaluate as if they were regressor outputs
        all_preds = []
        all_targets = []
        loader = torch.utils.data.DataLoader(ds, batch_size=1)
        t0 = time.time()
        for idx, (cond, target) in enumerate(loader):
            cond = cond.to("cuda")
            samples = ddpm.sample(cond, n_samples=n_samples)  # (K, 1, 1, 64, 64)
            mean_pred = samples.mean(dim=0)  # (1, 1, 64, 64)
            all_preds.append(mean_pred.cpu())
            all_targets.append(target)
            if (idx + 1) % 50 == 0:
                print(f"    {idx + 1}/{len(loader)}")

        elapsed = time.time() - t0
        preds = torch.cat(all_preds)   # (N, 1, 64, 64)
        targets = torch.cat(all_targets)  # (N, 1, 64, 64)
        print(f"  Sampling done in {elapsed:.1f}s")

        # Use evaluate_regressor by creating a dummy model that returns precomputed predictions
        raw = {}
        diff = preds - targets
        raw["rel_l2"] = (diff.pow(2).sum(dim=(1, 2, 3)).sqrt() / targets.pow(2).sum(dim=(1, 2, 3)).sqrt()).numpy()

        # PDE residual (Laplacian of prediction)
        pred_np = preds[:, 0].numpy()
        h = 1.0 / 63
        residuals = []
        for p in pred_np:
            lap = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * p[1:-1, 1:-1]) / (h * h)
            residuals.append(np.sqrt(np.mean(lap ** 2)))
        raw["pde_residual"] = np.array(residuals)

        # BC error
        target_np = targets[:, 0].numpy()
        bc_errs = []
        for p, t in zip(pred_np, target_np):
            bc_err = np.mean([
                np.abs(p[0, :] - t[0, :]).mean(),
                np.abs(p[-1, :] - t[-1, :]).mean(),
                np.abs(p[:, 0] - t[:, 0]).mean(),
                np.abs(p[:, -1] - t[:, -1]).mean(),
            ])
            bc_errs.append(bc_err)
        raw["bc_err"] = np.array(bc_errs)

        # Max principle violations
        violations = []
        for p, t in zip(pred_np, target_np):
            bc_min = min(t[0, :].min(), t[-1, :].min(), t[:, 0].min(), t[:, -1].min())
            bc_max = max(t[0, :].max(), t[-1, :].max(), t[:, 0].max(), t[:, -1].max())
            viol = (p[1:-1, 1:-1] > bc_max + 1e-6).any() or (p[1:-1, 1:-1] < bc_min - 1e-6).any()
            violations.append(float(viol))
        raw["max_viol"] = np.array(violations)

        # Energy
        energies_pred = []
        energies_true = []
        for p, t in zip(pred_np, target_np):
            dx = np.diff(p, axis=1) / h
            dy = np.diff(p, axis=0) / h
            energies_pred.append(0.5 * h * h * (np.sum(dx**2) + np.sum(dy**2)))
            dx_t = np.diff(t, axis=1) / h
            dy_t = np.diff(t, axis=0) / h
            energies_true.append(0.5 * h * h * (np.sum(dx_t**2) + np.sum(dy_t**2)))
        raw["energy_pred"] = np.array(energies_pred)
        raw["energy_true"] = np.array(energies_true)
        raw["rel_energy_err"] = np.abs(raw["energy_pred"] - raw["energy_true"]) / (np.abs(raw["energy_true"]) + 1e-8)

        summary = {k: {"mean": float(v.mean()), "std": float(v.std())} for k, v in raw.items()}
        results[split_name] = summary
        print(f"  rel_l2: {summary['rel_l2']['mean']:.6f}, pde_res: {summary['pde_residual']['mean']:.2f}")

    out = {"metrics": results, "n_samples": n_samples, "max_samples": max_samples}
    out_path = "/data/experiments/ddpm_phase1_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    volume.commit()
    print(f"\nSaved to {out_path}")
    return out


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 24,
    volumes={"/data": volume},
)
def evaluate_ood_all_regimes(max_samples: int = 150):
    """Evaluate generative models on OOD piecewise BCs across all 5 observation regimes.

    Tests whether the OOD generalization gap widens or narrows under observation uncertainty.
    Per-regime checkpointing.
    """
    import json
    import time
    import os
    import torch
    from diffphys.data.dataset import LaplacePDEDataset
    from diffphys.data.observation import REGIMES
    from diffphys.model.trainer import build_model, load_config, _build_ddpm
    from diffphys.model.ensemble import EnsemblePredictor
    from diffphys.model.flow_matching import ConditionalFlowMatcher
    from diffphys.evaluation.evaluate_uq import (
        evaluate_ensemble_uq,
        evaluate_ddpm_uq,
        evaluate_cfm_uq,
    )

    out_dir = "/data/experiments/ood_regimes"
    os.makedirs(out_dir, exist_ok=True)
    partial_path = os.path.join(out_dir, "ood_regimes_partial.json")

    n_uq_samples = 20
    run_params = {
        "max_samples": max_samples,
        "dataset": "test_ood.npz",
        "n_uq_samples": n_uq_samples,
        "configs": ["ensemble_phase2", "flow_matching", "ddpm_improved"],
        "eval_version": 2,
    }
    results, total_time = _load_partial_with_validation(partial_path, run_params)
    if results:
        print(f"Resuming: {list(results.keys())}")

    # Load models once
    print("Loading models...")
    cfg_ens = load_config("/root/configs/ensemble_phase2.yaml")
    models = []
    for i in range(5):
        m = build_model(cfg_ens["model"]).to("cuda")
        ck = torch.load(f"/data/experiments/ensemble_phase2/member_{i}/best.pt", map_location="cuda")
        m.load_state_dict(ck["model_state_dict"])
        models.append(m)
    ens = EnsemblePredictor(models)

    cfg_fm = load_config("/root/configs/flow_matching.yaml")
    fm_model = build_model(cfg_fm["model"]).to("cuda")
    fm_cfg = cfg_fm["flow_matching"]
    cfm = ConditionalFlowMatcher(
        fm_model, use_ot=fm_cfg.get("use_ot", True),
        n_sample_steps=fm_cfg.get("n_sample_steps", 50),
    ).to("cuda")
    ckpt = torch.load("/data/experiments/flow_matching/best.pt", map_location="cuda")
    cfm.load_state_dict(ckpt["model_state_dict"])
    cfm.eval()

    cfg_ddpm = load_config("/root/configs/ddpm_improved.yaml")
    ddpm_model = build_model(cfg_ddpm["model"]).to("cuda")
    ddpm = _build_ddpm(ddpm_model, cfg_ddpm["ddpm"]).to("cuda")
    ckpt = torch.load("/data/experiments/ddpm_improved/best.pt", map_location="cuda")
    ddpm.load_state_dict(ckpt["model_state_dict"])
    ddpm.eval()
    print("Models loaded.")

    for regime in REGIMES:
        if regime in results:
            print(f"\n  Skipping {regime} (already done)")
            continue

        print(f"\n{'='*50}")
        print(f"=== OOD regime: {regime} ===")
        print(f"{'='*50}")

        ds = LaplacePDEDataset("/data/test_ood.npz", regime=regime)
        if max_samples and max_samples < len(ds):
            ds = torch.utils.data.Subset(ds, range(max_samples))
        loader = torch.utils.data.DataLoader(ds, batch_size=32)

        regime_results = {}
        t0 = time.time()

        # Ensemble
        print(f"  Ensemble...")
        regime_results["ensemble"] = evaluate_ensemble_uq(ens, loader, "cuda")

        # Flow Matching
        print(f"  Flow Matching...")
        regime_results["flow_matching"] = evaluate_cfm_uq(cfm, loader, "cuda", n_samples=n_uq_samples)

        # DDPM
        print(f"  DDPM...")
        regime_results["ddpm_improved"] = evaluate_ddpm_uq(ddpm, loader, "cuda", n_samples=n_uq_samples)

        elapsed = time.time() - t0
        total_time += elapsed
        print(f"  Regime {regime} done in {elapsed:.1f}s")

        results[regime] = regime_results
        _save_partial(partial_path, results, total_time, run_params)
        volume.commit()
        print(f"  Checkpoint saved ({len(results)}/{len(REGIMES)} regimes)")

    out_path = os.path.join(out_dir, "ood_regimes_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "eval_time_seconds": total_time, "run_params": run_params}, f, indent=2)
    volume.commit()
    print(f"\nAll done! Saved to {out_path}")
    return {"results": results, "eval_time_seconds": total_time}


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    volumes={"/data": volume},
)
def diagnose_ddpm():
    """Quick conditioning collapse check on improved DDPM."""
    import numpy as np
    import torch
    from diffphys.model.trainer import build_model, load_config, _build_ddpm
    from diffphys.data.dataset import LaplacePDEDataset

    config = load_config("/root/configs/ddpm_improved.yaml")
    model = build_model(config["model"]).to("cuda")
    ddpm = _build_ddpm(model, config["ddpm"]).to("cuda")
    ckpt = torch.load("/data/experiments/ddpm_improved/best.pt", map_location="cuda")
    ddpm.load_state_dict(ckpt["model_state_dict"])

    ds = LaplacePDEDataset("/data/test_in.npz", regime="exact")
    indices = [0, len(ds)//4, len(ds)//2, 3*len(ds)//4, len(ds)-1]

    print("\n=== Conditioning Collapse Diagnostic ===\n")

    all_sample_means = []
    within_stds = []

    for i, idx in enumerate(indices):
        cond, target = ds[idx]
        cond = cond.unsqueeze(0).to("cuda")
        samples = ddpm.sample(cond, n_samples=3)  # (3, 1, 1, 64, 64)
        samples_np = samples[:, 0, 0].cpu().numpy()  # (3, 64, 64)
        gt_np = target[0].cpu().numpy()  # (64, 64)

        sample_mean = samples_np.mean(axis=0)
        all_sample_means.append(sample_mean)
        within_stds.append(samples_np.std(axis=0).mean())

        mse = ((gt_np - sample_mean) ** 2).mean()
        print(f"  Input {i} (idx={idx}): MSE(sample_mean, truth) = {mse:.6f}, within-BC std = {samples_np.std(axis=0).mean():.6f}")

    cross_bc_std = np.stack(all_sample_means).std(axis=0).mean()
    avg_within_std = np.mean(within_stds)

    print(f"\nCross-BC std (conditioning signal): {cross_bc_std:.6f}")
    print(f"Avg within-BC std (sample diversity): {avg_within_std:.6f}")

    if cross_bc_std < 0.01:
        print("\nWARNING: Conditioning collapse likely!")
    elif cross_bc_std > 0.1:
        print("\nPASS: Conditioning is working.")
    else:
        print(f"\nMARGINAL: Cross-BC std = {cross_bc_std:.4f}")


@app.local_entrypoint()
def main(eval_type: str, max_samples: int = 300, fresh: bool = False):
    if eval_type == "phase1":
        evaluate_phase1.remote()
    elif eval_type == "ensemble-uq":
        evaluate_ensemble_uq.remote()
    elif eval_type == "ddpm-uq":
        if fresh:
            _clear_partial("experiments/ddpm_improved/uq_partial.json")
        evaluate_ddpm_uq.remote(max_samples=max_samples)
    elif eval_type == "fm-uq":
        if fresh:
            _clear_partial("experiments/flow_matching/uq_partial.json")
        evaluate_fm_uq.remote(max_samples=max_samples)
    elif eval_type == "phase2-all":
        if fresh:
            _clear_partial("experiments/ddpm_improved/uq_partial.json")
            _clear_partial("experiments/flow_matching/uq_partial.json")
        h1 = evaluate_ensemble_uq.spawn()
        h2 = evaluate_ddpm_uq.spawn(max_samples=max_samples)
        h3 = evaluate_fm_uq.spawn(max_samples=max_samples)
        h1.get()
        h2.get()
        h3.get()
        print("All Phase 2 evaluations complete.")
    elif eval_type == "functional-crps":
        if fresh:
            _clear_partial("experiments/functional_crps/functional_partial.json")
        evaluate_functional_crps.remote(max_samples=max_samples)
    elif eval_type == "ood":
        if fresh:
            _clear_partial("experiments/ood/ood_partial.json")
        evaluate_ood.remote(max_samples=max_samples)
    elif eval_type == "conformal":
        if fresh:
            _clear_partial("experiments/conformal/conformal_partial.json")
        evaluate_conformal.remote(max_samples=max_samples)
    elif eval_type == "fig5":
        generate_fig5_data.remote()
    elif eval_type == "ddpm-phase1":
        evaluate_ddpm_phase1.remote(max_samples=max_samples)
    elif eval_type == "ood-regimes":
        if fresh:
            _clear_partial("experiments/ood_regimes/ood_regimes_partial.json")
        evaluate_ood_all_regimes.remote(max_samples=max_samples)
    elif eval_type == "diagnose":
        diagnose_ddpm.remote()
    else:
        raise ValueError(
            f"Unknown eval type: {eval_type}. "
            "Use: phase1, ensemble-uq, ddpm-uq, fm-uq, phase2-all, functional-crps, ood, ood-regimes, ddpm-phase1, conformal, fig5, diagnose"
        )


def _clear_partial(remote_path: str):
    """Remove stale partial results from volume for a fresh run."""
    import subprocess
    result = subprocess.run(
        ["modal", "volume", "rm", "diffphys-data", remote_path],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"Cleared {remote_path}")
    else:
        print(f"No partial to clear at {remote_path}")
