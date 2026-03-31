"""Remote evaluation on Modal T4 GPU.

Usage:
    modal run modal_deploy/evaluate_remote.py --eval-type phase1
    modal run modal_deploy/evaluate_remote.py --eval-type ensemble-uq
    modal run modal_deploy/evaluate_remote.py --eval-type ddpm-uq
    modal run modal_deploy/evaluate_remote.py --eval-type fm-uq
    modal run modal_deploy/evaluate_remote.py --eval-type phase2-all
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
    elif eval_type == "conformal":
        if fresh:
            _clear_partial("experiments/conformal/conformal_partial.json")
        evaluate_conformal.remote(max_samples=max_samples)
    elif eval_type == "diagnose":
        diagnose_ddpm.remote()
    else:
        raise ValueError(
            f"Unknown eval type: {eval_type}. "
            "Use: phase1, ensemble-uq, ddpm-uq, fm-uq, phase2-all, conformal, diagnose"
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
