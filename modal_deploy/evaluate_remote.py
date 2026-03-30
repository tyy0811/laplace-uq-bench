"""Remote evaluation on Modal T4 GPU.

Usage:
    modal run modal_deploy/evaluate_remote.py --eval-type phase1
    modal run modal_deploy/evaluate_remote.py --eval-type ensemble-uq
    modal run modal_deploy/evaluate_remote.py --eval-type ddpm-uq
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
    timeout=3600 * 8,
    volumes={"/data": volume},
)
def evaluate_ddpm_uq():
    """Evaluate DDPM under all observation regimes, with per-regime checkpointing."""
    import json
    import time
    import os
    import torch
    from diffphys.data.dataset import LaplacePDEDataset
    from diffphys.data.observation import REGIMES
    from diffphys.model.trainer import build_model, load_config
    from diffphys.model.ddpm import DDPM
    from diffphys.evaluation.evaluate_uq import evaluate_ddpm_uq as _eval_ddpm, _compute_uq_summary

    out_dir = "/data/experiments/ddpm_phase2"
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

    config = load_config("/root/configs/ddpm_phase2.yaml")
    model = build_model(config["model"]).to("cuda")
    ddpm = DDPM(model, **config["ddpm"]).to("cuda")
    ckpt = torch.load("/data/experiments/ddpm_phase2/best.pt", map_location="cuda")
    ddpm.load_state_dict(ckpt["model_state_dict"])

    for regime in REGIMES:
        if regime in results:
            print(f"\n  Skipping {regime} (already done)")
            continue

        print(f"\n=== Evaluating DDPM UQ: {regime} ===")
        ds = LaplacePDEDataset("/data/test_in.npz", regime=regime)
        loader = torch.utils.data.DataLoader(ds, batch_size=32)

        t0 = time.time()
        regime_results = _eval_ddpm(ddpm, loader, "cuda", n_samples=5)
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
    timeout=600,
    volumes={"/data": volume},
)
def diagnose_ddpm():
    """Quick conditioning collapse check on DDPM."""
    import numpy as np
    import torch
    from diffphys.model.trainer import build_model, load_config
    from diffphys.model.ddpm import DDPM
    from diffphys.data.dataset import LaplacePDEDataset

    config = load_config("/root/configs/ddpm_phase2.yaml")
    model = build_model(config["model"]).to("cuda")
    ddpm = DDPM(model, **config["ddpm"]).to("cuda")
    ckpt = torch.load("/data/experiments/ddpm_phase2/best.pt", map_location="cuda")
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
def main(eval_type: str):
    if eval_type == "phase1":
        evaluate_phase1.remote()
    elif eval_type == "ensemble-uq":
        evaluate_ensemble_uq.remote()
    elif eval_type == "ddpm-uq":
        evaluate_ddpm_uq.remote()
    elif eval_type == "diagnose":
        diagnose_ddpm.remote()
    else:
        raise ValueError(f"Unknown eval type: {eval_type}. Use: phase1, ensemble-uq, ddpm-uq, diagnose")
