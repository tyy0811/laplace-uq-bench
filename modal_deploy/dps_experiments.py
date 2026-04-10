"""DPS experiments on Modal: smoke test, guidance tuning, and full evaluation.

Usage:
    modal run modal_deploy/dps_experiments.py --eval-type smoke-test
    modal run modal_deploy/dps_experiments.py --eval-type tune-guidance
    modal run modal_deploy/dps_experiments.py --eval-type evaluate
"""

import modal

app = modal.App("diffphys-dps")

volume = modal.Volume.from_name("diffphys-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("modal_deploy/requirements.txt")
    .add_local_dir("src", "/root/src", copy=True)
    .add_local_dir("configs", "/root/configs", copy=True)
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .run_commands("cd /root && pip install -e .")
)


def _load_unconditional_ddpm(device="cuda"):
    """Load trained unconditional DDPM from volume."""
    import torch
    from diffphys.model.trainer import build_model, load_config
    from diffphys.model.unconditional_ddpm import UnconditionalDDPM

    config = load_config("/root/configs/unconditional_ddpm.yaml")
    model = build_model(config["model"]).to(device)
    ddpm_cfg = config["ddpm"]
    ddpm = UnconditionalDDPM(
        model,
        T=ddpm_cfg["T"],
        beta_start=ddpm_cfg.get("beta_start", 1e-4),
        beta_end=ddpm_cfg.get("beta_end", 0.02),
        schedule=ddpm_cfg.get("schedule", "cosine"),
        prediction=ddpm_cfg.get("prediction", "v"),
        min_snr_gamma=ddpm_cfg.get("min_snr_gamma", 5.0),
    ).to(device)

    ckpt = torch.load("/data/experiments/unconditional_ddpm/best.pt",
                       map_location=device)
    ddpm.load_state_dict(ckpt["model_state_dict"])
    ddpm.eval()
    print(f"Loaded unconditional DDPM (best val_loss={ckpt['val_loss']:.6f})")
    return ddpm


def _build_obs_operator(bc_top, bc_bottom, bc_left, bc_right,
                        mask_top, mask_bottom, mask_left, mask_right,
                        device="cuda"):
    """Build observation operator that extracts boundary values at observed locations.

    Returns (y_obs, obs_operator) where:
        y_obs: (M,) tensor of observed boundary values
        obs_operator: callable (B, 1, H, W) -> (B, M) predicted observations
    """
    import torch

    # Collect observed boundary values
    observed = []
    indices = []  # (edge, position) pairs

    for edge_idx, (bc, mask) in enumerate([
        (bc_top, mask_top), (bc_bottom, mask_bottom),
        (bc_left, mask_left), (bc_right, mask_right)
    ]):
        obs_mask = mask > 0.5
        observed.append(bc[obs_mask])
        positions = torch.where(obs_mask)[0]
        for pos in positions:
            indices.append((edge_idx, pos.item()))

    y_obs = torch.cat(observed).to(device)

    def obs_operator(x):
        """Extract boundary values at observed locations. x: (B, 1, H, W)."""
        B = x.shape[0]
        vals = []
        for edge_idx, pos in indices:
            if edge_idx == 0:    # top
                vals.append(x[:, 0, 0, pos])
            elif edge_idx == 1:  # bottom
                vals.append(x[:, 0, -1, pos])
            elif edge_idx == 2:  # left
                vals.append(x[:, 0, pos, 0])
            elif edge_idx == 3:  # right
                vals.append(x[:, 0, pos, -1])
        return torch.stack(vals, dim=1)  # (B, M)

    return y_obs, obs_operator


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    volumes={"/data": volume},
)
def smoke_test():
    """C2.3 + C3.1: Generate unconditional samples and run DPS on one test example."""
    import json
    import os
    import numpy as np
    import torch
    from diffphys.data.dataset import LaplacePDEDataset
    from diffphys.data.observation import apply_observation_regime
    from diffphys.data.conditioning import encode_conditioning
    from diffphys.model.dps_sampler import DPSSampler
    from diffphys.evaluation.metrics import pde_residual_norm, relative_l2_error

    ddpm = _load_unconditional_ddpm()

    # --- C2.3: Generate unconditional samples ---
    print("\n=== C2.3: Unconditional sample quality ===")
    torch.manual_seed(0)
    samples = ddpm.sample(8, H=64, W=64)
    print(f"  Sample shape: {samples.shape}")
    print(f"  Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
    print(f"  Sample mean: {samples.mean():.4f}")
    print(f"  Sample std: {samples.std():.4f}")

    # Check PDE residual — unconditional samples should look like Laplace solutions
    residuals = pde_residual_norm(samples)
    print(f"  PDE residual (mean): {residuals.mean():.4f}")
    print(f"  PDE residual (max): {residuals.max():.4f}")

    # --- C3.1: DPS smoke test on one sparse-noisy example ---
    print("\n=== C3.1: DPS smoke test (sparse-noisy, 1 example) ===")
    data = np.load("/data/test_in.npz")
    idx = 0
    field_true = torch.from_numpy(data["fields"][idx:idx+1]).unsqueeze(1).cuda()  # (1,1,64,64)

    bcs = [torch.from_numpy(data[k][idx]) for k in
           ["bc_top", "bc_bottom", "bc_left", "bc_right"]]
    obs_bcs, masks = [], []
    for bc in bcs:
        obs, mask = apply_observation_regime(bc, "sparse-noisy")
        obs_bcs.append(obs)
        masks.append(mask)

    y_obs, obs_op = _build_obs_operator(*obs_bcs, *masks)

    sampler = DPSSampler(ddpm, zeta_obs=1.0, zeta_pde=0.1)
    torch.manual_seed(42)
    dps_samples = sampler.sample(y_obs, obs_op, n_samples=5, H=64, W=64)

    print(f"  DPS sample shape: {dps_samples.shape}")
    print(f"  DPS finite: {torch.isfinite(dps_samples).all()}")

    # Compare to ground truth
    dps_mean = dps_samples.mean(dim=0, keepdim=True)  # (1,1,64,64)
    rel_l2 = relative_l2_error(dps_mean, field_true)
    pde_res = pde_residual_norm(dps_mean)
    print(f"  DPS mean rel_l2 vs truth: {rel_l2.item():.4f}")
    print(f"  DPS mean PDE residual: {pde_res.item():.4f}")

    # Also check per-sample residuals
    per_sample_res = pde_residual_norm(dps_samples)
    print(f"  Per-sample PDE residual: {per_sample_res.tolist()}")

    # Observation agreement
    y_pred = obs_op(dps_mean)
    obs_err = (y_obs - y_pred.squeeze(0)).pow(2).mean().sqrt()
    print(f"  Observation RMSE: {obs_err.item():.4f}")

    results = {
        "unconditional_samples": {
            "range": [samples.min().item(), samples.max().item()],
            "mean": samples.mean().item(),
            "std": samples.std().item(),
            "pde_residual_mean": residuals.mean().item(),
            "pde_residual_max": residuals.max().item(),
        },
        "dps_smoke_test": {
            "rel_l2": rel_l2.item(),
            "pde_residual_mean": pde_res.item(),
            "obs_rmse": obs_err.item(),
            "finite": torch.isfinite(dps_samples).all().item(),
            "per_sample_pde_residual": per_sample_res.tolist(),
        },
    }

    out_dir = "/data/experiments/dps"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "smoke_test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"\nResults saved to {out_dir}/smoke_test_results.json")
    return results


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 4,
    volumes={"/data": volume},
)
def tune_guidance(n_val: int = 20):
    """C3.2: Grid search over (zeta_obs, zeta_pde) on sparse-noisy validation set."""
    import json
    import os
    import time
    import numpy as np
    import torch
    from diffphys.data.dataset import LaplacePDEDataset
    from diffphys.data.observation import apply_observation_regime
    from diffphys.model.dps_sampler import DPSSampler
    from diffphys.evaluation.metrics import pde_residual_norm, relative_l2_error

    ddpm = _load_unconditional_ddpm()

    # Load validation data
    data = np.load("/data/val.npz")
    n_val = min(n_val, len(data["fields"]))

    zeta_obs_grid = [0.1, 0.5, 1.0, 5.0, 10.0]
    zeta_pde_grid = [0.0, 0.01, 0.1, 1.0]

    results = {}
    out_dir = "/data/experiments/dps"
    os.makedirs(out_dir, exist_ok=True)
    partial_path = os.path.join(out_dir, "tuning_partial.json")

    # Resume from partial if available
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} completed combos")

    for zeta_obs in zeta_obs_grid:
        for zeta_pde in zeta_pde_grid:
            key = f"obs={zeta_obs}_pde={zeta_pde}"
            if key in results:
                print(f"  Skipping {key} (already done)")
                continue

            print(f"\n=== zeta_obs={zeta_obs}, zeta_pde={zeta_pde} ===")
            sampler = DPSSampler(ddpm, zeta_obs=zeta_obs, zeta_pde=zeta_pde)

            all_rel_l2, all_obs_err, all_pde_res = [], [], []
            t0 = time.time()

            for i in range(n_val):
                field_true = torch.from_numpy(
                    data["fields"][i:i+1]).unsqueeze(1).cuda()
                bcs = [torch.from_numpy(data[k][i]) for k in
                       ["bc_top", "bc_bottom", "bc_left", "bc_right"]]
                obs_bcs, masks = [], []
                for bc in bcs:
                    obs, mask = apply_observation_regime(bc, "sparse-noisy")
                    obs_bcs.append(obs)
                    masks.append(mask)

                y_obs, obs_op = _build_obs_operator(*obs_bcs, *masks)

                torch.manual_seed(i)
                dps_samples = sampler.sample(y_obs, obs_op, n_samples=5,
                                             H=64, W=64)

                dps_mean = dps_samples.mean(dim=0, keepdim=True)
                all_rel_l2.append(relative_l2_error(dps_mean, field_true).item())
                all_pde_res.append(pde_residual_norm(dps_mean).item())

                y_pred = obs_op(dps_mean)
                obs_err = (y_obs - y_pred.squeeze(0)).pow(2).mean().sqrt()
                all_obs_err.append(obs_err.item())

            elapsed = time.time() - t0
            combo_results = {
                "mean_rel_l2": float(np.mean(all_rel_l2)),
                "mean_obs_rmse": float(np.mean(all_obs_err)),
                "mean_pde_residual": float(np.mean(all_pde_res)),
                "time_seconds": elapsed,
            }
            results[key] = combo_results
            print(f"  rel_l2={combo_results['mean_rel_l2']:.4f}  "
                  f"obs_rmse={combo_results['mean_obs_rmse']:.4f}  "
                  f"pde_res={combo_results['mean_pde_residual']:.4f}  "
                  f"({elapsed:.1f}s)")

            # Checkpoint
            with open(partial_path, "w") as f:
                json.dump(results, f, indent=2)
            volume.commit()

    # Find best combo
    best_key = min(results, key=lambda k: results[k]["mean_rel_l2"])
    print(f"\n=== Best combo (by rel_l2): {best_key} ===")
    print(f"  {results[best_key]}")

    out_path = os.path.join(out_dir, "tuning_results.json")
    with open(out_path, "w") as f:
        json.dump({"grid_results": results, "best": best_key,
                    "best_metrics": results[best_key]}, f, indent=2)
    volume.commit()
    print(f"\nSaved to {out_path}")
    return results


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 12,
    volumes={"/data": volume},
)
def evaluate_dps(n_samples_generative: int = 5, max_samples: int = 300):
    """C3.3: Full DPS evaluation across all regimes."""
    import json
    import os
    import time
    import numpy as np
    import torch
    from diffphys.data.dataset import LaplacePDEDataset
    from diffphys.data.observation import REGIMES, apply_observation_regime
    from diffphys.model.dps_sampler import DPSSampler
    from diffphys.evaluation.metrics import pde_residual_norm, relative_l2_error
    from diffphys.evaluation.uq_metrics import (
        pixelwise_coverage, crps_gaussian, calibration_error, sharpness,
    )

    ddpm = _load_unconditional_ddpm()

    # Tuned values from preflight (zeta_pde=0, no clipping)
    zeta_obs, zeta_pde, grad_clip = 100.0, 0.0, None
    print(f"Using tuned guidance: zeta_obs={zeta_obs}, zeta_pde={zeta_pde}, grad_clip={grad_clip}")

    sampler = DPSSampler(ddpm, zeta_obs=zeta_obs, zeta_pde=zeta_pde, grad_clip=grad_clip)

    out_dir = "/data/experiments/dps"
    os.makedirs(out_dir, exist_ok=True)
    partial_path = os.path.join(out_dir, "eval_partial.json")

    run_params = {
        "max_samples": max_samples,
        "n_samples_generative": n_samples_generative,
        "zeta_obs": zeta_obs,
        "zeta_pde": zeta_pde,
    }

    # Resume
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            state = json.load(f)
        if state.get("run_params") != run_params:
            print("Run params mismatch, starting fresh")
            results = {}
            total_time = 0.0
        else:
            results = state.get("results", {})
            total_time = state.get("eval_time_seconds", 0.0)
            print(f"Resuming from {len(results)} completed regimes")
    else:
        results = {}
        total_time = 0.0

    # Evaluate: 5 in-dist regimes + OOD exact + OOD all 5 regimes
    eval_tasks = []
    for regime in REGIMES:
        eval_tasks.append(("test_in", "/data/test_in.npz", regime, max_samples))
    eval_tasks.append(("test_ood", "/data/test_ood.npz", "exact", max_samples))
    for regime in REGIMES:
        eval_tasks.append(("test_ood", "/data/test_ood.npz", regime, min(150, max_samples)))

    per_example_all = {}  # saved separately for §10.5

    for dataset_name, dataset_path, regime, n_max in eval_tasks:
        key = f"{dataset_name}_{regime}"
        # OOD all-regimes keys need disambiguation from OOD exact
        if dataset_name == "test_ood" and key in results and regime != "exact":
            key = f"test_ood_regimes_{regime}"
        elif dataset_name == "test_ood" and regime != "exact":
            key = f"test_ood_regimes_{regime}"

        if key in results:
            print(f"\n  Skipping {key} (already done)")
            continue

        data = np.load(dataset_path)
        n = min(n_max, len(data["fields"]))

        print(f"\n=== DPS evaluation: {key} ({n} samples) ===")
        t0 = time.time()

        all_true, all_mean, all_std = [], [], []
        per_example = []

        for i in range(n):
            field_true = torch.from_numpy(
                data["fields"][i:i+1]).unsqueeze(1).cuda()
            bcs = [torch.from_numpy(data[k][i]) for k in
                   ["bc_top", "bc_bottom", "bc_left", "bc_right"]]
            obs_bcs, masks = [], []
            for bc in bcs:
                obs, mask = apply_observation_regime(bc, regime)
                obs_bcs.append(obs)
                masks.append(mask)

            y_obs, obs_op = _build_obs_operator(*obs_bcs, *masks)

            torch.manual_seed(i)
            dps_samples = sampler.sample(
                y_obs, obs_op, n_samples=n_samples_generative, H=64, W=64)

            mean = dps_samples.mean(dim=0)  # (1, 64, 64)
            std = dps_samples.std(dim=0)    # (1, 64, 64)

            all_true.append(field_true.squeeze(0).cpu())
            all_mean.append(mean.cpu())
            all_std.append(std.cpu())

            # Per-example metrics
            rel_l2_i = relative_l2_error(
                mean.unsqueeze(0), field_true).item()
            pde_res_i = pde_residual_norm(mean.unsqueeze(0)).item()
            y_pred = obs_op(mean.unsqueeze(0))
            obs_rmse_i = (y_obs - y_pred.squeeze(0)).pow(2).mean().sqrt().item()
            per_example.append({
                "index": i, "rel_l2": rel_l2_i,
                "pde_residual": pde_res_i, "obs_rmse": obs_rmse_i,
            })

            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{n}")

        true = torch.stack(all_true)
        mean = torch.stack(all_mean)
        std = torch.stack(all_std)

        rel_l2s = [e["rel_l2"] for e in per_example]
        obs_rmses = [e["obs_rmse"] for e in per_example]

        regime_results = {
            "coverage_50": pixelwise_coverage(true, mean, std, 0.50).item(),
            "coverage_90": pixelwise_coverage(true, mean, std, 0.90).item(),
            "coverage_95": pixelwise_coverage(true, mean, std, 0.95).item(),
            "crps": crps_gaussian(true, mean, std).mean().item(),
            "calibration_error": calibration_error(true, mean, std).item(),
            "sharpness": sharpness(std).item(),
            "rel_l2_mean": float(np.mean(rel_l2s)),
            "rel_l2_median": float(np.median(rel_l2s)),
            "rel_l2_std": float(np.std(rel_l2s)),
            "rel_l2_iqr": [float(np.percentile(rel_l2s, 25)),
                            float(np.percentile(rel_l2s, 75))],
            "obs_rmse_mean": float(np.mean(obs_rmses)),
            "pde_residual_mean": float(np.mean([e["pde_residual"] for e in per_example])),
        }

        elapsed = time.time() - t0
        total_time += elapsed
        print(f"  Completed {key} in {elapsed:.1f}s")
        for name, val in regime_results.items():
            if isinstance(val, list):
                print(f"    {name:20s}: {val}")
            else:
                print(f"    {name:20s}: {val:.6f}")

        results[key] = regime_results
        per_example_all[key] = per_example

        # Checkpoint
        with open(partial_path, "w") as f:
            json.dump({"results": results, "eval_time_seconds": total_time,
                       "run_params": run_params}, f, indent=2)
        volume.commit()

    out_path = os.path.join(out_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "eval_time_seconds": total_time,
                   "run_params": run_params}, f, indent=2)

    # Save per-example metrics separately (for §10.5 distribution analysis)
    per_example_path = os.path.join(out_dir, "eval_per_example.json")
    with open(per_example_path, "w") as f:
        json.dump(per_example_all, f, indent=2)

    volume.commit()
    print(f"\nAll done! Saved to {out_path} and {per_example_path}")
    return results


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 4,
    volumes={"/data": volume},
)
def tune_guidance_extended(n_val: int = 20):
    """Extended grid: higher zeta_obs + different grad_clip levels."""
    import json
    import os
    import time
    import numpy as np
    import torch
    from diffphys.data.observation import apply_observation_regime
    from diffphys.model.dps_sampler import DPSSampler
    from diffphys.evaluation.metrics import pde_residual_norm, relative_l2_error

    ddpm = _load_unconditional_ddpm()
    data = np.load("/data/val.npz")
    n_val = min(n_val, len(data["fields"]))

    configs = [
        # (zeta_obs, zeta_pde, grad_clip)
        (20.0, 0.0, 1.0),
        (50.0, 0.0, 1.0),
        (100.0, 0.0, 1.0),
        (10.0, 0.0, 5.0),
        (10.0, 0.0, 10.0),
        (10.0, 0.0, None),  # no clipping
        (20.0, 0.0, 5.0),
        (50.0, 0.0, 5.0),
        (50.0, 0.0, None),
        (100.0, 0.0, 5.0),
        (100.0, 0.0, None),
    ]

    results = {}
    out_dir = "/data/experiments/dps"
    os.makedirs(out_dir, exist_ok=True)
    partial_path = os.path.join(out_dir, "tuning_extended_partial.json")

    if os.path.exists(partial_path):
        with open(partial_path) as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} completed combos")

    for zeta_obs, zeta_pde, grad_clip in configs:
        clip_str = f"{grad_clip}" if grad_clip is not None else "none"
        key = f"obs={zeta_obs}_pde={zeta_pde}_clip={clip_str}"
        if key in results:
            print(f"  Skipping {key} (already done)")
            continue

        print(f"\n=== zeta_obs={zeta_obs}, zeta_pde={zeta_pde}, clip={clip_str} ===")
        sampler = DPSSampler(ddpm, zeta_obs=zeta_obs, zeta_pde=zeta_pde,
                             grad_clip=grad_clip)

        all_rel_l2, all_obs_err, all_pde_res = [], [], []
        t0 = time.time()
        any_nan = False

        for i in range(n_val):
            field_true = torch.from_numpy(
                data["fields"][i:i+1]).unsqueeze(1).cuda()
            bcs = [torch.from_numpy(data[k][i]) for k in
                   ["bc_top", "bc_bottom", "bc_left", "bc_right"]]
            obs_bcs, masks = [], []
            for bc in bcs:
                obs, mask = apply_observation_regime(bc, "sparse-noisy")
                obs_bcs.append(obs)
                masks.append(mask)

            y_obs, obs_op = _build_obs_operator(*obs_bcs, *masks)

            torch.manual_seed(i)
            dps_samples = sampler.sample(y_obs, obs_op, n_samples=5, H=64, W=64)

            if not torch.isfinite(dps_samples).all():
                print(f"  NaN at sample {i}, aborting this combo")
                any_nan = True
                break

            dps_mean = dps_samples.mean(dim=0, keepdim=True)
            all_rel_l2.append(relative_l2_error(dps_mean, field_true).item())
            all_pde_res.append(pde_residual_norm(dps_mean).item())
            y_pred = obs_op(dps_mean)
            all_obs_err.append(
                (y_obs - y_pred.squeeze(0)).pow(2).mean().sqrt().item())

        elapsed = time.time() - t0
        if any_nan:
            combo_results = {"mean_rel_l2": float("inf"), "nan": True,
                             "time_seconds": elapsed}
        else:
            combo_results = {
                "mean_rel_l2": float(np.mean(all_rel_l2)),
                "mean_obs_rmse": float(np.mean(all_obs_err)),
                "mean_pde_residual": float(np.mean(all_pde_res)),
                "time_seconds": elapsed,
            }
        results[key] = combo_results
        print(f"  rel_l2={combo_results['mean_rel_l2']:.4f}  ({elapsed:.1f}s)")

        with open(partial_path, "w") as f:
            json.dump(results, f, indent=2)
        volume.commit()

    # Find best
    best_key = min(results, key=lambda k: results[k]["mean_rel_l2"])
    print(f"\n=== Best combo: {best_key} ===")
    print(f"  {results[best_key]}")

    out_path = os.path.join(out_dir, "tuning_extended_results.json")
    with open(out_path, "w") as f:
        json.dump({"grid_results": results, "best": best_key,
                    "best_metrics": results[best_key]}, f, indent=2)
    volume.commit()
    print(f"\nSaved to {out_path}")
    return results


@app.local_entrypoint()
def main(eval_type: str, max_samples: int = 300, n_val: int = 20):
    if eval_type == "smoke-test":
        smoke_test.remote()
    elif eval_type == "tune-guidance":
        tune_guidance.remote(n_val=n_val)
    elif eval_type == "tune-extended":
        tune_guidance_extended.remote(n_val=n_val)
    elif eval_type == "evaluate":
        evaluate_dps.remote(max_samples=max_samples)
    else:
        raise ValueError(
            f"Unknown eval type: {eval_type}. "
            "Use: smoke-test, tune-guidance, tune-extended, evaluate"
        )
