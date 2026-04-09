"""DPS pre-flight checks before full C3.3 evaluation.

1. Fine zeta_pde sweep at zeta_obs=100 fixed
2. Per-example rel_l2 distribution at best settings
3. obs_rmse / sigma_obs ratio analysis

Usage:
    modal run modal_deploy/dps_preflight.py
"""

import modal

app = modal.App("diffphys-dps-preflight")

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
    import torch
    from diffphys.model.trainer import build_model, load_config
    from diffphys.model.unconditional_ddpm import UnconditionalDDPM

    config = load_config("/root/configs/unconditional_ddpm.yaml")
    model = build_model(config["model"]).to(device)
    ddpm_cfg = config["ddpm"]
    ddpm = UnconditionalDDPM(
        model, T=ddpm_cfg["T"],
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
    import torch
    observed = []
    indices = []
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
        B = x.shape[0]
        vals = []
        for edge_idx, pos in indices:
            if edge_idx == 0:    vals.append(x[:, 0, 0, pos])
            elif edge_idx == 1:  vals.append(x[:, 0, -1, pos])
            elif edge_idx == 2:  vals.append(x[:, 0, pos, 0])
            elif edge_idx == 3:  vals.append(x[:, 0, pos, -1])
        return torch.stack(vals, dim=1)
    return y_obs, obs_operator


def _run_dps_on_examples(ddpm, data, indices, regime, zeta_obs, zeta_pde,
                          grad_clip=None, n_samples=5):
    """Run DPS on a list of example indices, return per-example metrics."""
    import torch
    from diffphys.data.observation import apply_observation_regime
    from diffphys.model.dps_sampler import DPSSampler
    from diffphys.evaluation.metrics import pde_residual_norm, relative_l2_error

    sampler = DPSSampler(ddpm, zeta_obs=zeta_obs, zeta_pde=zeta_pde,
                         grad_clip=grad_clip)
    per_example = []
    for i in indices:
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
        sigma_obs = 0.1  # sparse-noisy noise level

        torch.manual_seed(i)
        dps_samples = sampler.sample(y_obs, obs_op, n_samples=n_samples,
                                     H=64, W=64)
        dps_mean = dps_samples.mean(dim=0, keepdim=True)

        rel_l2 = relative_l2_error(dps_mean, field_true).item()
        pde_res = pde_residual_norm(dps_mean).item()
        y_pred = obs_op(dps_mean)
        obs_rmse = (y_obs - y_pred.squeeze(0)).pow(2).mean().sqrt().item()

        per_example.append({
            "index": i,
            "rel_l2": rel_l2,
            "pde_residual": pde_res,
            "obs_rmse": obs_rmse,
            "obs_rmse_over_sigma": obs_rmse / sigma_obs,
        })
    return per_example


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 2,
    volumes={"/data": volume},
)
def preflight():
    import json
    import os
    import numpy as np
    import torch

    ddpm = _load_unconditional_ddpm()
    data = np.load("/data/val.npz")
    results = {}

    # === Check 1: Fine zeta_pde sweep at zeta_obs=100 ===
    print("=" * 60)
    print("CHECK 1: Fine zeta_pde sweep (zeta_obs=100 fixed)")
    print("=" * 60)

    zeta_pde_fine = [0.0, 0.001, 0.005, 0.01, 0.05]
    check1 = {}
    for zeta_pde in zeta_pde_fine:
        print(f"\n  zeta_pde={zeta_pde}:")
        examples = _run_dps_on_examples(
            ddpm, data, range(20), "sparse-noisy",
            zeta_obs=100.0, zeta_pde=zeta_pde, grad_clip=None)
        mean_rel_l2 = np.mean([e["rel_l2"] for e in examples])
        mean_obs_rmse = np.mean([e["obs_rmse"] for e in examples])
        mean_pde_res = np.mean([e["pde_residual"] for e in examples])
        print(f"    rel_l2={mean_rel_l2:.4f}  obs_rmse={mean_obs_rmse:.4f}  "
              f"pde_res={mean_pde_res:.4f}")
        check1[str(zeta_pde)] = {
            "mean_rel_l2": mean_rel_l2,
            "mean_obs_rmse": mean_obs_rmse,
            "mean_pde_residual": mean_pde_res,
        }

    best_pde = min(check1, key=lambda k: check1[k]["mean_rel_l2"])
    print(f"\n  Best zeta_pde: {best_pde} (rel_l2={check1[best_pde]['mean_rel_l2']:.4f})")
    results["check1_zeta_pde_sweep"] = check1
    results["check1_best_zeta_pde"] = float(best_pde)

    # === Check 2: Per-example distribution at best settings ===
    print("\n" + "=" * 60)
    print("CHECK 2: Per-example rel_l2 distribution (zeta_obs=100, best zeta_pde)")
    print("=" * 60)

    best_zeta_pde = float(best_pde)
    examples = _run_dps_on_examples(
        ddpm, data, range(20), "sparse-noisy",
        zeta_obs=100.0, zeta_pde=best_zeta_pde, grad_clip=None)

    rel_l2s = [e["rel_l2"] for e in examples]
    obs_rmses = [e["obs_rmse"] for e in examples]
    ratios = [e["obs_rmse_over_sigma"] for e in examples]

    print(f"\n  Rel L2:  mean={np.mean(rel_l2s):.4f}  "
          f"median={np.median(rel_l2s):.4f}  "
          f"std={np.std(rel_l2s):.4f}  "
          f"min={np.min(rel_l2s):.4f}  max={np.max(rel_l2s):.4f}")
    print(f"  IQR: [{np.percentile(rel_l2s, 25):.4f}, {np.percentile(rel_l2s, 75):.4f}]")
    print(f"\n  Per-example rel_l2: {[f'{x:.4f}' for x in rel_l2s]}")

    results["check2_per_example"] = examples
    results["check2_summary"] = {
        "rel_l2_mean": float(np.mean(rel_l2s)),
        "rel_l2_median": float(np.median(rel_l2s)),
        "rel_l2_std": float(np.std(rel_l2s)),
        "rel_l2_iqr": [float(np.percentile(rel_l2s, 25)),
                        float(np.percentile(rel_l2s, 75))],
        "rel_l2_min": float(np.min(rel_l2s)),
        "rel_l2_max": float(np.max(rel_l2s)),
    }

    # === Check 3: obs_rmse / sigma_obs ratio ===
    print("\n" + "=" * 60)
    print("CHECK 3: Observation noise floor analysis")
    print("=" * 60)

    print(f"\n  sigma_obs = 0.1 (sparse-noisy regime)")
    print(f"  obs_rmse / sigma_obs per example: {[f'{x:.3f}' for x in ratios]}")
    print(f"  Mean ratio: {np.mean(ratios):.3f}")
    print(f"  Median ratio: {np.median(ratios):.3f}")
    print(f"  Range: [{np.min(ratios):.3f}, {np.max(ratios):.3f}]")

    if 0.9 <= np.mean(ratios) <= 1.1:
        interpretation = "DPS saturates the observation noise floor — reconstruction error is observation-noise-limited"
    elif np.mean(ratios) < 0.9:
        interpretation = "DPS is BELOW noise floor — prior is helping disambiguate beyond observations"
    else:
        interpretation = "DPS is ABOVE noise floor — guidance not strong enough or prior fighting measurements"
    print(f"\n  Interpretation: {interpretation}")

    results["check3_noise_floor"] = {
        "sigma_obs": 0.1,
        "obs_rmse_mean": float(np.mean(obs_rmses)),
        "obs_rmse_over_sigma_mean": float(np.mean(ratios)),
        "obs_rmse_over_sigma_median": float(np.median(ratios)),
        "obs_rmse_over_sigma_range": [float(np.min(ratios)),
                                       float(np.max(ratios))],
        "interpretation": interpretation,
    }

    # Save
    out_dir = "/data/experiments/dps"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "preflight_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"\nAll results saved to {out_dir}/preflight_results.json")
    return results


@app.local_entrypoint()
def main():
    preflight.remote()
