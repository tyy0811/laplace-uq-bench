"""Day 6: Zero-shot adaptation experiment.

Evaluates conditional DDPM and DPS on observation patterns neither model
was trained on. The conditional model was trained on uniform sensor density
with sigma in [0, 0.2]. These patterns test distributional generalization.

Zero-shot patterns:
  1. extreme-noise: 16 pts/edge, sigma=0.5 (training max was 0.2)
  2. non-uniform: 16 top, 8 bottom, 32 left, 4 right
  3. single-edge: 64 pts on top only, other three edges unobserved

Usage:
    modal run modal_deploy/dps_zero_shot.py
"""

import modal

app = modal.App("diffphys-zero-shot")

volume = modal.Volume.from_name("diffphys-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("modal_deploy/requirements.txt")
    .add_local_dir("src", "/root/src", copy=True)
    .add_local_dir("configs", "/root/configs", copy=True)
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .run_commands("cd /root && pip install -e .")
)


def _apply_zero_shot_regime(bcs, pattern, rng=None):
    """Apply a zero-shot observation pattern to 4 BC edges.

    Args:
        bcs: list of 4 (nx,) tensors [top, bottom, left, right]
        pattern: one of "extreme-noise", "non-uniform", "single-edge"
        rng: optional torch.Generator

    Returns:
        obs_bcs: list of 4 (nx,) observed BC tensors
        masks: list of 4 (nx,) mask tensors
    """
    import torch
    from diffphys.data.observation import _linear_interp

    nx = bcs[0].shape[0]
    obs_bcs, masks = [], []

    if pattern == "extreme-noise":
        # Same as sparse-noisy but sigma=0.5 (5x training max)
        n_points = 16
        sigma = 0.5
        for bc in bcs:
            indices = torch.linspace(0, nx - 1, n_points).long()
            mask = torch.zeros(nx)
            mask[indices] = 1.0
            obs_values = bc[indices].clone()
            noise = torch.randn(n_points, generator=rng) * sigma
            obs_values = obs_values + noise
            x_obs = indices.float()
            x_all = torch.arange(nx, dtype=torch.float32)
            observed = _linear_interp(x_obs, obs_values, x_all)
            obs_bcs.append(observed)
            masks.append(mask)

    elif pattern == "non-uniform":
        # Different observation density per edge
        edge_points = [16, 8, 32, 4]  # top, bottom, left, right
        sigma = 0.1
        for bc, n_pts in zip(bcs, edge_points):
            indices = torch.linspace(0, nx - 1, n_pts).long()
            mask = torch.zeros(nx)
            mask[indices] = 1.0
            obs_values = bc[indices].clone()
            noise = torch.randn(n_pts, generator=rng) * sigma
            obs_values = obs_values + noise
            x_obs = indices.float()
            x_all = torch.arange(nx, dtype=torch.float32)
            observed = _linear_interp(x_obs, obs_values, x_all)
            obs_bcs.append(observed)
            masks.append(mask)

    elif pattern == "single-edge":
        # Only top edge observed (64 pts, no noise), rest completely unobserved
        for i, bc in enumerate(bcs):
            if i == 0:  # top
                mask = torch.ones(nx)
                obs_bcs.append(bc.clone())
                masks.append(mask)
            else:
                mask = torch.zeros(nx)
                # Unobserved: use zeros (no information)
                obs_bcs.append(torch.zeros(nx))
                masks.append(mask)

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return obs_bcs, masks


def _build_obs_operator(obs_bcs, masks, device="cuda"):
    """Build observation operator from observed BCs and masks."""
    import torch

    observed = []
    indices = []
    for edge_idx, (bc, mask) in enumerate(zip(obs_bcs, masks)):
        obs_mask = mask > 0.5
        observed.append(bc[obs_mask])
        positions = torch.where(obs_mask)[0]
        for pos in positions:
            indices.append((edge_idx, pos.item()))

    y_obs = torch.cat(observed).to(device)

    def obs_operator(x):
        vals = []
        for edge_idx, pos in indices:
            if edge_idx == 0:    vals.append(x[:, 0, 0, pos])
            elif edge_idx == 1:  vals.append(x[:, 0, -1, pos])
            elif edge_idx == 2:  vals.append(x[:, 0, pos, 0])
            elif edge_idx == 3:  vals.append(x[:, 0, pos, -1])
        return torch.stack(vals, dim=1)
    return y_obs, obs_operator


def _load_conditional_ddpm(device="cuda"):
    """Load the trained conditional improved DDPM."""
    import torch
    from diffphys.model.trainer import build_model, load_config, _build_ddpm

    config = load_config("/root/configs/ddpm_improved.yaml")
    model = build_model(config["model"]).to(device)
    ddpm = _build_ddpm(model, config["ddpm"]).to(device)
    ckpt = torch.load("/data/experiments/ddpm_improved/best.pt",
                       map_location=device)
    ddpm.load_state_dict(ckpt["model_state_dict"])
    ddpm.eval()
    print(f"Loaded conditional DDPM (val_loss={ckpt['val_loss']:.6f})")
    return ddpm


def _load_unconditional_ddpm(device="cuda"):
    """Load the trained unconditional DDPM."""
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
    print(f"Loaded unconditional DDPM (val_loss={ckpt['val_loss']:.6f})")
    return ddpm


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 6,
    volumes={"/data": volume},
)
def run_zero_shot(max_samples: int = 100):
    """Evaluate conditional DDPM and DPS on zero-shot observation patterns."""
    import json
    import os
    import time
    import numpy as np
    import torch
    from diffphys.data.conditioning import encode_conditioning
    from diffphys.model.dps_sampler import DPSSampler
    from diffphys.evaluation.metrics import pde_residual_norm, relative_l2_error

    cond_ddpm = _load_conditional_ddpm()
    uncond_ddpm = _load_unconditional_ddpm()
    sampler = DPSSampler(uncond_ddpm, zeta_obs=100.0, zeta_pde=0.0, grad_clip=None)

    data = np.load("/data/test_in.npz")
    n = min(max_samples, len(data["fields"]))

    patterns = ["extreme-noise", "non-uniform", "single-edge"]
    results = {}

    out_dir = "/data/experiments/dps_zero_shot"
    os.makedirs(out_dir, exist_ok=True)
    partial_path = os.path.join(out_dir, "partial.json")

    # Resume
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            state = json.load(f)
        if state.get("max_samples") == max_samples:
            results = state.get("results", {})
            print(f"Resuming from {len(results)} completed patterns")
        else:
            results = {}

    for pattern in patterns:
        if pattern in results:
            print(f"\n  Skipping {pattern} (already done)")
            continue

        print(f"\n{'='*60}")
        print(f"=== Zero-shot pattern: {pattern} ({n} examples) ===")
        print(f"{'='*60}")
        t0 = time.time()

        cond_per_example = []
        dps_per_example = []

        for i in range(n):
            field_true = torch.from_numpy(
                data["fields"][i:i+1]).unsqueeze(1).cuda()  # (1,1,64,64)
            bcs = [torch.from_numpy(data[k][i]) for k in
                   ["bc_top", "bc_bottom", "bc_left", "bc_right"]]

            rng = torch.Generator().manual_seed(i)
            obs_bcs, masks_list = _apply_zero_shot_regime(bcs, pattern, rng=rng)

            # --- Conditional DDPM ---
            cond = encode_conditioning(*obs_bcs, *masks_list).unsqueeze(0).cuda()
            torch.manual_seed(i)
            cond_samples = cond_ddpm.sample(cond, n_samples=5)  # (5, 1, 1, 64, 64)
            cond_samples = cond_samples[:, 0]  # (5, 1, 64, 64)
            cond_mean = cond_samples.mean(dim=0, keepdim=True)

            cond_rel_l2 = relative_l2_error(cond_mean, field_true).item()
            cond_pde = pde_residual_norm(cond_mean).item()

            # --- DPS ---
            y_obs, obs_op = _build_obs_operator(obs_bcs, masks_list)
            torch.manual_seed(i + 10000)  # different seed from conditional
            dps_samples = sampler.sample(y_obs, obs_op, n_samples=5, H=64, W=64)
            dps_mean = dps_samples.mean(dim=0, keepdim=True)

            dps_rel_l2 = relative_l2_error(dps_mean, field_true).item()
            dps_pde = pde_residual_norm(dps_mean).item()

            # Observation agreement
            y_pred_dps = obs_op(dps_mean)
            dps_obs_rmse = (y_obs - y_pred_dps.squeeze(0)).pow(2).mean().sqrt().item()

            y_pred_cond = obs_op(cond_mean)
            cond_obs_rmse = (y_obs - y_pred_cond.squeeze(0)).pow(2).mean().sqrt().item()

            cond_per_example.append({
                "index": i, "rel_l2": cond_rel_l2,
                "pde_residual": cond_pde, "obs_rmse": cond_obs_rmse,
            })
            dps_per_example.append({
                "index": i, "rel_l2": dps_rel_l2,
                "pde_residual": dps_pde, "obs_rmse": dps_obs_rmse,
            })

            if (i + 1) % 20 == 0:
                cond_mean_so_far = np.mean([e["rel_l2"] for e in cond_per_example])
                dps_mean_so_far = np.mean([e["rel_l2"] for e in dps_per_example])
                print(f"    {i+1}/{n}  cond_rel_l2={cond_mean_so_far:.4f}  dps_rel_l2={dps_mean_so_far:.4f}")

        elapsed = time.time() - t0

        # Summarize
        cond_l2s = [e["rel_l2"] for e in cond_per_example]
        dps_l2s = [e["rel_l2"] for e in dps_per_example]

        pattern_results = {
            "conditional_ddpm": {
                "rel_l2_mean": float(np.mean(cond_l2s)),
                "rel_l2_median": float(np.median(cond_l2s)),
                "rel_l2_std": float(np.std(cond_l2s)),
                "rel_l2_iqr": [float(np.percentile(cond_l2s, 25)),
                                float(np.percentile(cond_l2s, 75))],
                "obs_rmse_mean": float(np.mean([e["obs_rmse"] for e in cond_per_example])),
                "pde_residual_mean": float(np.mean([e["pde_residual"] for e in cond_per_example])),
                "per_example": cond_per_example,
            },
            "dps": {
                "rel_l2_mean": float(np.mean(dps_l2s)),
                "rel_l2_median": float(np.median(dps_l2s)),
                "rel_l2_std": float(np.std(dps_l2s)),
                "rel_l2_iqr": [float(np.percentile(dps_l2s, 25)),
                                float(np.percentile(dps_l2s, 75))],
                "obs_rmse_mean": float(np.mean([e["obs_rmse"] for e in dps_per_example])),
                "pde_residual_mean": float(np.mean([e["pde_residual"] for e in dps_per_example])),
                "per_example": dps_per_example,
            },
            "time_seconds": elapsed,
        }

        print(f"\n  {pattern} completed in {elapsed:.1f}s")
        print(f"  Conditional DDPM: rel_l2={pattern_results['conditional_ddpm']['rel_l2_mean']:.4f}  "
              f"obs_rmse={pattern_results['conditional_ddpm']['obs_rmse_mean']:.4f}  "
              f"pde_res={pattern_results['conditional_ddpm']['pde_residual_mean']:.2f}")
        print(f"  DPS:              rel_l2={pattern_results['dps']['rel_l2_mean']:.4f}  "
              f"obs_rmse={pattern_results['dps']['obs_rmse_mean']:.4f}  "
              f"pde_res={pattern_results['dps']['pde_residual_mean']:.2f}")

        # DPS advantage ratio
        ratio = pattern_results['conditional_ddpm']['rel_l2_mean'] / max(pattern_results['dps']['rel_l2_mean'], 1e-8)
        if ratio > 1:
            print(f"  → DPS wins by {ratio:.1f}x on rel L2")
        else:
            print(f"  → Conditional DDPM wins by {1/ratio:.1f}x on rel L2")

        results[pattern] = pattern_results

        # Checkpoint
        with open(partial_path, "w") as f:
            json.dump({"results": results, "max_samples": max_samples}, f, indent=2)
        volume.commit()

    # Final save
    out_path = os.path.join(out_dir, "zero_shot_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "max_samples": max_samples}, f, indent=2)
    volume.commit()
    print(f"\nAll done! Saved to {out_path}")
    return results


@app.local_entrypoint()
def main(max_samples: int = 100):
    run_zero_shot.remote(max_samples=max_samples)
