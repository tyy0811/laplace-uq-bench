"""Quick DDPM conditioning collapse diagnostic.

Generates 3 samples from 5 diverse test inputs.
If samples change with BCs → conditioning works.
If all 15 look identical → conditioning is broken.

Usage (Modal):
    modal run modal_deploy/evaluate_remote.py --eval-type diagnose

Usage (local):
    python scripts/diagnose_ddpm.py --checkpoint experiments/ddpm_phase2/best.pt
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from diffphys.model.trainer import build_model, load_config
from diffphys.model.ddpm import DDPM
from diffphys.data.dataset import LaplacePDEDataset


def run_diagnostic(checkpoint_path, test_npz="data/test_in.npz",
                   config_path="configs/ddpm_phase2.yaml", device="cpu"):
    config = load_config(config_path)
    model = build_model(config["model"]).to(device)
    ddpm = DDPM(model, **config["ddpm"]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    ddpm.load_state_dict(ckpt["model_state_dict"])

    ds = LaplacePDEDataset(test_npz, regime="exact")

    # Pick 5 diverse samples (spread across the dataset)
    indices = [0, len(ds)//4, len(ds)//2, 3*len(ds)//4, len(ds)-1]

    results = {"ground_truth": [], "conditioning": [], "samples": []}

    for idx in indices:
        cond, target = ds[idx]
        cond = cond.unsqueeze(0).to(device)  # (1, 8, 64, 64)
        target = target.unsqueeze(0)          # (1, 1, 64, 64)

        # Generate 3 samples
        samples = ddpm.sample(cond, n_samples=3)  # (3, 1, 1, 64, 64)

        results["ground_truth"].append(target.cpu().numpy())
        results["conditioning"].append(cond.cpu().numpy())
        results["samples"].append(samples.cpu().numpy())

    results["ground_truth"] = np.concatenate(results["ground_truth"], axis=0)  # (5, 1, 64, 64)
    results["conditioning"] = np.concatenate(results["conditioning"], axis=0)  # (5, 8, 64, 64)
    results["samples"] = np.stack(results["samples"], axis=0)                   # (5, 3, 1, 1, 64, 64)

    # Diagnostic metrics
    print("\n=== Conditioning Collapse Diagnostic ===\n")

    # 1. Do samples vary across different BCs?
    sample_means = results["samples"][:, :, 0, 0].mean(axis=1)  # (5, 64, 64) mean per input
    cross_bc_std = sample_means.std(axis=0).mean()
    print(f"Cross-BC std (should be >> 0 if conditioning works): {cross_bc_std:.6f}")

    # 2. Are samples for the same BC similar to each other?
    within_bc_stds = []
    for i in range(5):
        s = results["samples"][i, :, 0, 0]  # (3, 64, 64)
        within_bc_stds.append(s.std(axis=0).mean())
    avg_within_std = np.mean(within_bc_stds)
    print(f"Within-BC std (sample diversity): {avg_within_std:.6f}")

    # 3. How close are samples to ground truth?
    for i in range(5):
        gt = results["ground_truth"][i, 0]  # (64, 64)
        sample_mean = results["samples"][i, :, 0, 0].mean(axis=0)  # (64, 64)
        mse = ((gt - sample_mean) ** 2).mean()
        print(f"  Input {i}: MSE(sample_mean, truth) = {mse:.6f}")

    # Verdict
    print()
    if cross_bc_std < 0.01:
        print("WARNING: Cross-BC std is very low — possible conditioning collapse!")
        print("Action: Debug conditioning path before proceeding.")
    elif cross_bc_std > 0.1:
        print("PASS: Samples clearly respond to different BCs.")
        if avg_within_std < 0.001:
            print("NOTE: Very low within-BC diversity — model may be mode-collapsing to mean.")
        else:
            print(f"Within-BC diversity is {avg_within_std:.4f} — healthy spread.")
    else:
        print(f"MARGINAL: Cross-BC std = {cross_bc_std:.4f}. Conditioning partially working.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-npz", default="data/test_in.npz")
    parser.add_argument("--config", default="configs/ddpm_phase2.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_diagnostic(args.checkpoint, args.test_npz, args.config, args.device)
