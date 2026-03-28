"""Model evaluation on test splits with physics-aware metrics."""

import json
from pathlib import Path

import torch
import numpy as np

from ..data.dataset import LaplacePDEDataset
from ..model.trainer import build_model, load_config
from .metrics import (
    relative_l2_error,
    pde_residual_norm,
    bc_error,
    max_principle_violations,
    energy_functional,
)


def evaluate_regressor(model, loader, device, h=1.0 / 63):
    """Evaluate a deterministic model on a dataset.

    Returns dict of metric_name -> list of per-sample values.
    """
    model.eval()
    results = {
        "rel_l2": [], "pde_residual": [], "bc_err": [],
        "max_viol": [], "energy_pred": [], "energy_true": [],
    }

    with torch.no_grad():
        for cond, target in loader:
            cond, target = cond.to(device), target.to(device)
            pred = model(cond)

            results["rel_l2"].append(relative_l2_error(pred, target))
            results["pde_residual"].append(pde_residual_norm(pred, h))
            results["bc_err"].append(bc_error(pred, target))
            results["max_viol"].append(max_principle_violations(pred))
            results["energy_pred"].append(energy_functional(pred, h))
            results["energy_true"].append(energy_functional(target, h))

    return {k: torch.cat(v).cpu().numpy().tolist() for k, v in results.items()}


def summarize_results(results):
    """Compute mean and std for each metric."""
    summary = {}
    for k, vals in results.items():
        arr = np.array(vals)
        summary[k] = {"mean": float(arr.mean()), "std": float(arr.std())}
    # Energy error as relative difference
    e_pred = np.array(results["energy_pred"])
    e_true = np.array(results["energy_true"])
    rel_energy = np.abs(e_pred - e_true) / np.maximum(e_true, 1e-8)
    summary["rel_energy_err"] = {"mean": float(rel_energy.mean()), "std": float(rel_energy.std())}
    return summary


def run_evaluation(config_path, checkpoint_path, test_npz_paths, device="cpu"):
    """Full evaluation pipeline."""
    config = load_config(config_path)
    model = build_model(config["model"]).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    all_results = {}
    for split_name, npz_path in test_npz_paths.items():
        ds = LaplacePDEDataset(npz_path)
        loader = torch.utils.data.DataLoader(ds, batch_size=64)
        raw = evaluate_regressor(model, loader, device)
        all_results[split_name] = summarize_results(raw)

    return all_results
