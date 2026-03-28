"""Phase 2 UQ evaluation: ensemble vs DDPM under observation regimes."""

import json
from pathlib import Path

import torch
import numpy as np

from ..data.dataset import LaplacePDEDataset
from ..data.observation import REGIMES
from ..model.trainer import build_model, load_config
from ..model.ensemble import EnsemblePredictor
from ..model.ddpm import DDPM
from .uq_metrics import pixelwise_coverage, crps_gaussian, calibration_error, sharpness


def evaluate_ensemble_uq(ensemble, loader, device):
    """Evaluate ensemble UQ metrics."""
    all_true, all_mean, all_std = [], [], []

    for cond, target in loader:
        cond, target = cond.to(device), target.to(device)
        mean, var = ensemble.predict(cond)
        std = var.sqrt().clamp(min=1e-8)
        all_true.append(target.cpu())
        all_mean.append(mean.cpu())
        all_std.append(std.cpu())

    true = torch.cat(all_true)
    mean = torch.cat(all_mean)
    std = torch.cat(all_std)

    return _compute_uq_summary(true, mean, std)


def evaluate_ddpm_uq(ddpm, loader, device, n_samples=5):
    """Evaluate DDPM UQ metrics using sample mean/std."""
    all_true, all_mean, all_std = [], [], []

    for cond, target in loader:
        cond, target = cond.to(device), target.to(device)
        samples = ddpm.sample(cond, n_samples=n_samples)  # (K, B, 1, H, W)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0).clamp(min=1e-8)
        all_true.append(target.cpu())
        all_mean.append(mean.cpu())
        all_std.append(std.cpu())

    true = torch.cat(all_true)
    mean = torch.cat(all_mean)
    std = torch.cat(all_std)

    return _compute_uq_summary(true, mean, std)


def _compute_uq_summary(true, mean, std):
    return {
        "coverage_50": pixelwise_coverage(true, mean, std, 0.50).item(),
        "coverage_90": pixelwise_coverage(true, mean, std, 0.90).item(),
        "coverage_95": pixelwise_coverage(true, mean, std, 0.95).item(),
        "crps": crps_gaussian(true, mean, std).mean().item(),
        "calibration_error": calibration_error(true, mean, std).item(),
        "sharpness": sharpness(std).item(),
    }


def run_phase2_evaluation(model_type, config_path, checkpoint_paths,
                          test_npz, device="cpu"):
    """Evaluate a model under all observation regimes.

    Args:
        model_type: "ensemble" or "ddpm".
        config_path: YAML config path.
        checkpoint_paths: List of paths (ensemble) or single path (ddpm).
        test_npz: Path to test .npz file.
        device: Compute device.

    Returns:
        Dict mapping regime -> UQ metrics.
    """
    config = load_config(config_path)
    results = {}

    for regime in REGIMES:
        ds = LaplacePDEDataset(test_npz, regime=regime)
        loader = torch.utils.data.DataLoader(ds, batch_size=32)

        if model_type == "ensemble":
            models = []
            for cp in checkpoint_paths:
                m = build_model(config["model"]).to(device)
                ckpt = torch.load(cp, map_location=device)
                m.load_state_dict(ckpt["model_state_dict"])
                models.append(m)
            ensemble = EnsemblePredictor(models)
            results[regime] = evaluate_ensemble_uq(ensemble, loader, device)

        elif model_type == "ddpm":
            model = build_model(config["model"]).to(device)
            ddpm = DDPM(model, **config["ddpm"]).to(device)
            ckpt = torch.load(checkpoint_paths[0], map_location=device)
            ddpm.load_state_dict(ckpt["model_state_dict"])
            results[regime] = evaluate_ddpm_uq(ddpm, loader, device, n_samples=5)

    return results
