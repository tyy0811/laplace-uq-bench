"""Phase 2 UQ evaluation: ensemble vs DDPM under observation regimes."""

import json
from pathlib import Path

import torch
import numpy as np

from ..data.dataset import LaplacePDEDataset
from ..data.observation import REGIMES
from ..model.trainer import build_model, load_config, _build_ddpm
from ..model.ensemble import EnsemblePredictor
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


def evaluate_cfm_uq(cfm, loader, device, n_samples=20):
    """Evaluate flow matching UQ metrics using sample mean/std."""
    all_true, all_mean, all_std = [], [], []
    total_batches = len(loader)

    for batch_idx, (cond, target) in enumerate(loader):
        cond, target = cond.to(device), target.to(device)
        samples = cfm.sample(cond, n_samples=n_samples)  # (K, B, 1, H, W)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0).clamp(min=1e-8)
        all_true.append(target.cpu())
        all_mean.append(mean.cpu())
        all_std.append(std.cpu())
        if (batch_idx + 1) % 2 == 0 or batch_idx == total_batches - 1:
            print(f"    Batch {batch_idx + 1}/{total_batches}")

    true = torch.cat(all_true)
    mean = torch.cat(all_mean)
    std = torch.cat(all_std)

    return _compute_uq_summary(true, mean, std)


def evaluate_ddpm_uq(ddpm, loader, device, n_samples=20):
    """Evaluate DDPM UQ metrics using sample mean/std."""
    all_true, all_mean, all_std = [], [], []
    total_batches = len(loader)

    for batch_idx, (cond, target) in enumerate(loader):
        cond, target = cond.to(device), target.to(device)
        samples = ddpm.sample(cond, n_samples=n_samples)  # (K, B, 1, H, W)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0).clamp(min=1e-8)
        all_true.append(target.cpu())
        all_mean.append(mean.cpu())
        all_std.append(std.cpu())
        if (batch_idx + 1) % 2 == 0 or batch_idx == total_batches - 1:
            print(f"    Batch {batch_idx + 1}/{total_batches}")

    true = torch.cat(all_true)
    mean = torch.cat(all_mean)
    std = torch.cat(all_std)

    return _compute_uq_summary(true, mean, std)


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


# Normal quantile lookup (avoids scipy dependency on Modal)
_NORM_Z = {0.50: 0.6745, 0.90: 1.6449, 0.95: 1.9600}


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
        Dict with raw and conformal metrics at each target.
    """
    from .conformal import SpatialConformalPredictor, PixelwiseConformalPredictor

    results = {}

    for target in targets:
        pct = int(target * 100)
        alpha = 1.0 - target

        # Raw coverage (no conformal) on test set using Gaussian quantiles
        if target not in _NORM_Z:
            raise ValueError(
                f"Unsupported target {target}. Supported: {sorted(_NORM_Z.keys())}"
            )
        z = _NORM_Z[target]
        raw_lower = test_mean - z * np.maximum(test_std, 1e-8)
        raw_upper = test_mean + z * np.maximum(test_std, 1e-8)
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
                          test_npz, device="cpu", n_samples=None):
    """Evaluate a model under all observation regimes.

    Args:
        model_type: "ensemble", "ddpm", or "flow_matching".
        config_path: YAML config path.
        checkpoint_paths: List of paths (ensemble) or single path (ddpm/fm).
        test_npz: Path to test .npz file.
        device: Compute device.
        n_samples: Number of samples for generative models (default per model).

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
            ddpm = _build_ddpm(model, config["ddpm"]).to(device)
            ckpt = torch.load(checkpoint_paths[0], map_location=device)
            ddpm.load_state_dict(ckpt["model_state_dict"])
            results[regime] = evaluate_ddpm_uq(
                ddpm, loader, device, n_samples=n_samples or 20)

        elif model_type == "flow_matching":
            from ..model.flow_matching import ConditionalFlowMatcher
            model = build_model(config["model"]).to(device)
            fm_cfg = config["flow_matching"]
            cfm = ConditionalFlowMatcher(
                model,
                use_ot=fm_cfg.get("use_ot", True),
                n_sample_steps=fm_cfg.get("n_sample_steps", 50),
            ).to(device)
            ckpt = torch.load(checkpoint_paths[0], map_location=device)
            cfm.load_state_dict(ckpt["model_state_dict"])
            results[regime] = evaluate_cfm_uq(
                cfm, loader, device, n_samples=n_samples or 20)

    return results
