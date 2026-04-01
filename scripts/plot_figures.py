"""Generate Phase 2 evaluation figures.

Usage:
    python scripts/plot_figures.py --figure 9    # Conformal calibration plot
    python scripts/plot_figures.py --figure 10   # Training convergence comparison
    python scripts/plot_figures.py --all          # All figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path("figures")


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 9: Conformal calibration plot
# ---------------------------------------------------------------------------

def figure_9_conformal_calibration():
    """Target coverage vs achieved coverage for raw, pixelwise, and spatial conformal."""
    data = load_json("experiments/conformal/conformal_results.json")["results"]
    targets = [50, 90, 95]
    target_fracs = [t / 100 for t in targets]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    model_names = {"ensemble": "Ensemble", "flow_matching": "Flow Matching", "ddpm_improved": "Improved DDPM"}
    regimes = ["exact", "dense-noisy", "sparse-clean", "sparse-noisy", "very-sparse"]

    for ax, (model_key, model_label) in zip(axes, model_names.items()):
        model_data = data[model_key]

        # Average across regimes for each target
        raw_coverages = []
        pixel_coverages = []
        spatial_coverages = []

        for t in targets:
            raw_vals = [model_data[r][f"raw_coverage_{t}"] for r in regimes]
            pixel_vals = [model_data[r][f"pixelwise_{t}_coverage"] for r in regimes]
            spatial_vals = [model_data[r][f"spatial_{t}_coverage"] for r in regimes]
            raw_coverages.append(np.mean(raw_vals))
            pixel_coverages.append(np.mean(pixel_vals))
            spatial_coverages.append(np.mean(spatial_vals))

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
        ax.plot(target_fracs, raw_coverages, "o-", color="#d62728", label="Raw", markersize=8)
        ax.plot(target_fracs, pixel_coverages, "s-", color="#2ca02c", label="Pixelwise conformal", markersize=8)
        ax.plot(target_fracs, spatial_coverages, "^-", color="#1f77b4", label="Spatial conformal", markersize=8)

        ax.set_xlabel("Target coverage", fontsize=12)
        ax.set_title(model_label, fontsize=13, fontweight="bold")
        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        if ax == axes[0]:
            ax.set_ylabel("Achieved coverage", fontsize=12)
            ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Figure 9: Conformal Calibration (averaged across 5 regimes)", fontsize=14, y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "fig9_conformal_calibration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 9b: Conformal calibration per regime (detailed)
# ---------------------------------------------------------------------------

def figure_9b_conformal_per_regime():
    """Per-regime conformal calibration for ensemble (the main conformal story)."""
    data = load_json("experiments/conformal/conformal_results.json")["results"]["ensemble"]
    targets = [50, 90, 95]
    target_fracs = [t / 100 for t in targets]
    regimes = ["exact", "dense-noisy", "sparse-clean", "sparse-noisy", "very-sparse"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, method, title in [
        (axes[0], "raw_coverage", "Raw Ensemble"),
        (axes[1], "pixelwise", "Ensemble + Pixelwise Conformal"),
    ]:
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")
        for regime, color in zip(regimes, colors):
            if method == "raw_coverage":
                coverages = [data[regime][f"raw_coverage_{t}"] for t in targets]
            else:
                coverages = [data[regime][f"pixelwise_{t}_coverage"] for t in targets]
            ax.plot(target_fracs, coverages, "o-", color=color, label=regime, markersize=7)

        ax.set_xlabel("Target coverage", fontsize=12)
        ax.set_ylabel("Achieved coverage", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Figure 9b: Ensemble Calibration by Regime", fontsize=14, y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "fig9b_conformal_per_regime.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 10: Training convergence comparison
# ---------------------------------------------------------------------------

def figure_10_convergence():
    """Training convergence: FM vs Improved DDPM (val loss vs epoch)."""
    fm_hist = load_json("experiments/flow_matching/history.json")
    ddpm_hist = load_json("experiments/ddpm_improved/history.json")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: train loss
    ax = axes[0]
    fm_epochs = [h["epoch"] for h in fm_hist]
    fm_train = [h["train_loss"] for h in fm_hist]
    ddpm_epochs = [h["epoch"] for h in ddpm_hist]
    ddpm_train = [h["train_loss"] for h in ddpm_hist]

    ax.semilogy(fm_epochs, fm_train, "-", color="#ff7f0e", label="Flow Matching", linewidth=2)
    ax.semilogy(ddpm_epochs, ddpm_train, "-", color="#1f77b4", label="Improved DDPM", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Train Loss (log scale)", fontsize=12)
    ax.set_title("Training Loss", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Right: val loss
    ax = axes[1]
    fm_val = [h["val_loss"] for h in fm_hist]
    ddpm_val = [h["val_loss"] for h in ddpm_hist]

    ax.semilogy(fm_epochs, fm_val, "-", color="#ff7f0e", label="Flow Matching", linewidth=2, alpha=0.7)
    ax.semilogy(ddpm_epochs, ddpm_val, "-", color="#1f77b4", label="Improved DDPM", linewidth=2, alpha=0.7)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Val Loss (log scale)", fontsize=12)
    ax.set_title("Validation Loss", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 10: Training Convergence — Flow Matching vs Improved DDPM", fontsize=14, y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "fig10_convergence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure: Interval width comparison
# ---------------------------------------------------------------------------

def figure_interval_widths():
    """Mean interval width at 90% coverage across regimes for each model."""
    data = load_json("experiments/conformal/conformal_results.json")["results"]
    regimes = ["exact", "dense-noisy", "sparse-clean", "sparse-noisy", "very-sparse"]
    regime_labels = ["Exact", "Dense\nnoisy", "Sparse\nclean", "Sparse\nnoisy", "Very\nsparse"]
    models = {"ensemble": "Ensemble", "flow_matching": "Flow Matching", "ddpm_improved": "Improved DDPM"}
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(regimes))
    width = 0.25

    for i, (model_key, model_label) in enumerate(models.items()):
        widths = [data[model_key][r]["pixelwise_90_mean_width"] for r in regimes]
        ax.bar(x + i * width, widths, width, label=model_label, color=colors[i], alpha=0.85)

    ax.set_xlabel("Observation Regime", fontsize=12)
    ax.set_ylabel("Mean Interval Width (pixelwise, 90%)", fontsize=12)
    ax.set_title("Conformal Prediction Interval Sharpness", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(regime_labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = FIGURES_DIR / "fig_interval_widths.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 7: Functional CRPS bar chart (matched 5v5)
# ---------------------------------------------------------------------------

def figure_7_functional_crps():
    """Bar chart comparing functional CRPS across models for 5 derived quantities."""
    data = load_json("experiments/functional_crps/functional_crps_results.json")["results"]

    quantities = ["center_T", "subregion_mean_T", "max_interior_T", "dirichlet_energy", "top_edge_flux"]
    labels = ["Center T", "Subregion\nMean T", "Max\nInterior T", "Dirichlet\nEnergy", "Top Edge\nFlux"]
    models = [("ensemble", "Ensemble"), ("flow_matching", "Flow Matching"), ("ddpm_improved", "Improved DDPM")]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(quantities))
    width = 0.25

    for i, (model_key, model_label) in enumerate(models):
        vals = [data[model_key][f"mean_crps_{q}"] for q in quantities]
        errs = [data[model_key][f"std_crps_{q}"] for q in quantities]
        ax.bar(x + i * width, vals, width, label=model_label, color=colors[i],
               yerr=errs, capsize=3, alpha=0.85)

    ax.set_ylabel("CRPS (lower is better)", fontsize=12)
    ax.set_title("Functional CRPS — Matched 5 Samples (Sparse-Noisy Regime)", fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = FIGURES_DIR / "fig7_functional_crps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", type=str, help="Figure number to generate (7, 9, 9b, 10, widths)")
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(exist_ok=True)

    if args.all or args.figure == "9":
        figure_9_conformal_calibration()
    if args.all or args.figure == "9b":
        figure_9b_conformal_per_regime()
    if args.all or args.figure == "10":
        figure_10_convergence()
    if args.all or args.figure == "widths":
        figure_interval_widths()
    if args.all or args.figure == "7":
        figure_7_functional_crps()

    if args.all:
        print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
