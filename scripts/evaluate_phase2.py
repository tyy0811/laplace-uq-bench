"""Evaluate Phase 2 models (ensemble and DDPM) under all observation regimes.

Usage:
    python scripts/evaluate_phase2.py --model-type ensemble \
        --config configs/ensemble_phase2.yaml \
        --checkpoints experiments/ensemble_phase2/member_*/best.pt

    python scripts/evaluate_phase2.py --model-type ddpm \
        --config configs/ddpm_phase2.yaml \
        --checkpoints experiments/ddpm_phase2/best.pt
"""

import argparse
import json

from diffphys.evaluation.evaluate_uq import run_phase2_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["ensemble", "ddpm"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--test-npz", default="data/test_in.npz")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = run_phase2_evaluation(
        args.model_type, args.config, args.checkpoints,
        args.test_npz, args.device,
    )

    for regime, metrics in results.items():
        print(f"\n=== {regime} ===")
        for name, val in metrics.items():
            print(f"  {name:20s}: {val:.6f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
