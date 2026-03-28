"""Evaluate a trained model on test splits.

Usage:
    python scripts/evaluate.py --config configs/unet_regressor.yaml \
        --checkpoint experiments/unet_regressor/best.pt
"""

import argparse
import json

from diffphys.evaluation.evaluate import run_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args()

    test_splits = {
        "test_in": "data/test_in.npz",
        "test_ood": "data/test_ood.npz",
    }

    results = run_evaluation(args.config, args.checkpoint, test_splits, args.device)

    for split, metrics in results.items():
        print(f"\n=== {split} ===")
        for name, stats in metrics.items():
            print(f"  {name:20s}: {stats['mean']:.6f} +/- {stats['std']:.6f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
