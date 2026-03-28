"""Train a model from a YAML config file.

Usage:
    python scripts/train.py --config configs/unet_regressor.yaml
    python scripts/train.py --config configs/unet_regressor.yaml --device cuda
"""

import argparse

import torch

from diffphys.model.trainer import load_config, train, train_ddpm, train_ensemble


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", default=None, help="Device (cpu/cuda/mps)")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if "ensemble" in config:
        print(f"Training {len(config['ensemble']['seeds'])}-member ensemble on {device}")
        train_ensemble(config, device=device)
    elif "ddpm" in config:
        print(f"Training DDPM on {device}")
        train_ddpm(config, device=device)
    else:
        print(f"Training {config['model']['type']} on {device}")
        train(config, device=device)


if __name__ == "__main__":
    main()
