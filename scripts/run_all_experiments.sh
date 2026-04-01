#!/bin/bash
set -euo pipefail

echo "=== Phase 1: Deterministic Baselines ==="

echo "Training U-Net regressor..."
python scripts/train.py --config configs/unet_regressor.yaml --device cuda

echo "Training FNO..."
python scripts/train.py --config configs/fno.yaml --device cuda

echo "Evaluating Phase 1 models..."
python scripts/evaluate.py --config configs/unet_regressor.yaml \
    --checkpoint experiments/unet_regressor/best.pt \
    --output experiments/unet_regressor/results.json --device cuda

python scripts/evaluate.py --config configs/fno.yaml \
    --checkpoint experiments/fno/best.pt \
    --output experiments/fno/results.json --device cuda

echo "=== Phase 2: Probabilistic Models (mixed-regime training) ==="

echo "Training ensemble (5 members, mixed regime)..."
python scripts/train.py --config configs/ensemble_phase2.yaml --device cuda

echo "Training Improved DDPM (cosine schedule, v-prediction, Min-SNR)..."
python scripts/train.py --config configs/ddpm_improved.yaml --device cuda

echo "Training Flow Matching (mixed regime, OT coupling)..."
python scripts/train.py --config configs/flow_matching.yaml --device cuda

echo "Phase 2 UQ evaluation..."
python scripts/evaluate_phase2.py --model-type ensemble \
    --config configs/ensemble_phase2.yaml \
    --checkpoints experiments/ensemble_phase2/member_*/best.pt \
    --output experiments/ensemble_phase2/uq_results.json --device cuda

python scripts/evaluate_phase2.py --model-type ddpm \
    --config configs/ddpm_improved.yaml \
    --checkpoints experiments/ddpm_improved/best.pt \
    --n-samples 20 \
    --output experiments/ddpm_improved/uq_results.json --device cuda

python scripts/evaluate_phase2.py --model-type flow_matching \
    --config configs/flow_matching.yaml \
    --checkpoints experiments/flow_matching/best.pt \
    --output experiments/flow_matching/uq_results.json --device cuda

echo "=== Done ==="
