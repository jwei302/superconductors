#!/bin/bash
#SBATCH --job-name=dpo_train
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=dpo_artifacts/dpo_b%x_%j.out
#SBATCH --error=dpo_artifacts/dpo_b%x_%j.err

# Day 3: train DPO at one β value on the pre-built pairs file.
# Pass beta via the BETA env var; the script tags the output dir with it.

set -e
PROJECT_DIR=/nfs/roberts/project/pi_jks79/jw2933/GuidedMatDiffusion
cd "${PROJECT_DIR}"
source .venv/bin/activate

PAIRS_FILE="${PROJECT_DIR}/dpo_artifacts/pairs_v1.pt"
MODEL_PATH="${PROJECT_DIR}/models/superconductor_generator"
BETA=${BETA:?BETA must be set, e.g. BETA=5 sbatch train_dpo.sbatch}
OUT_DIR="${PROJECT_DIR}/dpo_artifacts/dpo_b${BETA}"
MAX_STEPS=${MAX_STEPS:-5000}
VAL_EVERY=${VAL_EVERY:-500}

echo "=== DPO training: β=${BETA}, max_steps=${MAX_STEPS}, val_every=${VAL_EVERY} ==="
nvidia-smi -L

python scripts/dpo/dpo_train.py \
    --pairs_file "${PAIRS_FILE}" \
    --model_path "${MODEL_PATH}" \
    --out_dir "${OUT_DIR}" \
    --beta "${BETA}" \
    --max_steps "${MAX_STEPS}" \
    --val_every "${VAL_EVERY}" \
    --batch_size 1 \
    --lr 1e-4 \
    --gpus 1 \
    --seed 0

echo "=== done; checkpoint at ${OUT_DIR}/dpo_final.ckpt ==="
