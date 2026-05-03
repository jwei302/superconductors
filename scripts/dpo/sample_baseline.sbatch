#!/bin/bash
#SBATCH --job-name=dpo_d1_sample
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=dpo_artifacts/day1_baseline_pool/sample_%j.out
#SBATCH --error=dpo_artifacts/day1_baseline_pool/sample_%j.err

# Day 1: sample 2,000 crystals from π_ref (SFT'd GuidedMatDiffusion checkpoint)
# at scaled y = +1.2218 (Tc = 10 K target). Also score them with CHGNet.

set -e
PROJECT_DIR=/nfs/roberts/project/pi_jks79/jw2933/GuidedMatDiffusion
cd "${PROJECT_DIR}"
source .venv/bin/activate

MODEL_PATH="${PROJECT_DIR}/models/superconductor_generator"   # must be absolute for Hydra
SAVE_DIR="${PROJECT_DIR}/dpo_artifacts/day1_baseline_pool/cifs_y10k"
BAND_GAP=1.2218          # scaled Tc=10K (mean=3.604, std=5.235)
GUIDE_W=${GUIDE_W:-2.0}  # CFG weight; override via env if desired
BATCH_SIZE=${BATCH_SIZE:-100}
NUM_BATCHES=${NUM_BATCHES:-20}   # 100 * 20 = 2,000 crystals

echo "=== sampling ${BATCH_SIZE} x ${NUM_BATCHES} = $((BATCH_SIZE * NUM_BATCHES)) crystals at scaled y=${BAND_GAP}, guide_w=${GUIDE_W} ==="
nvidia-smi -L
python scripts/generation.py \
    --model_path "${MODEL_PATH}" \
    --dataset supccomb_12 \
    --save_path "${SAVE_DIR}" \
    --band_gap "${BAND_GAP}" \
    --guide_w "${GUIDE_W}" \
    --batch_size "${BATCH_SIZE}" \
    --num_batches_to_samples "${NUM_BATCHES}" \
    --label "baseline_y10k_w${GUIDE_W}"

echo "=== scoring with CHGNet single-point ==="
python scripts/dpo/score_ehull.py \
    --cif_dir "${SAVE_DIR}" \
    --out dpo_artifacts/day1_baseline_pool/scores_y10k.json \
    --cache dpo_artifacts/score_cache/chgnet_cache.json \
    --batch_size 64

echo "=== Day 1 done ==="
