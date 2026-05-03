#!/bin/bash
#SBATCH --job-name=dpo_eval
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=dpo_artifacts/eval_b%x_%j.out
#SBATCH --error=dpo_artifacts/eval_b%x_%j.err

# Day 4: sample 2k crystals from one DPO checkpoint at the given β, score with CHGNet.
# BETA env var required.

set -e
PROJECT_DIR=/nfs/roberts/project/pi_jks79/jw2933/GuidedMatDiffusion
cd "${PROJECT_DIR}"
source .venv/bin/activate

BETA=${BETA:?BETA must be set}
REF_MODEL=${PROJECT_DIR}/models/superconductor_generator
N_BATCHES=${N_BATCHES:-20}
BATCH_SIZE=${BATCH_SIZE:-100}
BAND_GAP=${BAND_GAP:-1.2218}
GUIDE_W=${GUIDE_W:-2.0}

DPO_CKPT=${PROJECT_DIR}/dpo_artifacts/dpo_b${BETA}/dpo_final.ckpt
SAVE_DIR=${PROJECT_DIR}/dpo_artifacts/eval/cifs_dpo_b${BETA}
SCORES=${PROJECT_DIR}/dpo_artifacts/eval/scores_dpo_b${BETA}.json

if [ ! -f "${DPO_CKPT}" ]; then
    echo "[error] no checkpoint at ${DPO_CKPT}" >&2
    exit 1
fi

echo "=== β=${BETA}: sampling 2,000 from DPO ckpt ==="
python scripts/dpo/sample_dpo_ckpt.py \
    --ref_model_path "${REF_MODEL}" \
    --dpo_ckpt "${DPO_CKPT}" \
    --save_path "${SAVE_DIR}" \
    --band_gap "${BAND_GAP}" \
    --guide_w "${GUIDE_W}" \
    --batch_size "${BATCH_SIZE}" \
    --num_batches_to_samples "${N_BATCHES}" \
    --seed 0 \
    --label "dpo_b${BETA}_y10k_w${GUIDE_W}"

echo "=== β=${BETA}: scoring with CHGNet single-point ==="
python scripts/dpo/score_ehull.py \
    --cif_dir "${SAVE_DIR}" \
    --out "${SCORES}" \
    --cache "${PROJECT_DIR}/dpo_artifacts/score_cache/chgnet_cache.json" \
    --batch_size 32

echo "=== β=${BETA} done ==="
