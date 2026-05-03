#!/bin/bash
#SBATCH --job-name=dpo_d4_eval
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=dpo_artifacts/eval_%j.out
#SBATCH --error=dpo_artifacts/eval_%j.err

# Day 4: sample 2k crystals from each DPO checkpoint, score with CHGNet,
# emit a hit-rate comparison table vs the Day-1 baseline pool.

set -e
PROJECT_DIR=/nfs/roberts/project/pi_jks79/jw2933/GuidedMatDiffusion
cd "${PROJECT_DIR}"
source .venv/bin/activate

REF_MODEL=${PROJECT_DIR}/models/superconductor_generator
N_BATCHES=${N_BATCHES:-20}
BATCH_SIZE=${BATCH_SIZE:-100}
BAND_GAP=${BAND_GAP:-1.2218}
GUIDE_W=${GUIDE_W:-2.0}

for BETA in 1 5 25; do
    DPO_DIR=${PROJECT_DIR}/dpo_artifacts/dpo_b${BETA}
    DPO_CKPT=${DPO_DIR}/dpo_final.ckpt
    SAVE_DIR=${PROJECT_DIR}/dpo_artifacts/eval/cifs_dpo_b${BETA}
    SCORES=${PROJECT_DIR}/dpo_artifacts/eval/scores_dpo_b${BETA}.json

    if [ ! -f "${DPO_CKPT}" ]; then
        echo "[skip] β=${BETA}: no checkpoint at ${DPO_CKPT}"
        continue
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
done

echo "=== summary ==="
python -c "
import json, numpy as np
from pathlib import Path

# Day-1 baseline scores → defines the bottom-25% threshold for the hit-rate metric
baseline = json.loads(Path('dpo_artifacts/day1_baseline_pool/scores_y10k.json').read_text())
b_e = np.array([r['energy_per_atom'] for r in baseline if r['valid']])
thresh = np.percentile(b_e, 25)   # baseline's 25th-percentile = 'top quartile of stability'
print(f'baseline: n={len(b_e)}  median E/atom={np.median(b_e):.3f}  25th-pct (hit threshold)={thresh:.3f}')
print(f'baseline hit-rate (E/atom < threshold): {(b_e < thresh).mean():.3%}  (= 25% by construction)')

print()
print(f'{\"pool\":<12} {\"n\":>5} {\"valid\":>6} {\"med E/atom\":>11} {\"hit-rate\":>9} {\"Δ vs base\":>11}')
print('-' * 60)
print(f'{\"baseline\":<12} 2000 {len(b_e):>6} {np.median(b_e):>11.3f} {0.25:>9.3%} {0.0:>+11.3%}')
for beta in [1, 5, 25]:
    p = Path(f'dpo_artifacts/eval/scores_dpo_b{beta}.json')
    if not p.exists():
        print(f'dpo_b{beta:<6} (skipped — no scores file)')
        continue
    rows = json.loads(p.read_text())
    e = np.array([r['energy_per_atom'] for r in rows if r['valid']])
    hr = (e < thresh).mean() if len(e) else 0.0
    print(f'{\"dpo_b\"+str(beta):<12} {len(rows):>5} {len(e):>6} {np.median(e):>11.3f} {hr:>9.3%} {hr-0.25:>+11.3%}')
"

echo "=== Day 4 done ==="
