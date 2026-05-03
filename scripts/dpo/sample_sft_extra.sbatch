#!/bin/bash
#SBATCH --job-name=sft_8k
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=dpo_artifacts/sft_8k_%j.out
#SBATCH --error=dpo_artifacts/sft_8k_%j.err

# Sample 8000 fresh crystals from π_ref (no DPO ckpt → falls back to SFT) at seed=1.
# This gives the "best-of-4N" SFT screening pool to compare against DPO@β=1 at matched compute.

set -e
PROJECT_DIR=/nfs/roberts/project/pi_jks79/jw2933/GuidedMatDiffusion
cd "${PROJECT_DIR}"
source .venv/bin/activate

SAVE_DIR=${PROJECT_DIR}/dpo_artifacts/eval/cifs_sft_8k_seed1
SCORES=${PROJECT_DIR}/dpo_artifacts/eval/scores_sft_8k_seed1.json
REF_MODEL=${PROJECT_DIR}/models/superconductor_generator

echo "=== sampling 8,000 from π_ref (seed=1) ==="
python scripts/dpo/sample_dpo_ckpt.py \
    --ref_model_path "${REF_MODEL}" \
    --dpo_ckpt none \
    --save_path "${SAVE_DIR}" \
    --band_gap 1.2218 \
    --guide_w 2.0 \
    --batch_size 100 \
    --num_batches_to_samples 80 \
    --seed 1 \
    --label "sft_8k_y10k_w2"

echo "=== scoring 8000 with CHGNet ==="
python scripts/dpo/score_ehull.py \
    --cif_dir "${SAVE_DIR}" \
    --out "${SCORES}" \
    --cache "${PROJECT_DIR}/dpo_artifacts/score_cache/chgnet_cache.json" \
    --batch_size 32

echo "=== compute-matched comparison ==="
python -c "
import json, numpy as np
from pathlib import Path

# Original 2000 baseline pool (defines threshold T)
b1 = json.loads(Path('dpo_artifacts/day1_baseline_pool/scores_y10k.json').read_text())
b1_e = np.array([r['energy_per_atom'] for r in b1 if r['valid']])
T = np.percentile(b1_e, 25)

# Fresh 8000 SFT pool
b2 = json.loads(Path('dpo_artifacts/eval/scores_sft_8k_seed1.json').read_text())
b2_e = np.array([r['energy_per_atom'] for r in b2 if r['valid']])

# DPO β=1 pool
d1 = json.loads(Path('dpo_artifacts/eval/scores_dpo_b1.json').read_text())
d1_e = np.array([r['energy_per_atom'] for r in d1 if r['valid']])

print(f'threshold T = baseline 25th-pct = {T:.3f} eV/atom')
print()
print(f'{\"pool\":<25} {\"n\":>5} {\"valid\":>6} {\"med E\":>8} {\"hits\":>5} {\"hit-rate\":>9}')
print('-' * 70)
for name, e in [
    ('SFT @ N=2000  (seed=0)', b1_e),
    ('SFT @ 4N=8000 (seed=1)', b2_e),
    ('DPO β=1 @ N=2000', d1_e),
]:
    hits = (e < T).sum()
    print(f'{name:<25} {len(e):>5} {len(e):>6} {np.median(e):>8.3f} {hits:>5} {hits/len(e):>9.3%}')

print()
print('--- compute-matched screening ---')
print('Question: how many SFT samples to get the same number of hits as DPO@β=1 (n=2000)?')
n_dpo = len(d1_e)
hits_dpo = int((d1_e < T).sum())
print(f'  DPO@β=1: {n_dpo} samples → {hits_dpo} hits at T={T:.3f}')
sft_yield = (b2_e < T).mean()
n_sft_needed = hits_dpo / sft_yield if sft_yield > 0 else float('inf')
print(f'  SFT@8000 hit-yield: {sft_yield:.3%} → need {n_sft_needed:.0f} samples for {hits_dpo} hits')
print(f'  DPO sample-compute saving vs SFT: {1 - n_dpo/n_sft_needed:.1%}')

print()
print('--- top-K quality at matched K ---')
# Take top-K=500 from each pool; compare median energy.
K = 500
print(f'Top-{K} median E/atom (lower = better):')
for name, e in [
    ('SFT @ N=2000', b1_e),
    ('SFT @ 4N=8000', b2_e),
    ('DPO β=1 @ N=2000', d1_e),
]:
    top = np.sort(e)[:K]
    print(f'  {name:<22}  median={np.median(top):.3f}  10th-pct={np.percentile(top, 10):.3f}')
"
echo "=== sft_8k done ==="
