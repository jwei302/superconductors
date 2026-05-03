#!/bin/bash
# Submit one eval job at a time on gpu_devel (1/user QoS), wait for each to
# finish before submitting the next. Then run the comparison summary.
set -u
PROJECT_DIR=/nfs/roberts/project/pi_jks79/jw2933/GuidedMatDiffusion
cd "${PROJECT_DIR}"

log() { echo "[$(date -Is)] $*"; }

devel_busy() {
    local n; n=$(squeue -u "$USER" -p gpu_devel -h 2>/dev/null | wc -l)
    [ "$n" -gt 0 ]
}

log "wait for any prior gpu_devel job to clear"
while devel_busy; do sleep 60; done
log "gpu_devel clear"

for BETA in 1 5 25; do
    log "β=${BETA}: submit eval"
    jobid=$(BETA="${BETA}" sbatch scripts/dpo/eval_dpo_one.sbatch | awk '{print $4}')
    log "β=${BETA}: jobid=${jobid}"

    while devel_busy; do sleep 60; done
    state=$(sacct -j "${jobid}" -X -h -o State 2>/dev/null | tr -d ' ' | head -1)
    log "β=${BETA}: ended (state=${state})"
done

log "===== running summary ====="
source .venv/bin/activate
python -c "
import json, numpy as np
from pathlib import Path

baseline = json.loads(Path('dpo_artifacts/day1_baseline_pool/scores_y10k.json').read_text())
b_e = np.array([r['energy_per_atom'] for r in baseline if r['valid']])
thresh = np.percentile(b_e, 25)
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
log "===== run_eval_sequential.sh DONE ====="
