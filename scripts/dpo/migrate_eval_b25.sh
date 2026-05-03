#!/bin/bash
# Once gpu_devel β=5 (10360513) finishes, check if h200 β=25 (10360592) has
# started running. If still PENDING, cancel it and resubmit to gpu_devel.
# If RUNNING/COMPLETED, leave alone.
set -u
PROJECT_DIR=/nfs/roberts/project/pi_jks79/jw2933/GuidedMatDiffusion
cd "${PROJECT_DIR}"

DEVEL_JOB=10360513   # β=5 on gpu_devel
H200_JOB=10360592    # β=25 on gpu_h200

log() { echo "[$(date -Is)] $*"; }

devel_busy() {
    local n; n=$(squeue -u "$USER" -p gpu_devel -h 2>/dev/null | wc -l)
    [ "$n" -gt 0 ]
}

job_state() {
    local s; s=$(squeue -j "$1" -h -o "%T" 2>/dev/null)
    if [ -z "$s" ]; then
        s=$(sacct -j "$1" -X -h -o State 2>/dev/null | tr -d ' ' | head -1)
    fi
    echo "${s:-UNKNOWN}"
}

log "wait for β=5 (${DEVEL_JOB}) on gpu_devel to finish"
while devel_busy; do sleep 60; done
log "β=5 finished; gpu_devel slot free"

state=$(job_state "${H200_JOB}")
log "β=25 (${H200_JOB}) on h200 state: ${state}"

if [ "$state" = "PENDING" ]; then
    log "still pending → cancelling h200 job and migrating to gpu_devel"
    scancel "${H200_JOB}" || true
    sleep 10
    while devel_busy; do sleep 60; done
    new_jobid=$(BETA=25 sbatch scripts/dpo/eval_dpo_one.sbatch | awk '{print $4}')
    log "submitted β=25 on gpu_devel: jobid=${new_jobid}"
    while devel_busy; do sleep 60; done
    final=$(job_state "${new_jobid}")
    log "β=25 gpu_devel job ${new_jobid} ended (state=${final})"
else
    log "β=25 already ${state} on h200 → leaving alone"
fi

log "===== running summary ====="
source .venv/bin/activate
python -c "
import json, numpy as np
from pathlib import Path

baseline = json.loads(Path('dpo_artifacts/day1_baseline_pool/scores_y10k.json').read_text())
b_e = np.array([r['energy_per_atom'] for r in baseline if r['valid']])
thresh = np.percentile(b_e, 25)
print(f'baseline: n={len(b_e)} median E/atom={np.median(b_e):.3f} 25th-pct={thresh:.3f}')
print(f'baseline hit-rate: {(b_e < thresh).mean():.3%} (= 25% by construction)')
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
log "===== migrate_eval_b25.sh DONE ====="
