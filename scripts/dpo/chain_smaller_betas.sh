#!/bin/bash
# Combined train+eval orchestration for the smaller-β extension:
#   1. wait for β=0.5 train (10381774) on gpu_devel to finish
#   2. eval β=0.5 on gpu_devel
#   3. handle β=0.1 train (10381775) — migrate from h200 to gpu_devel if still PENDING
#   4. eval β=0.1 on gpu_devel
#   5. print full β-sweep comparison table
set -u
PROJECT_DIR=/nfs/roberts/project/pi_jks79/jw2933/GuidedMatDiffusion
cd "${PROJECT_DIR}"

DEVEL_TRAIN_05=10381774   # β=0.5 train on gpu_devel
H200_TRAIN_01=10381775    # β=0.1 train on gpu_h200

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

wait_devel_clear() {
    while devel_busy; do sleep 60; done
}

submit_eval_devel() {
    local beta=$1
    local jobid
    jobid=$(BETA="${beta}" sbatch scripts/dpo/eval_dpo_one.sbatch | awk '{print $4}')
    log "β=${beta}: eval submitted on gpu_devel as ${jobid}"
    wait_devel_clear
    local final
    final=$(job_state "${jobid}")
    log "β=${beta}: eval ended (state=${final})"
}

# ============================================================
# Phase 1: wait for β=0.5 train on gpu_devel
# ============================================================
log "phase 1: wait for β=0.5 train (${DEVEL_TRAIN_05}) on gpu_devel"
wait_devel_clear
log "β=0.5 train ended (state=$(job_state ${DEVEL_TRAIN_05}))"
ls -la dpo_artifacts/dpo_b0.5/dpo_final.ckpt 2>&1 || log "WARN: β=0.5 ckpt missing"

# ============================================================
# Phase 2: β=0.5 eval on gpu_devel
# ============================================================
log "phase 2: submit β=0.5 eval"
if [ -f dpo_artifacts/dpo_b0.5/dpo_final.ckpt ]; then
    submit_eval_devel 0.5
else
    log "skip β=0.5 eval (no ckpt)"
fi

# ============================================================
# Phase 3: handle β=0.1 train (migrate from h200 if still pending)
# ============================================================
log "phase 3: handle β=0.1 train"
state=$(job_state "${H200_TRAIN_01}")
log "β=0.1 (${H200_TRAIN_01}) state on h200: ${state}"

if [ "$state" = "PENDING" ]; then
    log "still pending → scancel h200 + resubmit on gpu_devel"
    scancel "${H200_TRAIN_01}" || true
    sleep 10
    wait_devel_clear
    new_jobid=$(BETA=0.1 sbatch \
        --partition=gpu_devel \
        --time=01:00:00 \
        --job-name=dpo_dv_b0.1 \
        --output=dpo_artifacts/dpo_devel_b0.1_%j.out \
        --error=dpo_artifacts/dpo_devel_b0.1_%j.err \
        scripts/dpo/train_dpo.sbatch | awk '{print $4}')
    log "β=0.1: gpu_devel train jobid=${new_jobid}"
    wait_devel_clear
    log "β=0.1 train ended (state=$(job_state ${new_jobid}))"
elif [ "$state" = "RUNNING" ]; then
    log "β=0.1 RUNNING on h200 → wait for completion"
    while [ "$(job_state ${H200_TRAIN_01})" = "RUNNING" ] || [ "$(job_state ${H200_TRAIN_01})" = "PENDING" ]; do
        sleep 60
    done
    log "β=0.1 h200 train ended (state=$(job_state ${H200_TRAIN_01}))"
else
    log "β=0.1 state=${state} → assume done; check ckpt"
fi

ls -la dpo_artifacts/dpo_b0.1/dpo_final.ckpt 2>&1 || log "WARN: β=0.1 ckpt missing"

# ============================================================
# Phase 4: β=0.1 eval on gpu_devel
# ============================================================
log "phase 4: submit β=0.1 eval"
if [ -f dpo_artifacts/dpo_b0.1/dpo_final.ckpt ]; then
    wait_devel_clear
    submit_eval_devel 0.1
else
    log "skip β=0.1 eval (no ckpt)"
fi

# ============================================================
# Phase 5: full β-sweep table
# ============================================================
log "phase 5: full β-sweep comparison"
source .venv/bin/activate
python -c "
import json, numpy as np
from pathlib import Path
from scipy import stats

baseline = json.loads(Path('dpo_artifacts/day1_baseline_pool/scores_y10k.json').read_text())
b_e = np.array([r['energy_per_atom'] for r in baseline if r['valid']])
T_25 = np.percentile(b_e, 25)
T_10 = np.percentile(b_e, 10)
T_05 = np.percentile(b_e, 5)
print(f'thresholds: 25th={T_25:.3f}  10th={T_10:.3f}  5th={T_05:.3f} eV/atom')
print()
print(f'{\"pool\":<12} {\"valid\":>6} {\"med\":>8} {\"mean\":>8} {\"hr@25%\":>7} {\"hr@10%\":>7} {\"hr@5%\":>7} {\"top500\":>8} {\"KS\":>5}')
print('-'*90)
def show(name, e):
    hr25 = (e < T_25).mean(); hr10 = (e < T_10).mean(); hr05 = (e < T_05).mean()
    top500 = np.median(np.sort(e)[:500])
    ks, _ = stats.ks_2samp(e, b_e, alternative='greater')
    print(f'{name:<12} {len(e):>6} {np.median(e):>8.3f} {np.mean(e):>8.3f} {hr25:>6.2%} {hr10:>6.2%} {hr05:>6.2%} {top500:>8.3f} {ks:>5.3f}')
show('baseline', b_e)
for beta in ['0.1','0.5','1','5','25']:
    p = Path(f'dpo_artifacts/eval/scores_dpo_b{beta}.json')
    if not p.exists():
        print(f'dpo_b{beta:<8} (skipped — no scores)')
        continue
    e = np.array([r['energy_per_atom'] for r in json.loads(p.read_text()) if r['valid']])
    show(f'dpo_b{beta}', e)
"
log "===== chain_smaller_betas.sh DONE ====="
