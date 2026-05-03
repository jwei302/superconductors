#!/bin/bash
# Once gpu_devel β=0.5 (10381774) finishes, check if h200 β=0.1 (10381775) has
# started. If still PENDING, cancel and resubmit to gpu_devel. Else leave alone.
set -u
PROJECT_DIR=/nfs/roberts/project/pi_jks79/jw2933/GuidedMatDiffusion
cd "${PROJECT_DIR}"

DEVEL_JOB=10381774
H200_JOB=10381775

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

log "wait for β=0.5 (${DEVEL_JOB}) on gpu_devel to finish"
while devel_busy; do sleep 60; done
log "β=0.5 finished; gpu_devel slot free"

state=$(job_state "${H200_JOB}")
log "β=0.1 (${H200_JOB}) on h200 state: ${state}"

if [ "$state" = "PENDING" ]; then
    log "still pending → cancelling h200 job and migrating to gpu_devel"
    scancel "${H200_JOB}" || true
    sleep 10
    while devel_busy; do sleep 60; done
    new_jobid=$(BETA=0.1 sbatch \
        --partition=gpu_devel \
        --time=01:00:00 \
        --job-name=dpo_dv_b0.1 \
        --output=dpo_artifacts/dpo_devel_b0.1_%j.out \
        --error=dpo_artifacts/dpo_devel_b0.1_%j.err \
        scripts/dpo/train_dpo.sbatch | awk '{print $4}')
    log "submitted β=0.1 on gpu_devel: jobid=${new_jobid}"
    while devel_busy; do sleep 60; done
    final=$(job_state "${new_jobid}")
    log "β=0.1 gpu_devel job ${new_jobid} ended (state=${final})"
else
    log "β=0.1 already ${state} on h200 → leaving alone"
fi

log "===== both done; checkpoints: ====="
ls -la dpo_artifacts/dpo_b0.5/dpo_final.ckpt dpo_artifacts/dpo_b0.1/dpo_final.ckpt 2>&1 || true
log "===== migrate_train_b0.1.sh DONE ====="
