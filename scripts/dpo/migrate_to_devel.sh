#!/bin/bash
# Once user's gpu_devel queue is clear, sequentially run any *pending* h200 β-sweep
# jobs on gpu_devel (which has 1/user QoS so we go one at a time).
# h200 jobs that are already RUNNING/COMPLETED are left alone.

set -u
PROJECT_DIR=/nfs/roberts/project/pi_jks79/jw2933/GuidedMatDiffusion
H200_JOBS=("10306811:1" "10306812:5" "10306813:25")   # h200_jobid:beta

cd "${PROJECT_DIR}"

log() { echo "[$(date -Is)] $*"; }

is_pending() {
    local state; state=$(squeue -j "$1" -h -o "%T" 2>/dev/null)
    [ "$state" = "PENDING" ]
}

devel_busy() {
    local n; n=$(squeue -u "$USER" -p gpu_devel -h 2>/dev/null | wc -l)
    [ "$n" -gt 0 ]
}

submit_devel() {
    local beta=$1
    local sb=/tmp/dpo_devel_b${beta}.sbatch
    cat > "$sb" <<EOF
#!/bin/bash
#SBATCH --job-name=dpo_dv_b${beta}
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=${PROJECT_DIR}/dpo_artifacts/dpo_devel_b${beta}_%j.out
#SBATCH --error=${PROJECT_DIR}/dpo_artifacts/dpo_devel_b${beta}_%j.err

set -e
cd ${PROJECT_DIR}
source .venv/bin/activate

python scripts/dpo/dpo_train.py \\
    --pairs_file ${PROJECT_DIR}/dpo_artifacts/pairs_v1.pt \\
    --model_path ${PROJECT_DIR}/models/superconductor_generator \\
    --out_dir ${PROJECT_DIR}/dpo_artifacts/dpo_b${beta} \\
    --beta ${beta} \\
    --max_steps 5000 \\
    --val_every 500 \\
    --batch_size 1 \\
    --lr 1e-4 \\
    --gpus 1 \\
    --seed 0
EOF
    sbatch "$sb" | awk '{print $4}'
}

log "wait for user's gpu_devel queue to clear"
while devel_busy; do
    sleep 60
done
log "gpu_devel queue is clear"

for pair in "${H200_JOBS[@]}"; do
    h200_jobid="${pair%:*}"
    beta="${pair#*:}"

    state=$(squeue -j "$h200_jobid" -h -o "%T" 2>/dev/null)
    state="${state:-DONE}"
    if ! is_pending "$h200_jobid"; then
        log "β=${beta}: h200 job ${h200_jobid} is ${state} — leaving alone"
        continue
    fi

    log "β=${beta}: cancelling h200 ${h200_jobid} and migrating to gpu_devel..."
    scancel "$h200_jobid" || true
    sleep 10

    # Make sure my own previous gpu_devel run (if any) is gone
    while devel_busy; do
        sleep 60
    done

    new_jobid=$(submit_devel "$beta")
    log "β=${beta}: gpu_devel job ${new_jobid} submitted"

    # Wait for it to finish (gpu_devel = 1/user, so 'queue clear' is equivalent to 'job done')
    while devel_busy; do
        sleep 60
    done
    state=$(sacct -j "$new_jobid" -X -h -o State 2>/dev/null | tr -d ' ' | head -1)
    log "β=${beta}: gpu_devel job ${new_jobid} ended (state=${state})"
done

log "===== final state ====="
sacct -j 10306811,10306812,10306813 --format=JobID,JobName,State,Elapsed,NodeList -X 2>&1 || true
ls -la dpo_artifacts/dpo_b*/dpo_final.ckpt 2>&1 || true
log "===== migrate_to_devel.sh DONE ====="
