#!/bin/bash
#SBATCH --job-name=grpo_bee
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=hydra_jobs/grpo_bee_%j.out
#SBATCH --error=hydra_jobs/grpo_bee_%j.err

# GRPO post-training of the superconductor generator with ONLY the BEE-NET
# (Allen-Dynes Tc) reward. MEGNet/M3GNet are disabled so no TensorFlow/matgl
# is required. Reward weights for meg/m3g are zeroed. The BEE reward uses the
# public BETE-NET CSO ensemble (lmax=1) as a substitute for the unreleased
# EMD checkpoints (configured as defaults in conf/grpo/default.yaml).
set -e
PROJECT_DIR=/nfs/roberts/scratch/pi_jks79/jw2933/GuidedMatDiffusion
cd "${PROJECT_DIR}"
set -a; source .env; set +a
source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "=== GRPO BEE-only on $(hostname) ==="
nvidia-smi -L || true

python grpo/run_grpo.py \
    expname=${EXPNAME:-grpo_beeonly} \
    optim.optimizer.lr=${LR:-5.0e-7} \
    paths.policy_checkpoint="${PROJECT_ROOT}/models/superconductor_generator" \
    grpo.rewards.megnet.enabled=false \
    grpo.rewards.m3gnet.enabled=false \
    grpo.loss.meg_weight=0 \
    grpo.loss.m3g_weight=0 \
    grpo.sampling.num_groups_per_update=2 \
    grpo.sampling.num_branches=6 \
    grpo.checkpoint.interval=25 \
    grpo.eval.interval=100 \
    grpo.eval.num_prompts=16 \
    grpo.eval.compare_reference=false \
    grpo.eval.run_m3g=false \
    "grpo.eval.guide_w_values=[2.0]"

echo "=== GRPO done ==="
