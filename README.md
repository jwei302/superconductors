# Preference and Reward Optimization for Superconductor Crystal Diffusion

Post-training methods for [GuidedMatDiffusion](https://github.com/paprakash/GuidedMatDiffusion) — a property-conditioned crystal diffusion model built on [DiffCSP](https://github.com/jiaor17/DiffCSP) — with the goal of raising the hit rate of generated crystals through downstream physical screens at matched sampling compute.

This repository contains two complementary post-training projects:

1. **DPO** — offline Diffusion Direct Preference Optimization with [CHGNet](https://github.com/CederGroupHub/chgnet) stability scores.
2. **GRPO** — online Group Relative Policy Optimization with a composite BEE-NET and MEGNet reward.

Both methods freeze the equivariant DiffCSP backbone and update only an adapter and property-embedding pathway. The upstream pretrained model and its training pipeline are reused as-is.

---

## Setup

Install via [uv](https://docs.astral.sh/uv/):

```bash
uv sync                                  # creates .venv from pyproject.toml + uv.lock
source .venv/bin/activate

# torch_scatter / torch_sparse must match the locked torch 2.3.0 + CUDA build:
uv pip install torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

Copy `.env.template` → `.env` and set:
```
PROJECT_ROOT=<absolute path of this repo>
HYDRA_JOBS=<absolute path to save hydra outputs>
WANDB_DIR=<absolute path to save wandb outputs>
```

Pretrained SFT model:
```bash
mkdir -p models
huggingface-cli download paprakash/GuidedMatDiffusion_model \
    --local-dir models/superconductor_generator
```

---

## Project 1: DPO

Offline preference optimization against CHGNet stability scores. The pipeline is sample → score → build pairs → train → sample → score.

### Train one DPO checkpoint

```bash
python scripts/dpo/dpo_train.py \
    --pairs_file dpo_artifacts/pairs_v1.pt \
    --model_path models/superconductor_generator \
    --out_dir    dpo_artifacts/dpo_b1 \
    --beta 1 --max_steps 5000 --batch_size 1 --gpus 1 --seed 0
```

`pairs_v1.pt` is committed in this repo (~25 MB) so you can train without rebuilding pairs. Each run takes ~3.5 min on an H200 and writes `dpo_b<beta>/dpo_final.ckpt` plus per-step metrics CSV.

### Sample and score one DPO checkpoint

```bash
python scripts/dpo/sample_dpo_ckpt.py \
    --ref_model_path models/superconductor_generator \
    --dpo_ckpt   dpo_artifacts/dpo_b1/dpo_final.ckpt \
    --save_path  dpo_artifacts/eval/cifs_dpo_b1 \
    --band_gap 1.2218 --guide_w 2.0 \
    --batch_size 100 --num_batches_to_samples 20 --seed 0

python scripts/dpo/score_ehull.py \
    --cif_dir dpo_artifacts/eval/cifs_dpo_b1 \
    --out     dpo_artifacts/eval/scores_dpo_b1.json \
    --cache   dpo_artifacts/score_cache/chgnet_cache.json
```

To regenerate the headline figures and LaTeX table from the committed scores: `python scripts/dpo/make_figures.py`.

---

## Project 2: GRPO

Online policy optimization with grouped rollouts, BEE-NET, MEGNet, M3GNet, and optional phase-C proxy rewards.

The full RL post-training workflow is documented in [README_GRPO.md](README_GRPO.md), including environment setup, reward benchmarking, GRPO smoke tests, phase A/B/C training, and publishing guidance for teammates.

---

## Acknowledgments

- [GuidedMatDiffusion](https://github.com/paprakash/GuidedMatDiffusion) (Prakash et al., 2025) — base SFT model and conditioning pipeline
- [DiffCSP](https://github.com/jiaor17/DiffCSP) (Jiao et al., NeurIPS 2023) — equivariant graph diffusion backbone
- [Diffusion-DPO](https://arxiv.org/abs/2311.12908) (Wallace et al., CVPR 2024) — preference-optimization formulation
- [CHGNet](https://github.com/CederGroupHub/chgnet) (Deng et al., 2023) — universal pretrained energy model used for stability scoring
