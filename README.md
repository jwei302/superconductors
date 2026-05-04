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

Online policy optimization against a composite BEE-NET and MEGNet reward, with optional M3GNet and proxy rewards for later-stage stability tuning. The pipeline is sample grouped rollouts -> score rewards -> normalize within each prompt group -> update the policy -> evaluate against the frozen reference model.

### Prepare reward dependencies

The GRPO reward stack expects BEE-NET and MEGNet under `external/`:

```bash
mkdir -p external
git clone https://github.com/henniggroup/BEE-NET.git external/BEE-NET
git clone https://github.com/materialyzeai/megnet.git external/megnet
```

The GRPO checkpoint path should point to a GuidedMatDiffusion policy directory containing a Lightning checkpoint, `hparams.yaml`, and `prop_scaler.pt`.

### Train one GRPO checkpoint

```bash
python -m diffcsp.run_rl \
    paths.policy_checkpoint=models/superconductor_generator \
    grpo.schedule.max_updates=2000 \
    grpo.rewards.m3gnet.enabled=false \
    grpo.rewards.proxies.enabled=false
```

This phase uses grouped diffusion rollouts, BEE-NET shaping, terminal MEGNet rewards, and evaluation against the frozen reference policy.

### Train with M3GNet or proxy rewards

```bash
python -m diffcsp.run_rl \
    paths.policy_checkpoint=models/superconductor_generator \
    grpo.schedule.max_updates=5000 \
    grpo.rewards.m3gnet.enabled=true \
    grpo.rewards.m3gnet.top_fraction=0.25 \
    grpo.rewards.proxies.enabled=false
```

For phase C, train proxy rewards from offline labels, then enable `grpo.rewards.proxies.enabled=true` in [conf/grpo/default.yaml](conf/grpo/default.yaml) or on the command line. RL checkpoints (`rl_update_*.pt`) and evaluation summaries (`eval_update_*.json`) are written under `HYDRA_JOBS`.

---

## Acknowledgments

- [GuidedMatDiffusion](https://github.com/paprakash/GuidedMatDiffusion) (Prakash et al., 2025) — base SFT model and conditioning pipeline
- [DiffCSP](https://github.com/jiaor17/DiffCSP) (Jiao et al., NeurIPS 2023) — equivariant graph diffusion backbone
- [Diffusion-DPO](https://arxiv.org/abs/2311.12908) (Wallace et al., CVPR 2024) — preference-optimization formulation
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) (Shao et al., 2024) — Group Relative Policy Optimization algorithmic reference
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (DeepSeek-AI, 2025) — large-scale GRPO-style reinforcement learning for reasoning models
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017) — policy-gradient foundation for clipped RL objectives
- [CHGNet](https://github.com/CederGroupHub/chgnet) (Deng et al., 2023) — universal pretrained energy model used for stability scoring
- [BEE-NET superconductor discovery workflow](https://www.nature.com/articles/s41524-026-01964-8) (Gibson et al., 2026) — Eliashberg spectral-function and `T_c` reward model
- [MEGNet](https://doi.org/10.1021/acs.chemmater.9b01294) (Chen et al., 2019) — graph-network models for crystal formation energy and electronic-property rewards
- [M3GNet](https://www.nature.com/articles/s43588-022-00349-3) (Chen and Ong, 2022) — universal graph interatomic potential used for stability-oriented reward signals
