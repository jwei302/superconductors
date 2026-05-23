# Preference and Reward Optimization for Superconductor Crystal Diffusion

Post-training methods for [GuidedMatDiffusion](https://github.com/paprakash/GuidedMatDiffusion) — a property-conditioned crystal diffusion model built on [DiffCSP](https://github.com/jiaor17/DiffCSP) — with the goal of raising the hit rate of generated crystals through downstream physical screens at matched sampling compute.

This repository contains three complementary post-training projects, each in its own self-contained folder:

| Project | Folder | Method |
|---|---|---|
| **DPO** | [`dpo/`](dpo/README.md) | Offline Diffusion Direct Preference Optimization with [CHGNet](https://github.com/CederGroupHub/chgnet) stability scores |
| **GRPO** | [`grpo/`](grpo/README.md) | Online Group Relative Policy Optimization with a composite BEE-NET + MEGNet reward |
| **VPO** | [`vpo/`](vpo/README.md) | Online Vector (multi-objective) Policy Optimization over a BEE-NET + CHGNet + M3GNet reward *vector* ([Bahlous-Boldi et al., 2026](https://arxiv.org/abs/2605.22817)) |

Both methods freeze the equivariant DiffCSP backbone and update only an adapter and property-embedding pathway. The upstream pretrained model and its training pipeline are reused as-is.

## Repository layout

```
diffcsp/            # shared base: DiffCSP diffusion model, GNN decoder, data, common utils, run.py
conf/               # shared Hydra config groups (data, model, optim, train, logging)
scripts/            # shared base scripts (evaluate, sample, generation, optimization, eval_utils)
dpo/                # DPO method: run_dpo.py, module.py, preference_dataset.py, scripts/, README
grpo/               # GRPO method: run_grpo.py, trainer/rewards/prompts, conf/, scripts/, README
artifacts/
  dpo/              # committed DPO results (scores, metrics, figures, pairs_v1.pt)
  grpo/             # GRPO results (currently empty; runs write to HYDRA_JOBS)
external/           # BEE-NET and MEGNet reward models (git submodules; GRPO only)
```

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

Clone the GRPO reward submodules (only needed for GRPO):

```bash
git submodule update --init --recursive
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

## Projects

- **DPO** — offline preference optimization against CHGNet stability scores (sample → score → build pairs → train → sample → score). See [`dpo/README.md`](dpo/README.md).
- **GRPO** — online policy optimization against a composite BEE-NET + MEGNet reward, with optional M3GNet and proxy rewards (sample grouped rollouts → score → normalize within group → update → evaluate vs. frozen reference). See [`grpo/README.md`](grpo/README.md).

---

## Acknowledgments

- [GuidedMatDiffusion](https://github.com/paprakash/GuidedMatDiffusion) (Prakash et al., 2025) — base SFT model and conditioning pipeline
- [DiffCSP](https://github.com/jiaor17/DiffCSP) (Jiao et al., NeurIPS 2023) — equivariant graph diffusion backbone
- [CHGNet](https://github.com/CederGroupHub/chgnet) (Deng et al., 2023) — universal pretrained energy model used for stability scoring
- [BEE-NET superconductor discovery workflow](https://www.nature.com/articles/s41524-026-01964-8) (Gibson et al., 2026) — Eliashberg spectral-function and `T_c` reward model
- [MEGNet](https://doi.org/10.1021/acs.chemmater.9b01294) (Chen et al., 2019) — graph-network models for crystal formation energy and electronic-property rewards
- [M3GNet](https://www.nature.com/articles/s43588-022-00349-3) (Chen and Ong, 2022) — universal graph interatomic potential used for stability-oriented reward signals
```

