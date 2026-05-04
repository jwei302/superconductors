# GRPO Post-Training for GuidedMatDiffusion

This document describes the RL post-training extension added on top of GuidedMatDiffusion.

The goal is to replace most of the front-end filtering pipeline with GRPO fine-tuning so that the generator itself produces structures that are:

- closer to the requested `tcad` target,
- metallic-like under MEGNet,
- lower formation-energy under MEGNet,
- more stable under M3GNet,
- optionally better on offline `E_hull` / phonon proxy rewards in phase C.

## What Was Added

Core RL code:

- [diffcsp/run_rl.py](diffcsp/run_rl.py)
- [diffcsp/rl/prompts.py](diffcsp/rl/prompts.py)
- [diffcsp/rl/rewards.py](diffcsp/rl/rewards.py)
- [diffcsp/rl/trainer.py](diffcsp/rl/trainer.py)

Diffusion changes:

- [diffcsp/pl_modules/diffusion_w_type.py](diffcsp/pl_modules/diffusion_w_type.py)
- [diffcsp/pl_modules/diff_utils.py](diffcsp/pl_modules/diff_utils.py)

Configs:

- [conf/rl.yaml](conf/rl.yaml)
- [conf/grpo/default.yaml](conf/grpo/default.yaml)
- [conf/optim/rl.yaml](conf/optim/rl.yaml)

Utilities:

- [scripts/benchmark_reward_models.py](scripts/benchmark_reward_models.py)
- [scripts/train_reward_proxies.py](scripts/train_reward_proxies.py)
- [scripts/bee_net_smoketest.py](scripts/bee_net_smoketest.py)
- [scripts/megnet_smoketest.py](scripts/megnet_smoketest.py)

Environment hardening:

- [diffcsp/common/utils.py](diffcsp/common/utils.py)
- [pyproject.toml](pyproject.toml)

## Publishing Recommendation

Do not make a tiny repo containing only the new RL files.

The RL extension depends directly on the existing GuidedMatDiffusion codebase, config tree, dataset loaders, and diffusion model implementation. The cleanest publish path is:

1. Fork the full GuidedMatDiffusion repo or create a new branch from it.
2. Commit the RL code and docs into that fork/branch.
3. Keep model weights, datasets, and local outputs out of git.

If you create a separate GitHub repo, it should still contain the full GuidedMatDiffusion source tree plus the RL additions. Otherwise teammates will not be able to run the code without manually reconstructing the base project.

## What To Push

Push these code and config files:

- `README.md`
- `README_GRPO.md`
- `.gitignore`
- `pyproject.toml`
- `diffcsp/common/utils.py`
- `diffcsp/pl_modules/diff_utils.py`
- `diffcsp/pl_modules/diffusion_w_type.py`
- `diffcsp/rl/__init__.py`
- `diffcsp/rl/prompts.py`
- `diffcsp/rl/rewards.py`
- `diffcsp/rl/trainer.py`
- `diffcsp/run_rl.py`
- `conf/rl.yaml`
- `conf/grpo/default.yaml`
- `conf/optim/rl.yaml`
- `scripts/benchmark_reward_models.py`
- `scripts/train_reward_proxies.py`
- `scripts/bee_net_smoketest.py`
- `scripts/megnet_smoketest.py`

Do not push these local artifacts:

- `checkpoints/`
- `hf_models/`
- `data/`
- `out/`
- `wandb/`, `runs/`, `singlerun/`
- `elign.pdf`
- `guidedmatdiff.pdf`
- `visualization.ipynb`
- local `.env`

## External Dependencies

The reward code currently expects these repos to exist under `external/`:

- BEE-NET: `https://github.com/henniggroup/BEE-NET.git`
- MEGNet: `https://github.com/materialyzeai/megnet.git`

Recommended approach:

- do not commit `external/BEE-NET` and `external/megnet` as copied folders with nested `.git` directories,
- either add them properly as git submodules,
- or keep them out of git and ask teammates to clone them locally after cloning this repo.

Example local setup:

```bash
mkdir -p external
git clone https://github.com/henniggroup/BEE-NET.git external/BEE-NET
git clone https://github.com/materialyzeai/megnet.git external/megnet
```

## Required Assets That Stay Outside Git

Teammates also need these local assets:

- a GuidedMatDiffusion policy checkpoint directory, for example `hf_models/superconductor_generator/`
- BEE-Net CSO checkpoints under `checkpoints/bee_net/CSO/`
- the superconductivity dataset CSVs used by `conf/data/supccomb_12.yaml`

The RL code assumes the base checkpoint directory contains:

- a Lightning `.ckpt`
- `hparams.yaml`
- `prop_scaler.pt`

## Environment Setup

Clone the repo and enter it:

```bash
git clone <your-fork-or-branch-url>
cd GuidedMatDiffusion
```

Install the package:

```bash
pip install -e .
```

Install PyG binary dependencies for your PyTorch and CUDA build. Example for PyTorch 2.3.0:

```bash
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.3.0+${CUDA}.html
```

Install extra packages used by the RL utilities if they are not already present:

```bash
pip install pandas ase matgl
```

Create `.env` from `.env.template` and set:

```bash
PROJECT_ROOT=/absolute/path/to/GuidedMatDiffusion
HYDRA_JOBS=/absolute/path/for/hydra/outputs
WABDB_DIR=/absolute/path/for/wandb/outputs
```

Then export it:

```bash
source .env
```

## Known Runtime Risk

If imports fail with `torch_scatter` / `GLIBC` errors, the environment is wrong for the current host.

Typical fix:

- reinstall the PyG binaries for the exact host, PyTorch, and CUDA combination,
- or run in a container / node with a compatible GLIBC,
- or use a known-good training environment already used for GuidedMatDiffusion.

## Reward Model Smoke Tests

Test BEE-Net on one CIF:

```bash
python scripts/bee_net_smoketest.py \
  --cif /path/to/sample.cif \
  --ensemble-size 1 \
  --device cpu
```

Test MEGNet on one CIF:

```bash
python scripts/megnet_smoketest.py \
  --cif /path/to/sample.cif
```

Benchmark reward cost on a small batch of CIFs:

```bash
python scripts/benchmark_reward_models.py \
  --cif /path/to/a.cif \
  --cif /path/to/b.cif \
  --skip-m3g
```

Use this benchmark before large RL runs. BEE-Net and MEGNet should be validated first. M3GNet is more expensive and should be enabled only after a phase-A smoke test works.

## Base Supervised Fine-Tuning

If you do not already have the superconductivity checkpoint, first produce it with the original training path:

```bash
python diffcsp/run.py \
  data=supccomb_12 \
  model=diffusion_w_type \
  expname=<expname> \
  train.ckpt_path=<path_to_foundation_checkpoint>
```

Use the resulting checkpoint directory as the GRPO policy checkpoint.

## GRPO Training Overview

The GRPO trainer uses:

- grouped shared-prefix diffusion rollouts,
- BEE-Net dense shaping over the last denoising segment,
- terminal MEGNet rewards on every branch,
- terminal M3GNet rewards on a top-ranked subset,
- optional offline proxy rewards in phase C,
- evaluation against the frozen reference policy.

Default config entrypoint:

```bash
python -m diffcsp.run_rl
```

Main RL settings live in [conf/grpo/default.yaml](conf/grpo/default.yaml).

## Phase A: Smoke Test

Start with a very small run and disable expensive rewards:

```bash
python -m diffcsp.run_rl \
  paths.policy_checkpoint=/absolute/path/to/hf_models/superconductor_generator \
  grpo.schedule.max_updates=10 \
  grpo.eval.interval=5 \
  grpo.checkpoint.interval=5 \
  grpo.rewards.m3gnet.enabled=false \
  grpo.rewards.proxies.enabled=false
```

This verifies:

- checkpoint loading,
- rollout generation,
- BEE-Net shaping,
- MEGNet terminal rewards,
- GRPO optimizer step,
- evaluation JSON writing.

## Phase A: Real Run

Once the smoke test works, run a longer phase-A job:

```bash
python -m diffcsp.run_rl \
  paths.policy_checkpoint=/absolute/path/to/hf_models/superconductor_generator \
  grpo.schedule.max_updates=2000 \
  grpo.rewards.m3gnet.enabled=false \
  grpo.rewards.proxies.enabled=false
```

## Phase B: Add M3GNet Stability Reward

Enable the M3GNet subset reward after phase A is stable:

```bash
python -m diffcsp.run_rl \
  paths.policy_checkpoint=/absolute/path/to/hf_models/superconductor_generator \
  grpo.schedule.max_updates=5000 \
  grpo.rewards.m3gnet.enabled=true \
  grpo.rewards.m3gnet.top_fraction=0.25 \
  grpo.rewards.proxies.enabled=false
```

If cost is too high, reduce:

- `grpo.sampling.num_branches`
- `grpo.sampling.num_groups_per_update`
- `grpo.rewards.m3gnet.top_fraction`

## Phase C: Train and Use Proxy Rewards

Use this only after you have offline labels such as `E_hull` or phonon pass/fail.

Prepare a CSV manifest with at least:

- one column containing a CIF path, default column name `cif`
- one or more label columns, for example `ehull` or `phonon_pass`

Train proxies:

```bash
python scripts/train_reward_proxies.py \
  --manifest /path/to/proxy_manifest.csv \
  --cif-column cif \
  --out-dir /absolute/path/to/proxy_checkpoints \
  --regression-target ehull_proxy:ehull:minimize:1.0 \
  --binary-target phonon_proxy:phonon_pass:maximize:1.0
```

This writes checkpoint files like:

- `/absolute/path/to/proxy_checkpoints/ehull_proxy.pt`
- `/absolute/path/to/proxy_checkpoints/phonon_proxy.pt`

Then enable them in [conf/grpo/default.yaml](conf/grpo/default.yaml) by editing:

```yaml
rewards:
  proxies:
    enabled: true
    entries:
      - checkpoint: /absolute/path/to/proxy_checkpoints/ehull_proxy.pt
        weight: 1.0
      - checkpoint: /absolute/path/to/proxy_checkpoints/phonon_proxy.pt
        weight: 1.0
```

Then run phase C:

```bash
python -m diffcsp.run_rl \
  paths.policy_checkpoint=/absolute/path/to/hf_models/superconductor_generator \
  grpo.schedule.max_updates=10000 \
  grpo.rewards.m3gnet.enabled=true \
  grpo.rewards.proxies.enabled=true
```

## Outputs

Hydra writes outputs under `HYDRA_JOBS`.

The RL trainer writes:

- `rl_config.yaml`
- `reward_benchmark.json`
- `rl_update_*.pt`
- `eval_update_*.json`

Use the eval JSONs to compare:

- policy vs frozen reference,
- BEE target satisfaction,
- cheap-filter pass rate,
- invalid rate,
- duplicate rate,
- optional M3G and proxy reward summaries.

## Suggested GitHub Workflow

Recommended:

1. fork the full GuidedMatDiffusion repo,
2. create a branch such as `feature/grpo-rl`,
3. commit only code, config, and docs,
4. keep weights and datasets in local storage or release assets,
5. open a PR for teammates.

If teammates need exact external versions, the best long-term cleanup is to convert `external/BEE-NET` and `external/megnet` into git submodules. Until then, keep the README clone commands above.
