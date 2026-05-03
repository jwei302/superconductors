# Diffusion-DPO for Guided Crystal Generation

A short-cycle implementation of [Diffusion-DPO (Wallace et al., CVPR 2024)](https://arxiv.org/abs/2311.12908) on top of [GuidedMatDiffusion](https://github.com/paprakash/GuidedMatDiffusion) — a property-conditioned crystal diffusion model built on [DiffCSP](https://github.com/jiaor17/DiffCSP). The objective is to use offline preference optimization, with stability scored by [CHGNet](https://github.com/CederGroupHub/chgnet), to improve the crystal pool returned by the SFT model at matched sampling compute.

This repo is a fork/extension of GuidedMatDiffusion. The upstream pretrained model and its training pipeline are reused as-is; only the DPO module, preference dataset, and evaluation tooling are new.

---

## Headline result

Five-point β sweep, 2000 samples per pool at scaled `y = 10 K` and CFG weight `w = 2.0`, scored by CHGNet single-point energy. Hit-rate is the fraction of valid samples whose CHGNet `E/atom` falls below the SFT pool's 25th-percentile energy threshold.

| pool      | n    | median `E/atom` | hit-rate@25th | top-500 median | #unique chemsys |
|-----------|------|-----------------|---------------|-----------------|-----------------|
| baseline (SFT) | 1999 | −9.85 | 25.0% | −10.96 | 826 |
| DPO β=25  | 1998 | −9.93 | 27.1% | −10.95 | 830 |
| DPO β=5   | 1998 | −9.95 | 27.7% | −11.01 | 833 |
| **DPO β=1** (sweet spot) | **1998** | **−9.99** | **30.2%** | **−11.08** | **797** |
| DPO β=0.5 | 1996 | −10.11 | 33.7% | −11.21 | 791 |
| DPO β=0.1 † | 1999 | −11.83 | 90.2% | −12.40 | **158** |

† **β=0.1 is reward-hacked.** Structure-uniqueness stays at 100%, but the chemical-system distribution collapses by 5× (158 vs ~830 chemsys; 67% of samples in the top-10 chemsys, all Ta- and Re-rich 5d transition metals which have intrinsically low CHGNet `E/atom`). The metric improvement does not correspond to genuine stability gain. β=1 is the honest sweet spot.

A best-of-N comparison (sampling 8000 from SFT and screening) shows DPO @ N=2000 is **17.7% more sample-compute efficient** than SFT screening at matched hit yield, but SFT @ 4× compute beats DPO @ 1× compute on top-K screening quality — a fair comparison would be DPO @ 4× vs SFT @ 4×, which is left as future work.

See `dpo_artifacts/figures/` for the full figure set including the β-sweep, energy-distribution shift, chemistry-collapse audit, training-time component shares, and σ(z) trajectories.

---

## Repository layout

```
GuidedMatDiffusion/                 (fork of paprakash/GuidedMatDiffusion)
├── diffcsp/                        upstream model code (unchanged)
│   ├── pl_modules/
│   │   ├── diffusion_w_type.py     property-conditioned CSPDiffusion (π_ref)
│   │   ├── cspnet.py               equivariant GNN decoder
│   │   └── dpo_module.py           [NEW] DPODiffusion LightningModule
│   └── pl_data/
│       └── preference_dataset.py   [NEW] paired-batch loader for DPO
│
├── scripts/
│   ├── generation.py               upstream sample loop
│   ├── eval_utils.py               upstream model loading
│   └── dpo/                        [NEW] DPO entry points
│       ├── build_preference_pairs.py     score → pair → cache π_ref errors
│       ├── score_ehull.py                CHGNet single-point E/atom + structure hash
│       ├── dpo_train.py                  Lightning trainer entry point
│       ├── sample_dpo_ckpt.py            sample from a DPO checkpoint (or fall back to π_ref)
│       ├── compute_ehull.py              retroactive E_above_hull via Materials Project hull
│       ├── audit_pools.py                chemistry / element / uniqueness audit
│       ├── make_figures.py               publication figures + LaTeX table
│       ├── test_dpo_loss.py              loss correctness + log-2 sanity test
│       ├── test_pairs_pipeline.py        end-to-end synthetic pipeline test
│       └── *.sbatch, *.sh                SLURM job specs (Yale Bouchet)
│
├── conf/                           Hydra configs for the SFT model (unchanged)
│
└── dpo_artifacts/                  evaluation outputs (committed: scores + figures + metrics)
    ├── pairs_v1.pt                 cached pair set (1k pairs × 4 t-draws)
    ├── day1_baseline_pool/
    │   └── scores_y10k.json        2000 SFT samples scored by CHGNet
    ├── eval/
    │   └── scores_*.json           DPO and SFT-best-of-N pool scores
    ├── dpo_b{0.1,0.5,1,5,25}/dpo/version_0/
    │   └── metrics.csv             per-step training metrics
    ├── audit_summary.json          chemistry/uniqueness audit results
    ├── score_cache/                CHGNet structure-hash → energy cache
    └── figures/                    publication figures (PDF + PNG) + table1.tex
```

CIF directories, training checkpoints, and the SFT model weights are gitignored — they are large and are reproducible from the commands below.

---

## Setup

### Environment

The project uses the same env as GuidedMatDiffusion plus CHGNet for scoring and `mp-api` for the optional E_above_hull check. Install via [uv](https://docs.astral.sh/uv/):

```bash
uv sync                                  # creates .venv/ from pyproject.toml + uv.lock
source .venv/bin/activate

# torch_scatter / torch_sparse must match the installed torch + CUDA build.
# For torch 2.3.0 + CUDA 12.1 (the locked stack):
uv pip install torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

`torch-scatter` and `torch-sparse` wheels are CUDA-version-specific. Substitute `cpu` or `cu118` if needed (Linux/Windows only; macOS is CPU-only).

Copy `.env.template` → `.env` and set:
```
PROJECT_ROOT=<absolute path of this repo>
HYDRA_JOBS=<absolute path to save hydra outputs>
WANDB_DIR=<absolute path to save wandb outputs>
```

### Pretrained SFT model

```bash
mkdir -p models
huggingface-cli download paprakash/GuidedMatDiffusion_model \
    --local-dir models/superconductor_generator
```

The Hydra run dir layout (`hparams.yaml`, `prop_scaler.pt`, `lattice_scaler.pt`, `*.ckpt`) is what `eval_utils.load_model` consumes.

### CHGNet

`chgnet>=0.3.0` is in `pyproject.toml` and resolves automatically. No download needed; the model loads on first use.

### (Optional) Materials Project API key for E_hull

```bash
export MP_API_KEY=<your key from https://next-gen.materialsproject.org/api>
```

---

## Reproducing the results

All commands assume `cd <repo root> && source .venv/bin/activate`. The `scripts/dpo/*.sbatch` files are SLURM templates targeted at Yale Bouchet (gpu_h200 / gpu_devel partitions); on other systems, run the inner `python ...` invocations directly. Each phase produces a self-contained artifact in `dpo_artifacts/` so you can pick up at any step.

All results in this repo were produced with `seed=0` for sampling and `seed=0` for training. Runs are deterministic up to GPU-kernel non-determinism.

### Phase 1 — Baseline pool (CHGNet single-point on 2000 SFT samples)

```bash
# Sample 2000 from π_ref at scaled y=1.2218 (≈ Tc=10K), CFG w=2.0
python scripts/generation.py \
    --model_path models/superconductor_generator \
    --dataset supccomb_12 \
    --save_path dpo_artifacts/day1_baseline_pool/cifs_y10k \
    --band_gap 1.2218 --guide_w 2.0 \
    --batch_size 100 --num_batch_to_sample 20 \
    --seed 0

# Score with CHGNet single-point, cache by canonical structure hash
python scripts/dpo/score_ehull.py \
    --cif_dir dpo_artifacts/day1_baseline_pool/cifs_y10k \
    --out dpo_artifacts/day1_baseline_pool/scores_y10k.json \
    --cache dpo_artifacts/score_cache/chgnet_cache.json
```

The `scores_y10k.json` defines the 25th-percentile threshold used downstream as the hit-rate cutoff.

### Phase 2 — Preference pairs

For each of 1000 winner/loser pairs (top-25% / bottom-25% by `E/atom` with a reward-gap filter), 4 timestep draws are taken and π_ref's three-component denoising error is cached on disk so DPO training does not need π_ref in GPU memory.

```bash
python scripts/dpo/build_preference_pairs.py \
    --score_file dpo_artifacts/day1_baseline_pool/scores_y10k.json \
    --cif_dir dpo_artifacts/day1_baseline_pool/cifs_y10k \
    --model_path models/superconductor_generator \
    --out_path dpo_artifacts/pairs_v1.pt \
    --top_frac 0.25 --bottom_frac 0.25 --gap_margin 0.5 \
    --k_draws 4 --max_pairs 1000 --y 1.2218
```

Output: `pairs_v1.pt` (~25 MB) — included in this repo for direct reuse.

**Sanity check** (CPU; runs in seconds):
```bash
python scripts/dpo/test_dpo_loss.py
```
Verifies `loss = log 2 ± 1e-9` at θ = θ_ref, forward antisymmetry of the logit, and 50-step convergence on a synthetic batch.

### Phase 3 — Train DPO

```bash
for BETA in 0.1 0.5 1 5 25; do
    python scripts/dpo/dpo_train.py \
        --pairs_file dpo_artifacts/pairs_v1.pt \
        --model_path models/superconductor_generator \
        --out_dir dpo_artifacts/dpo_b${BETA} \
        --beta ${BETA} \
        --max_steps 5000 --val_every 500 \
        --batch_size 1 --lr 1e-4 --gpus 1 --seed 0
done
```

Each run takes ~3.5 min on an H200. Outputs `dpo_b<beta>/dpo_final.ckpt` and per-step metrics CSV at `dpo_b<beta>/dpo/version_0/metrics.csv` (committed for the five β values used here).

### Phase 4 — Evaluate

For each DPO checkpoint, sample 2000 crystals at the same conditioning + seed as the baseline pool, then score:

```bash
for BETA in 0.1 0.5 1 5 25; do
    python scripts/dpo/sample_dpo_ckpt.py \
        --ref_model_path models/superconductor_generator \
        --dpo_ckpt dpo_artifacts/dpo_b${BETA}/dpo_final.ckpt \
        --save_path dpo_artifacts/eval/cifs_dpo_b${BETA} \
        --band_gap 1.2218 --guide_w 2.0 \
        --batch_size 100 --num_batches_to_samples 20 --seed 0

    python scripts/dpo/score_ehull.py \
        --cif_dir dpo_artifacts/eval/cifs_dpo_b${BETA} \
        --out dpo_artifacts/eval/scores_dpo_b${BETA}.json \
        --cache dpo_artifacts/score_cache/chgnet_cache.json
done
```

### Phase 5 — Best-of-N control

A 4× sample budget from SFT, used to verify that the DPO improvement is not just a screening artifact:

```bash
python scripts/dpo/sample_dpo_ckpt.py \
    --ref_model_path models/superconductor_generator \
    --dpo_ckpt none \
    --save_path dpo_artifacts/eval/cifs_sft_8k_seed1 \
    --band_gap 1.2218 --guide_w 2.0 \
    --batch_size 100 --num_batches_to_samples 80 --seed 1

python scripts/dpo/score_ehull.py \
    --cif_dir dpo_artifacts/eval/cifs_sft_8k_seed1 \
    --out dpo_artifacts/eval/scores_sft_8k_seed1.json \
    --cache dpo_artifacts/score_cache/chgnet_cache.json
```

### Phase 6 — Audit (chemistry collapse / mode-collapse check)

```bash
python scripts/dpo/audit_pools.py \
    --out dpo_artifacts/audit_summary.json --n_workers 8
```

Reports `# unique chemsys`, `top-10 chemsys share`, and structure-hash uniqueness for every pool. This is what flags β=0.1 as reward-hacked.

### Phase 7 — Figures + table

```bash
python scripts/dpo/make_figures.py
```

Writes six figures (PDF + PNG) and a LaTeX table to `dpo_artifacts/figures/`.

### Phase 8 — Optional: retroactive E_above_hull

Requires `MP_API_KEY` in env. Re-scores all valid structures against the Materials Project convex hull and prints hit-rate at thermodynamic thresholds (50/100/200/500 meV/atom). This converts the headline metric from "empirical baseline-quantile shift" to a physical stability claim, modulo the ~30-50 meV/atom CHGNet vs DFT calibration offset.

```bash
for POOL in dpo_b0.1 dpo_b0.5 dpo_b1 dpo_b5 dpo_b25; do
    python scripts/dpo/compute_ehull.py \
        --scores dpo_artifacts/eval/scores_${POOL}.json \
        --cif_dir dpo_artifacts/eval/cifs_${POOL} \
        --out dpo_artifacts/eval/scores_${POOL}_ehull.json
done
```

---

## Method (one paragraph)

Three-term DPO logit per pair `(x^w, x^l)` with shared `(t, ε_L, rand_x, ε_A, m)`:

```
Δ_φ(x) = w_L · ‖ε_L − ε^L_φ(x_t,t,y,m)‖²
       + w_F · ‖d_log_p_wn(σ·rand_x,σ)/√σ²_norm − s^F_φ(x_t,t,y,m)‖²
       + w_A · ‖ε_A − ε^A_φ(x_t,t,y,m)‖²

L_DPO = −E [ log σ( −β · T · [(Δ_θ(x^w) − Δ_ref(x^w)) − (Δ_θ(x^l) − Δ_ref(x^l))] ) ]
```

with `(w_L, w_F, w_A) = (1, 1, 20)` matched to the SFT loss weights in `conf/model/diffusion_w_type.yaml`, `T = 1000`, and `m = property_indicator` Bernoulli(`1 − p_uncond`) for CFG dropout. The CSPDiffusion `decoder` is reused unmodified; only the `AdapterModule` parameters and `property_embedding` are trained (~14% of total params). π_ref's regression errors are cached at pair-construction time, so training never loads π_ref into GPU memory.

---

## Acknowledgments

- [GuidedMatDiffusion](https://github.com/paprakash/GuidedMatDiffusion) (Prakash et al., 2025) — base SFT model and conditioning pipeline
- [DiffCSP](https://github.com/jiaor17/DiffCSP) (Jiao et al., NeurIPS 2023) — equivariant graph diffusion backbone
- [Diffusion-DPO](https://arxiv.org/abs/2311.12908) (Wallace et al., CVPR 2024) — preference-optimization formulation
- [CHGNet](https://github.com/CederGroupHub/chgnet) (Deng et al., 2023) — universal pretrained energy model used for stability scoring
- [Materials Project](https://next-gen.materialsproject.org/) — convex-hull entries for the optional E_above_hull computation

---

## Citation

```bibtex
@misc{prakash2025guidediff,
    title={Guided Diffusion for the Discovery of New Superconductors},
    author={Pawan Prakash and Jason B. Gibson and Zhongwei Li and Gabriele Di Gianluca and Juan Esquivel and Eric Fuemmeler and Benjamin Geisler and Jung Soo Kim and Adrian Roitberg and Ellad B. Tadmor and Mingjie Liu and Stefano Martiniani and Gregory R. Stewart and James J. Hamlin and Peter J. Hirschfeld and Richard G. Hennig},
    year={2025},
    eprint={2509.25186},
    archivePrefix={arXiv},
    primaryClass={cond-mat.supr-con},
    url={https://arxiv.org/abs/2509.25186}
}

@inproceedings{wallace2024diffusiondpo,
    title={Diffusion Model Alignment Using Direct Preference Optimization},
    author={Bram Wallace and Meihua Dang and Rafael Rafailov and Linqi Zhou and Aaron Lou and Senthil Purushwalkam and Stefano Ermon and Caiming Xiong and Shafiq Joty and Nikhil Naik},
    booktitle={CVPR},
    year={2024}
}
```
