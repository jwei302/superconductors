# DPO — Diffusion Direct Preference Optimization

Offline preference optimization of the GuidedMatDiffusion superconductor generator against
[CHGNet](https://github.com/CederGroupHub/chgnet) stability scores. The pipeline is
**sample → score → build pairs → train → sample → score**. The equivariant DiffCSP backbone is
frozen; only an adapter and the property-embedding pathway are updated.

## Layout

```
dpo/
  run_dpo.py            # training entry point (was scripts/dpo/dpo_train.py)
  module.py             # DPODiffusion LightningModule (three-term DPO logit)
  preference_dataset.py # PreferenceDataset + collate_pairs (winner/loser, shared t,m,noise)
  scripts/              # pair-building, sampling, scoring, figures, sbatch/sh helpers
artifacts/dpo/          # committed scores, metrics, figures, pairs_v1.pt
```

Shared base model and utilities stay in `diffcsp/` and `scripts/eval_utils.py`. See the repo
root [README.md](../README.md) for environment setup.

## Train one DPO checkpoint

```bash
python dpo/run_dpo.py \
    --pairs_file artifacts/dpo/pairs_v1.pt \
    --model_path models/superconductor_generator \
    --out_dir    artifacts/dpo/dpo_b1 \
    --beta 1 --max_steps 5000 --batch_size 1 --gpus 1 --seed 0
```

`pairs_v1.pt` is committed in this repo (~25 MB) so you can train without rebuilding pairs. Each
run takes ~3.5 min on an H200 and writes `dpo_b<beta>/dpo_final.ckpt` plus a per-step metrics CSV.

## Sample and score one DPO checkpoint

```bash
python dpo/scripts/sample_dpo_ckpt.py \
    --ref_model_path models/superconductor_generator \
    --dpo_ckpt   artifacts/dpo/dpo_b1/dpo_final.ckpt \
    --save_path  artifacts/dpo/eval/cifs_dpo_b1 \
    --band_gap 1.2218 --guide_w 2.0 \
    --batch_size 100 --num_batches_to_samples 20 --seed 0

python dpo/scripts/score_ehull.py \
    --cif_dir artifacts/dpo/eval/cifs_dpo_b1 \
    --out     artifacts/dpo/eval/scores_dpo_b1.json \
    --cache   artifacts/dpo/score_cache/chgnet_cache.json
```

To regenerate the headline figures and LaTeX table from the committed scores:
`python dpo/scripts/make_figures.py`.
