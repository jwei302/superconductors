"""Train Diffusion-DPO on a pre-built preference pair file.

Loads π_ref from a Hydra run dir, wraps it in DPODiffusion (adapter-only training),
and runs a Lightning trainer over PreferenceDataset → DataLoader.

Validation signal during training: every --val_every steps, log mean implicit-reward
gap and σ(z) on a fixed held-out pair-batch — see `--val_holdout_frac`.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_utils import load_model   # noqa: E402
from diffcsp.pl_modules.dpo_module import DPODiffusion   # noqa: E402
from diffcsp.pl_data.preference_dataset import PreferenceDataset, collate_pairs   # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_file", required=True, type=Path)
    ap.add_argument("--model_path", required=True, type=Path, help="π_ref Hydra run dir")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--beta", type=float, default=200.0)
    ap.add_argument("--total_timesteps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=1, help="See preference_dataset.py — entries must share t,m within a batch.")
    ap.add_argument("--val_holdout_frac", type=float, default=0.1)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--normalize_components", action="store_true")
    ap.add_argument("--gpus", type=int, default=1 if torch.cuda.is_available() else 0)
    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load π_ref
    print(f"[load] π_ref ← {args.model_path}")
    sft, _, _ = load_model(args.model_path.resolve(), load_data=False)

    # 2. Build DPODiffusion
    dpo = DPODiffusion(
        policy=sft, beta=args.beta, total_timesteps=args.total_timesteps,
        normalize_components=args.normalize_components, train_only_adapters=True, lr=args.lr,
    )

    # 3. Pair dataset + train/val split
    full = PreferenceDataset(args.pairs_file)
    n = len(full)
    n_val = max(1, int(n * args.val_holdout_frac))
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ds = Subset(full, train_idx)
    val_ds = Subset(full, val_idx)
    print(f"[data] total {n} entries → train {len(train_ds)} / val {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_pairs, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_pairs, num_workers=0)

    # 4. Trainer
    logger = CSVLogger(save_dir=str(args.out_dir), name="dpo")
    ckpt_cb = ModelCheckpoint(dirpath=str(args.out_dir), filename="dpo-{step:05d}",
                              every_n_train_steps=args.val_every, save_last=True, save_top_k=-1)
    trainer = Trainer(
        default_root_dir=str(args.out_dir),
        max_steps=args.max_steps,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        logger=logger,
        callbacks=[ckpt_cb, LearningRateMonitor()],
        check_val_every_n_epoch=None,
        val_check_interval=args.val_every,
        log_every_n_steps=10,
    )

    print(f"[train] β={args.beta} T={args.total_timesteps} lr={args.lr} max_steps={args.max_steps}")
    trainer.fit(dpo, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint(str(args.out_dir / "dpo_final.ckpt"))
    print(f"[done] checkpoint saved → {args.out_dir / 'dpo_final.ckpt'}")


if __name__ == "__main__":
    main()
