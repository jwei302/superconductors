"""Preference dataset for offline Diffusion-DPO.

A pairs file (`pairs_v1.pt`) holds a list of dicts, one per (winner, loser, noise-draw)
sample. Each entry has:
  - winner: PyG Data with lengths/angles/frac_coords/atom_types/num_atoms/num_nodes/y
  - loser:  PyG Data, ditto
  - t:      int timestep in [1, T]
  - m:      0 or 1 (CFG property_indicator)
  - rand_l_w/rand_x_w/rand_t_w: cached noise tensors for winner
  - rand_l_l/rand_x_l/rand_t_l: cached noise tensors for loser
  - ref_loss_w: tensor [3] = (mse_lattice, mse_coord, mse_type) under π_ref
  - ref_loss_l: tensor [3] same for loser
  - reward_gap: float, the explicit-reward gap that produced the pair (for diagnostics)

Each entry is fully self-contained — no need to recompute π_ref forward at training time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data


class PreferenceDataset(Dataset):
    def __init__(self, pairs_file: str | Path):
        self.entries: list[dict[str, Any]] = torch.load(pairs_file, map_location="cpu", weights_only=False)
        if not isinstance(self.entries, list):
            raise ValueError(f"Expected list of pair dicts in {pairs_file}, got {type(self.entries)}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.entries[idx]


def collate_pairs(entries: list[dict[str, Any]]) -> Any:
    """Collate a list of pair-entries into a single batch object usable by `DPODiffusion.dpo_step`.

    PyG handles the variable-N atom batching for `winner` / `loser` graphs. Noise tensors
    follow the graph batching (concat along atom dim for rand_x and rand_t). Per-pair
    scalars (t, m, ref_loss_*) are stacked.

    Note: this requires that all entries in the batch share the same `t` and `m` value
    (because the policy decoder evaluates a single timestep per forward pass). The
    pair-builder assigns t/m per entry independently, so to use a batch_size > 1 the
    entries should be grouped by (t, m). For Day 2 simplicity we just take batch_size=1
    in practice — this collate_fn supports larger batches when grouping is done.
    """
    winners = Batch.from_data_list([e["winner"] for e in entries])
    losers = Batch.from_data_list([e["loser"] for e in entries])

    # Concatenate noise along their natural dims.
    rand_l_w = torch.stack([e["rand_l_w"] for e in entries], dim=0)   # [B, 3, 3]
    rand_l_l = torch.stack([e["rand_l_l"] for e in entries], dim=0)
    rand_x_w = torch.cat([e["rand_x_w"] for e in entries], dim=0)     # [N_total, 3]
    rand_x_l = torch.cat([e["rand_x_l"] for e in entries], dim=0)
    rand_t_w = torch.cat([e["rand_t_w"] for e in entries], dim=0)     # [N_total, MAX_ATOMIC_NUM]
    rand_t_l = torch.cat([e["rand_t_l"] for e in entries], dim=0)

    ref_w = torch.stack([e["ref_loss_w"] for e in entries], dim=0)    # [B, 3]
    ref_l = torch.stack([e["ref_loss_l"] for e in entries], dim=0)

    ts = torch.tensor([e["t"] for e in entries], dtype=torch.long)
    ms = torch.tensor([e["m"] for e in entries], dtype=torch.float)
    ys = torch.cat([e["winner"].y if e["winner"].y.dim() > 0 else e["winner"].y.unsqueeze(0)
                    for e in entries], dim=0)

    if not torch.all(ts == ts[0]):
        raise ValueError("All entries in a batch must share the same `t`. Group entries by t before batching.")
    if not torch.all(ms == ms[0]):
        raise ValueError("All entries in a batch must share the same `m`. Group entries by m before batching.")

    # SimpleNamespace-like object the DPO module expects.
    return _Batch(
        winner=winners, loser=losers,
        t=int(ts[0]), m=ms, y=ys,
        rand_l_w=rand_l_w, rand_x_w=rand_x_w, rand_t_w=rand_t_w,
        rand_l_l=rand_l_l, rand_x_l=rand_x_l, rand_t_l=rand_t_l,
        ref_loss_w=ref_w, ref_loss_l=ref_l,
    )


class _Batch:
    """SimpleNamespace-style holder that survives Lightning's batch-transfer hooks."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def to(self, device, *args, **kwargs):
        for k, v in list(self.__dict__.items()):
            if torch.is_tensor(v):
                self.__dict__[k] = v.to(device, *args, **kwargs)
            elif hasattr(v, "to"):
                self.__dict__[k] = v.to(device, *args, **kwargs)
        return self


def make_loader(pairs_file: str | Path, batch_size: int = 1, shuffle: bool = True,
                num_workers: int = 0) -> DataLoader:
    ds = PreferenceDataset(pairs_file)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_pairs)
