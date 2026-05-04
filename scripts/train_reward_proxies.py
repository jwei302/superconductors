#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from diffcsp.rl.rewards import (
    BEECSORewardModel,
    MEGNetRewardModel,
    PROXY_FEATURE_NAMES,
    ProxyMLP,
    build_proxy_feature_vector,
)


@dataclass
class TargetSpec:
    name: str
    column: str
    task_type: str
    direction: str
    reward_weight: float


def parse_target_spec(raw_value: str, default_task_type: str, default_direction: str) -> TargetSpec:
    parts = raw_value.split(":")
    if len(parts) < 2:
        raise ValueError(
            f"Target spec '{raw_value}' must have at least name:column and may optionally add direction:weight."
        )
    name, column = parts[0], parts[1]
    direction = parts[2] if len(parts) >= 3 else default_direction
    reward_weight = float(parts[3]) if len(parts) >= 4 else 1.0
    return TargetSpec(
        name=name,
        column=column,
        task_type=default_task_type,
        direction=direction,
        reward_weight=reward_weight,
    )


def structure_to_clean_state(cif_path: Path) -> dict:
    structure = Structure.from_file(cif_path)
    return {
        "num_atoms": torch.LongTensor([len(structure)]),
        "atom_types": torch.LongTensor(structure.atomic_numbers),
        "frac_coords": torch.tensor(structure.frac_coords, dtype=torch.float32),
        "lattices": torch.tensor(structure.lattice.matrix, dtype=torch.float32).unsqueeze(0),
    }


def build_feature_table(
    manifest: pd.DataFrame,
    repo_root: Path,
    cif_column: str,
    bee_device: str,
    bee_ensemble_size: int,
) -> tuple[np.ndarray, list[int]]:
    bee = BEECSORewardModel(
        repo_root=repo_root,
        ensemble_size=bee_ensemble_size,
        device=bee_device,
    )
    meg = MEGNetRewardModel(repo_root=repo_root)

    features = []
    valid_indices = []
    for idx, row in manifest.iterrows():
        cif_path = Path(row[cif_column]).expanduser()
        if not cif_path.is_absolute():
            cif_path = (repo_root / cif_path).resolve()
        clean_state = structure_to_clean_state(cif_path)
        bee_output = bee.score_states([clean_state])[0]
        meg_output = meg.score_states([clean_state])[0]
        feature_vector = build_proxy_feature_vector(clean_state, bee_output, meg_output)
        if feature_vector is None:
            continue
        features.append(feature_vector)
        valid_indices.append(idx)

    if not features:
        raise RuntimeError("No valid structures were available to build proxy features.")

    return np.asarray(features, dtype=np.float32), valid_indices


def fit_one_proxy(
    output_dir: Path,
    features: np.ndarray,
    labels: np.ndarray,
    spec: TargetSpec,
    hidden_dim: int,
    depth: int,
    epochs: int,
    batch_size: int,
    lr: float,
    val_fraction: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    split = max(1, int(round(len(labels) * (1.0 - val_fraction))))
    train_idx = indices[:split]
    val_idx = indices[split:] if split < len(indices) else indices[:1]

    train_x = features[train_idx]
    val_x = features[val_idx]
    feature_mean = train_x.mean(axis=0)
    feature_std = train_x.std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    train_x = (train_x - feature_mean) / feature_std
    val_x = (val_x - feature_mean) / feature_std

    label_mean = 0.0
    label_std = 1.0
    train_y = labels[train_idx].astype(np.float32)
    val_y = labels[val_idx].astype(np.float32)
    if spec.task_type == "regression":
        label_mean = float(train_y.mean())
        label_std = float(max(train_y.std(), 1e-6))
        train_y = (train_y - label_mean) / label_std
        val_y = (val_y - label_mean) / label_std

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(train_x, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(val_x, dtype=torch.float32),
            torch.tensor(val_y, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    model = ProxyMLP(input_dim=features.shape[1], hidden_dim=hidden_dim, depth=depth)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if spec.task_type == "binary":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                losses.append(float(criterion(preds, batch_y).item()))
        val_loss = float(np.mean(losses))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    checkpoint = {
        "state_dict": best_state,
        "metadata": {
            "name": spec.name,
            "task_type": spec.task_type,
            "direction": spec.direction,
            "reward_weight": spec.reward_weight,
            "feature_names": PROXY_FEATURE_NAMES,
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "label_mean": label_mean,
            "label_std": label_std,
            "hidden_dim": hidden_dim,
            "depth": depth,
            "val_loss": best_val,
        },
    }
    checkpoint_path = output_dir / f"{spec.name}_proxy.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"saved {checkpoint_path} val_loss={best_val:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--manifest", required=True, help="CSV file with CIF paths and target labels.")
    parser.add_argument("--cif-column", default="cif")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--regression-target", action="append", default=[])
    parser.add_argument("--binary-target", action="append", default=[])
    parser.add_argument("--bee-device", default="cpu")
    parser.add_argument("--bee-ensemble-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_specs = []
    for raw_value in args.regression_target:
        target_specs.append(parse_target_spec(raw_value, "regression", "minimize"))
    for raw_value in args.binary_target:
        target_specs.append(parse_target_spec(raw_value, "binary", "maximize"))
    if not target_specs:
        raise ValueError("Provide at least one --regression-target or --binary-target.")

    manifest = pd.read_csv(Path(args.manifest).resolve())
    features, valid_indices = build_feature_table(
        manifest=manifest,
        repo_root=repo_root,
        cif_column=args.cif_column,
        bee_device=args.bee_device,
        bee_ensemble_size=args.bee_ensemble_size,
    )
    filtered_manifest = manifest.iloc[valid_indices].reset_index(drop=True)

    for spec in target_specs:
        if spec.column not in filtered_manifest.columns:
            raise KeyError(f"Target column '{spec.column}' not found in manifest.")
        label_series = filtered_manifest[spec.column]
        if spec.task_type == "binary":
            label_mask = label_series.notna()
            labels = label_series[label_mask].astype(float).to_numpy()
        else:
            label_mask = np.isfinite(label_series.to_numpy(dtype=float))
            labels = label_series.to_numpy(dtype=float)[label_mask]

        if labels.size == 0:
            raise RuntimeError(f"No usable labels found for target '{spec.name}'.")
        task_features = features[np.asarray(label_mask, dtype=bool)]
        fit_one_proxy(
            output_dir=output_dir,
            features=task_features,
            labels=labels,
            spec=spec,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
