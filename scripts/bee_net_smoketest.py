#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch_geometric as tg
from ase import Atom
from ase.io import read as ase_read
from ase.neighborlist import neighbor_list


REPO_ROOT = Path(__file__).resolve().parents[1]
BEE_WORKFLOW_ROOT = REPO_ROOT / "external" / "BEE-NET" / "workflow"
if str(BEE_WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(BEE_WORKFLOW_ROOT))

from utils.model import PeriodicNetwork  # noqa: E402


DEFAULT_DTYPE = torch.float64
torch.set_default_dtype(DEFAULT_DTYPE)

R_MAX = 4.0
FREQ_FINAL = np.arange(0.25, 101, 2)
MU_STAR = 0.1


def build_type_encodings():
    type_encoding = {}
    specie_am = []
    for z in range(1, 119):
        specie = Atom(z)
        type_encoding[specie.symbol] = z - 1
        specie_am.append(specie.mass)
    type_onehot = torch.eye(len(type_encoding), dtype=DEFAULT_DTYPE)
    am_onehot = torch.diag(torch.tensor(specie_am, dtype=DEFAULT_DTYPE))
    return type_encoding, type_onehot, am_onehot


TYPE_ENCODING, TYPE_ONEHOT, AM_ONEHOT = build_type_encodings()


def build_data_from_atoms(atoms, r_max: float = R_MAX):
    symbols = list(atoms.symbols).copy()
    positions = torch.from_numpy(atoms.positions.copy()).to(DEFAULT_DTYPE)
    lattice = torch.from_numpy(atoms.cell.array.copy()).to(DEFAULT_DTYPE).unsqueeze(0)

    edge_src, edge_dst, edge_shift = neighbor_list(
        "ijS", a=atoms, cutoff=r_max, self_interaction=True
    )
    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[
        torch.from_numpy(edge_src)
    ]
    edge_vec = (
        positions[torch.from_numpy(edge_dst)]
        - positions[torch.from_numpy(edge_src)]
        + torch.einsum(
            "ni,nij->nj",
            torch.tensor(edge_shift, dtype=DEFAULT_DTYPE),
            lattice[edge_batch],
        )
    )

    x = AM_ONEHOT[[TYPE_ENCODING[symbol] for symbol in symbols]]
    z = TYPE_ONEHOT[[TYPE_ENCODING[symbol] for symbol in symbols]]

    return tg.data.Data(
        pos=positions,
        lattice=lattice,
        symbol=symbols,
        x=x,
        z=z,
        edge_index=torch.stack(
            [torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0
        ),
        edge_shift=torch.tensor(edge_shift, dtype=DEFAULT_DTYPE),
        edge_vec=edge_vec,
        edge_len=edge_vec.norm(dim=1).cpu().numpy(),
        target=torch.zeros((1, len(FREQ_FINAL)), dtype=DEFAULT_DTYPE),
    )


def get_model(device: torch.device):
    init_dict = dict(
        in_dim=118,
        em_dim=64,
        irreps_in="64x0e",
        irreps_out=f"{len(FREQ_FINAL)}x0e",
        irreps_node_attr="64x0e",
        layers=2,
        mul=32,
        lmax=3,
        max_radius=R_MAX,
        num_neighbors=17.133204568916714,
        reduce_output=True,
        p=0.0,
    )
    model = PeriodicNetwork(**init_dict)
    model.pool = True
    model.to(device)
    model.eval()
    return model


def cal_lambda(freq_w, alpha_f):
    lambda_f = 0.0
    for i in range(1, len(freq_w)):
        if freq_w[i] > 0:
            dw = freq_w[i] - freq_w[i - 1]
            lambda_f += (alpha_f[i] / freq_w[i]) * dw
    return 2.0 * lambda_f


def cal_w_log(freq_w, alpha_f, lamb):
    if lamb <= 0:
        return float("nan")
    value = 0.0
    for i in range(1, len(freq_w)):
        if freq_w[i] > 0:
            dw = freq_w[i] - freq_w[i - 1]
            value += alpha_f[i] * np.log(freq_w[i]) * dw / freq_w[i]
    return np.exp(2.0 * value / lamb) / 0.08617


def cal_w_sq(freq_w, alpha_f, lamb):
    if lamb <= 0:
        return float("nan")
    value = 0.0
    for i in range(1, len(freq_w)):
        if freq_w[i] > 0:
            dw = freq_w[i] - freq_w[i - 1]
            value += alpha_f[i] * freq_w[i] * dw
    return ((2.0 * value / lamb) ** 0.5) / 0.08617


def cal_tc(lamb, omega_log, mu: float = MU_STAR):
    denom = lamb - mu * (1.0 + 0.62 * lamb)
    if denom <= 0 or omega_log <= 0:
        return float("nan")
    frac = -1.04 * (1.0 + lamb) / denom
    return (omega_log / 1.2) * np.exp(frac)


def cal_tc_ad(lamb, wlog, w2, tc, mu: float = MU_STAR):
    if not np.isfinite(tc) or not np.isfinite(wlog) or not np.isfinite(w2) or wlog <= 0:
        return float("nan")
    f1 = (1 + (lamb / (2.46 * (1 + 3.8 * mu))) ** 1.5) ** (1 / 3)
    f2 = 1 + ((lamb**2) * ((w2 / wlog) - 1)) / (
        lamb**2 + (1.82 * (1 + 6.3 * mu) * (w2 / wlog)) ** 2
    )
    return f1 * f2 * tc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif", required=True, help="Path to a CIF file to score.")
    parser.add_argument(
        "--checkpoints-dir",
        default=str(REPO_ROOT / "checkpoints" / "bee_net" / "CSO"),
        help="Directory containing BEE-NET CSO checkpoints.",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=1,
        help="How many ensemble members to average for the smoke test.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Checkpoint index to start from within the ensemble directory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    cif_path = Path(args.cif).resolve()
    checkpoints_dir = Path(args.checkpoints_dir).resolve()
    checkpoint_paths = sorted(checkpoints_dir.glob("model_CSO_derived_EMD_0_*.pt1.pt"))

    if not cif_path.is_file():
        raise FileNotFoundError(f"CIF not found: {cif_path}")
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    if args.start_index >= len(checkpoint_paths):
        raise ValueError(
            f"start-index {args.start_index} out of range for {len(checkpoint_paths)} checkpoints"
        )

    selected_paths = checkpoint_paths[args.start_index : args.start_index + args.ensemble_size]
    if not selected_paths:
        raise ValueError("No checkpoints selected. Adjust --start-index/--ensemble-size.")

    device = resolve_device(args.device)
    atoms = ase_read(str(cif_path))
    data = build_data_from_atoms(atoms)
    dataloader = tg.loader.DataLoader([data], batch_size=1)
    model = get_model(device)

    predictions = []
    checkpoint_timings = []

    for checkpoint_path in selected_paths:
        started = time.perf_counter()
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                output = model(batch).detach().cpu().numpy()[0]
        predictions.append(output)
        checkpoint_timings.append(time.perf_counter() - started)

    pred_avg = np.mean(np.asarray(predictions), axis=0)
    lamb = cal_lambda(FREQ_FINAL, pred_avg)
    wlog = cal_w_log(FREQ_FINAL, pred_avg, lamb)
    w2 = cal_w_sq(FREQ_FINAL, pred_avg, lamb)
    tc = cal_tc(lamb, wlog)
    tcad = cal_tc_ad(lamb, wlog, w2, tc)

    print(f"CIF: {cif_path}")
    print(f"Formula: {atoms.get_chemical_formula()}")
    print(f"Device: {device}")
    print(f"Checkpoints used: {len(selected_paths)}")
    print(f"Mean time per checkpoint (s): {np.mean(checkpoint_timings):.4f}")
    print(f"Total inference time (s): {np.sum(checkpoint_timings):.4f}")
    print(f"Predicted spectrum bins: {pred_avg.shape[0]}")
    print(f"lambda: {lamb:.6f}")
    print(f"wlog (K): {wlog:.6f}")
    print(f"w2 (K): {w2:.6f}")
    print(f"Tc Allen-Dynes (K): {tc:.6f}")
    print(f"Tc corrected / tcad (K): {tcad:.6f}")
    print("First 10 spectrum values:", np.array2string(pred_avg[:10], precision=5))


if __name__ == "__main__":
    main()
