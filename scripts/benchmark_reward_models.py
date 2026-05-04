#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from diffcsp.rl.rewards import (
    BEECSORewardModel,
    MEGNetRewardModel,
    M3GNetRewardModel,
    TorchProxyRewardModel,
)


def structure_to_clean_state(cif_path: Path) -> dict:
    from pymatgen.core import Structure

    structure = Structure.from_file(cif_path)
    return {
        "num_atoms": torch.LongTensor([len(structure)]),
        "atom_types": torch.LongTensor(structure.atomic_numbers),
        "frac_coords": torch.tensor(structure.frac_coords, dtype=torch.float32),
        "lattices": torch.tensor(structure.lattice.matrix, dtype=torch.float32).unsqueeze(0),
    }


def bench_model(name: str, builder, states, repeats: int):
    started = time.perf_counter()
    model = builder()
    warm_load_s = time.perf_counter() - started

    started = time.perf_counter()
    outputs = None
    for _ in range(repeats):
        outputs = model.score_states(states)
    steady_state_s = (time.perf_counter() - started) / max(repeats, 1)

    print(f"{name} warm_load_s={warm_load_s:.4f} steady_state_s={steady_state_s:.4f} batch_size={len(states)}")
    if outputs is not None:
        print(f"{name} first_output={outputs[0]}")


def bench_proxy(checkpoint_path: Path, states, bee_outputs, meg_outputs, repeats: int):
    started = time.perf_counter()
    model = TorchProxyRewardModel(checkpoint_path)
    warm_load_s = time.perf_counter() - started

    started = time.perf_counter()
    outputs = None
    for _ in range(repeats):
        outputs = model.score_states(states, bee_outputs, meg_outputs)
    steady_state_s = (time.perf_counter() - started) / max(repeats, 1)

    print(
        f"Proxy[{checkpoint_path.name}] warm_load_s={warm_load_s:.4f} "
        f"steady_state_s={steady_state_s:.4f} batch_size={len(states)}"
    )
    if outputs is not None:
        print(f"Proxy[{checkpoint_path.name}] first_output={outputs[0]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--cif", action="append", required=True, help="One or more CIF files to score.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bee-device", default="cpu")
    parser.add_argument("--bee-ensemble-size", type=int, default=4)
    parser.add_argument("--skip-m3g", action="store_true")
    parser.add_argument("--proxy-checkpoint", action="append", default=[])
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    states = [structure_to_clean_state(Path(cif).resolve()) for cif in args.cif]

    bench_model(
        "BEE-Net-CSO",
        lambda: BEECSORewardModel(
            repo_root=repo_root,
            ensemble_size=args.bee_ensemble_size,
            device=args.bee_device,
        ),
        states,
        args.repeats,
    )
    bench_model(
        "MEGNet",
        lambda: MEGNetRewardModel(repo_root=repo_root),
        states,
        args.repeats,
    )
    bee_model = BEECSORewardModel(
        repo_root=repo_root,
        ensemble_size=args.bee_ensemble_size,
        device=args.bee_device,
    )
    meg_model = MEGNetRewardModel(repo_root=repo_root)
    bee_outputs = bee_model.score_states(states)
    meg_outputs = meg_model.score_states(states)
    if not args.skip_m3g:
        bench_model(
            "M3GNet",
            lambda: M3GNetRewardModel(),
            states,
            args.repeats,
        )
    for checkpoint in args.proxy_checkpoint:
        bench_proxy(Path(checkpoint).resolve(), states, bee_outputs, meg_outputs, args.repeats)


if __name__ == "__main__":
    main()
