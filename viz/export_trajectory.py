"""Export DiffCSP denoising trajectories to a compact JSON for the web scrubber.

Runs ab-initio sampling (no dataset file needed — atom counts are drawn from the
`train_dist` prior in scripts/generation.py), captures the full reverse-diffusion
trajectory (`traj_stack`), converts each frame to Cartesian coordinates + element
symbols, downsamples to a slider-friendly number of frames, and writes
`docs/data/trajectories.json` consumed by the GitHub-Pages site.

Usage (CPU is fine; ~1-2 min per batch of structures at 1000 steps):
    python viz/export_trajectory.py \
        --model_path models/superconductor_generator \
        --dataset supccomb_12 --n 3 --frames 100 \
        --guide_w 2.0 --band_gap 1.2218 --seed 0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from torch_geometric.data import Batch  # noqa: E402

from diffcsp.common.data_utils import chemical_symbols  # noqa: E402
from eval_utils import load_model  # noqa: E402
from generation import SampleDataset  # noqa: E402


def frame_indices(n_steps_total, n_frames):
    """Pick frame indices over [0, n_steps_total] with denser sampling near the
    crystal end (low diffusion-t), where most structure forms. Index 0 == pure
    noise (t=T), index -1 == final crystal (t=0)."""
    if n_frames >= n_steps_total + 1:
        return list(range(n_steps_total + 1))
    # quadratic spacing: sparse at the noisy start, dense at the clean end.
    u = np.linspace(0.0, 1.0, n_frames)
    idx = (u ** 1.7 * n_steps_total).round().astype(int)
    idx = sorted(set(idx.tolist()) | {0, n_steps_total})
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/superconductor_generator")
    ap.add_argument("--dataset", default="supccomb_12")
    ap.add_argument("--n", type=int, default=3, help="number of crystals to sample")
    ap.add_argument("--frames", type=int, default=100, help="frames kept per crystal")
    ap.add_argument("--guide_w", type=float, default=2.0)
    ap.add_argument("--band_gap", type=float, default=1.2218,
                    help="SCALED target Tc (prop_scaler space)")
    ap.add_argument("--step_lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docs/data/trajectories.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, _, cfg = load_model(args.model_path, load_data=False)
    model = model.to("cpu").eval()

    prop_scaler = torch.load(Path(args.model_path) / "prop_scaler.pt")
    try:
        tc_actual = float(prop_scaler.inverse_transform(
            torch.tensor([[args.band_gap]])).item())
    except Exception:
        tc_actual = None

    ds = SampleDataset(args.dataset, args.n)
    batch = Batch.from_data_list([ds[i] for i in range(len(ds))])

    band_gap = torch.full((batch.num_graphs,), args.band_gap, dtype=torch.float32)
    print(f"sampling {batch.num_graphs} crystals, "
          f"num_atoms={batch.num_atoms.tolist()} ...", flush=True)
    _, traj = model.sample(batch, band_gap, guide_w=args.guide_w, step_lr=args.step_lr)

    # traj_stack: atom_types (S, N) ints; all_frac_coords (S, N, 3);
    # all_lattices (S, B, 3, 3); frame 0 == noise (t=T), frame -1 == crystal (t=0).
    atom_types = traj["atom_types"].cpu().numpy()           # (S, N)
    fracs = traj["all_frac_coords"].cpu().numpy()           # (S, N, 3)
    lattices = traj["all_lattices"].cpu().numpy()           # (S, B, 3, 3)
    num_atoms = batch.num_atoms.cpu().numpy().tolist()
    S = fracs.shape[0]
    n_steps_total = S - 1
    keep = frame_indices(n_steps_total, args.frames)

    # split node arrays per crystal. We export per-frame fractional coords plus a
    # single fixed (final) lattice per crystal so the web viewer can hold the unit
    # cell in one orientation while atoms denoise inside it (cart = frac @ L_fixed).
    offsets = np.cumsum([0] + num_atoms)
    s_final = keep[-1]  # last kept frame == t=0 (the finished crystal)
    crystals = []
    for b, n_at in enumerate(num_atoms):
        lo, hi = offsets[b], offsets[b + 1]
        L_fixed = lattices[s_final, b]                      # (3, 3)
        frames = []
        for s in keep:
            fr = fracs[s, lo:hi]                            # (n_at, 3) wrapped frac
            zs = atom_types[s, lo:hi].astype(int)
            elems = [chemical_symbols[int(z)] if 0 <= z < len(chemical_symbols)
                     else "X" for z in zs]
            # diffusion timestep for this frame (S-1 == noise, 0 == crystal)
            t_diff = n_steps_total - s
            frames.append({
                "t": int(t_diff),
                "elems": elems,
                "frac": [[round(float(c), 4) for c in pos] for pos in fr],
            })
        final_elems = frames[-1]["elems"]
        comp = {}
        for e in final_elems:
            comp[e] = comp.get(e, 0) + 1
        formula = "".join(f"{e}{comp[e] if comp[e] > 1 else ''}"
                          for e in sorted(comp))
        crystals.append({
            "id": b,
            "num_atoms": int(n_at),
            "formula": formula,
            "lattice": [[round(float(v), 4) for v in row] for row in L_fixed],
            "frames": frames,
        })

    out = {
        "meta": {
            "model_path": args.model_path,
            "dataset": args.dataset,
            "guide_w": args.guide_w,
            "band_gap_scaled": args.band_gap,
            "tc_target_actual": tc_actual,
            "n_steps_total": int(n_steps_total),
            "n_frames": len(keep),
            "seed": args.seed,
        },
        "crystals": crystals,
    }

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path} ({kb:.0f} KB): {len(crystals)} crystals, "
          f"{len(keep)} frames each", flush=True)
    for c in crystals:
        print(f"  crystal {c['id']}: {c['formula']} ({c['num_atoms']} atoms)")


if __name__ == "__main__":
    main()
