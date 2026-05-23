"""Sample real crystals from the generator (SFT baseline and the latest GRPO
checkpoint), score each with BEE-NET, and export structures + predicted Tc to
docs/data/grpo_samples.json for the website's "GRPO trains the material
generator" gallery. Each group shows the GRPO idea: candidates above the group
mean are reinforced, below are suppressed.

CPU is fine (small groups). Usage:
    python viz/export_grpo_samples.py --n 6 --band_gap 1.2218 --guide_w 2.0 --seed 1
"""
import argparse
import glob
import json
import os
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
from grpo.rewards import BEECSORewardModel  # noqa: E402


def latest_ckpt():
    cands = glob.glob(str(PROJECT_ROOT / "hydra_jobs" / "singlerun" / "*" / "grpo_beeonly" / "rl_update_*.pt"))
    return max(cands, key=os.path.getmtime) if cands else None


def sample_group(model, dataset, n, band_gap, guide_w, step_lr):
    ds = SampleDataset(dataset, n)
    batch = Batch.from_data_list([ds[i] for i in range(len(ds))])
    bg = torch.full((batch.num_graphs,), band_gap, dtype=torch.float32)
    _, traj = model.sample(batch, bg, guide_w=guide_w, step_lr=step_lr)
    atoms = traj["atom_types"][-1].cpu().numpy()
    fracs = traj["all_frac_coords"][-1].cpu().numpy()
    latts = traj["all_lattices"][-1].cpu().numpy()
    na = batch.num_atoms.cpu().numpy().tolist()
    off = np.cumsum([0] + na)
    out = []
    for b, n_at in enumerate(na):
        lo, hi = off[b], off[b + 1]
        out.append({"n": n_at, "z": atoms[lo:hi], "frac": fracs[lo:hi], "L": latts[b]})
    return out


def clean_state(c):
    return {
        "num_atoms": torch.tensor([c["n"]]),
        "atom_types": torch.tensor(c["z"]),
        "frac_coords": torch.tensor(c["frac"]),
        "lattices": torch.tensor(c["L"]).unsqueeze(0),
    }


def render(c):
    L = np.asarray(c["L"], dtype=float)
    cart = np.asarray(c["frac"], dtype=float) @ L
    elems = [chemical_symbols[int(z)] if 0 <= int(z) < len(chemical_symbols) else "X" for z in c["z"]]
    comp = {}
    for e in elems:
        comp[e] = comp.get(e, 0) + 1
    formula = "".join(f"{e}{comp[e] if comp[e] > 1 else ''}" for e in sorted(comp))
    return {
        "formula": formula,
        "elems": elems,
        "xyz": [[round(float(x), 4) for x in p] for p in cart],
        "lattice": [[round(float(v), 4) for v in r] for r in L],
    }


def build_group(model, bee, args):
    cs = sample_group(model, args.dataset, args.n, args.band_gap, args.guide_w, args.step_lr)
    scores = bee.score_states([clean_state(c) for c in cs])
    tcads = [float(s.get("tcad", float("nan"))) for s in scores]
    valid = [t for t in tcads if t == t]
    mean = sum(valid) / len(valid) if valid else 0.0
    crystals = []
    for c, t in zip(cs, tcads):
        r = render(c)
        r["tcad"] = round(t, 2) if t == t else None
        r["adv"] = round(t - mean, 2) if t == t else None
        crystals.append(r)
    crystals.sort(key=lambda r: (r["tcad"] is None, -(r["tcad"] or 0)))
    return {"mean_tcad": round(mean, 2), "crystals": crystals}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/superconductor_generator")
    ap.add_argument("--dataset", default="supccomb_12")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--band_gap", type=float, default=1.2218)
    ap.add_argument("--guide_w", type=float, default=2.0)
    ap.add_argument("--step_lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="docs/data/grpo_samples.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    prop_scaler = torch.load(Path(args.model_path) / "prop_scaler.pt")
    try:
        tc_actual = float(prop_scaler.inverse_transform(torch.tensor([[args.band_gap]])).item())
    except Exception:
        tc_actual = None

    bee = BEECSORewardModel(repo_root=".", checkpoints_dir="checkpoints/bee_net/CSO",
                            ensemble_size=4, start_index=0, device="cpu", lmax=1, strict_load=False)

    model, _, _ = load_model(args.model_path, load_data=False)
    model = model.to("cpu").eval()
    print("sampling SFT baseline group ...", flush=True)
    baseline = build_group(model, bee, args)

    out = {"target_tc": tc_actual, "band_gap_scaled": args.band_gap, "guide_w": args.guide_w,
           "baseline": baseline, "grpo": None}

    ckpt = latest_ckpt()
    if ckpt:
        update = int("".join(filter(str.isdigit, os.path.basename(ckpt))) or 0)
        print(f"loading GRPO checkpoint {ckpt} (update {update}) ...", flush=True)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state["state_dict"])
        model.eval()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print("sampling GRPO group ...", flush=True)
        grpo = build_group(model, bee, args)
        grpo["update"] = update
        out["grpo"] = grpo

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {out_path}: baseline mean Tc={baseline['mean_tcad']}, "
          f"grpo={'update '+str(out['grpo']['update']) if out['grpo'] else 'n/a'}", flush=True)


if __name__ == "__main__":
    main()
