"""Build a preference-pair file for offline Diffusion-DPO.

Inputs:
  --score_file:   JSON from scripts/dpo/score_ehull.py (list of dicts with name/hash/energy_per_atom/valid)
  --cif_dir:      directory containing the CIFs that were scored
  --model_path:   Hydra run dir of π_ref (must have *.ckpt + hparams.yaml + scalers)
  --out_path:     destination .pt file
  --top_frac:     fraction of valid pool to use as winners (default 0.25)
  --bottom_frac:  fraction to use as losers (default 0.25)
  --gap_margin:   minimum |E_loser - E_winner| (eV/atom) to keep a pair (default 0.0)
  --k_draws:      number of (t, ε, m) noise tuples cached per pair (default 4)
  --max_pairs:    cap total pair count after filtering (default 1000)
  --y:            scaled property value (default 1.2218 = Tc=10K). All pairs use the same y.

Output (`pairs_v1.pt`): list of dicts, one per (pair, draw) entry. Each contains:
    winner, loser:        PyG Data
    t:                    int in [1, T]
    m:                    0 or 1 (CFG property_indicator)
    rand_l_w/rand_x_w/rand_t_w (and _l)
    ref_loss_w, ref_loss_l: tensor [3] = per-component π_ref MSE
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from pymatgen.core import Structure
from torch_geometric.data import Batch, Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # project root + scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_utils import load_model   # noqa: E402
from diffcsp.pl_modules.dpo_module import DPODiffusion   # noqa: E402
from diffcsp.pl_modules.diffusion_w_type import MAX_ATOMIC_NUM   # noqa: E402


def pymatgen_to_pyg(struct: Structure, y: float) -> Data:
    return Data(
        num_atoms=torch.tensor([len(struct)], dtype=torch.long),
        num_nodes=len(struct),
        atom_types=torch.tensor([site.specie.Z for site in struct.sites], dtype=torch.long),
        frac_coords=torch.tensor(struct.frac_coords, dtype=torch.float),
        lengths=torch.tensor([list(struct.lattice.abc)], dtype=torch.float),
        angles=torch.tensor([list(struct.lattice.angles)], dtype=torch.float),
        y=torch.tensor([y], dtype=torch.float),
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score_file", required=True, type=Path)
    ap.add_argument("--cif_dir", required=True, type=Path)
    ap.add_argument("--model_path", required=True, type=Path)
    ap.add_argument("--out_path", required=True, type=Path)
    ap.add_argument("--top_frac", type=float, default=0.25)
    ap.add_argument("--bottom_frac", type=float, default=0.25)
    ap.add_argument("--gap_margin", type=float, default=0.0)
    ap.add_argument("--k_draws", type=int, default=4)
    ap.add_argument("--max_pairs", type=int, default=1000)
    ap.add_argument("--y", type=float, default=1.2218, help="Scaled property value (default = Tc=10K)")
    ap.add_argument("--p_uncond", type=float, default=0.2, help="CFG dropout rate (matches SFT)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    rng = torch.Generator().manual_seed(args.seed)

    # 1. Load scores
    print(f"[load] scores from {args.score_file}")
    rows = json.loads(args.score_file.read_text())
    valid_rows = [r for r in rows if r["valid"] and r["energy_per_atom"] is not None and r["hash"] is not None]
    print(f"  {len(valid_rows)}/{len(rows)} valid")
    if len(valid_rows) < 4:
        raise RuntimeError("Not enough valid structures to form pairs.")

    # 2. Sort by energy_per_atom ascending (lower = more stable = better)
    valid_rows.sort(key=lambda r: r["energy_per_atom"])
    n = len(valid_rows)
    n_top = max(1, int(n * args.top_frac))
    n_bot = max(1, int(n * args.bottom_frac))
    winners = valid_rows[:n_top]
    losers = valid_rows[-n_bot:]
    print(f"  winners (top {args.top_frac:.0%}): {n_top}   E/atom range [{winners[0]['energy_per_atom']:.3f}, {winners[-1]['energy_per_atom']:.3f}]")
    print(f"  losers (bot {args.bottom_frac:.0%}):  {n_bot}   E/atom range [{losers[0]['energy_per_atom']:.3f}, {losers[-1]['energy_per_atom']:.3f}]")

    # 3. Form pairs (Cartesian product), filter by gap_margin, subsample
    candidates = []
    for w in winners:
        for l in losers:
            gap = l["energy_per_atom"] - w["energy_per_atom"]
            if gap >= args.gap_margin:
                candidates.append((w, l, gap))
    print(f"  {len(candidates)} candidate pairs (gap_margin = {args.gap_margin})")
    if len(candidates) > args.max_pairs:
        idxs = torch.randperm(len(candidates), generator=rng)[: args.max_pairs].tolist()
        candidates = [candidates[i] for i in idxs]
        print(f"  → subsampled to {args.max_pairs}")
    if not candidates:
        raise RuntimeError("No pairs after filtering — increase top/bottom_frac or lower gap_margin.")

    # 4. Convert CIFs → PyG Data (cached so we only parse each CIF once)
    print(f"[load] CIFs from {args.cif_dir}")
    pyg_cache: dict[str, Data] = {}
    def get_pyg(name: str) -> Data:
        if name not in pyg_cache:
            cif_path = args.cif_dir / f"{name}.cif"
            struct = Structure.from_file(str(cif_path))
            pyg_cache[name] = pymatgen_to_pyg(struct, args.y)
        return pyg_cache[name]

    # 5. Load π_ref
    print(f"[load] π_ref from {args.model_path}")
    sft, _, _ = load_model(args.model_path.resolve(), load_data=False)
    sft.to(args.device).eval()
    dpo_helper = DPODiffusion(policy=sft, beta=1.0, total_timesteps=1000, train_only_adapters=False)

    # 6. For each pair, build k_draws (t, m, noise) tuples and cache π_ref MSEs.
    print(f"[cache] {len(candidates)} pairs × {args.k_draws} draws = {len(candidates) * args.k_draws} entries")
    T = sft.beta_scheduler.timesteps
    entries = []
    for pi, (w_row, l_row, gap) in enumerate(candidates):
        sw = get_pyg(w_row["name"])
        sl = get_pyg(l_row["name"])
        # Each draw: independent t, m, noise.
        for k in range(args.k_draws):
            t_val = int(torch.randint(1, T + 1, (1,), generator=rng).item())
            m_val = 1 if torch.rand((), generator=rng).item() > args.p_uncond else 0
            rand_l_w = torch.randn(3, 3, generator=rng)
            rand_x_w = torch.randn(sw.num_nodes, 3, generator=rng)
            rand_t_w = torch.randn(sw.num_nodes, MAX_ATOMIC_NUM, generator=rng)
            rand_l_l = torch.randn(3, 3, generator=rng)
            rand_x_l = torch.randn(sl.num_nodes, 3, generator=rng)
            rand_t_l = torch.randn(sl.num_nodes, MAX_ATOMIC_NUM, generator=rng)

            # Forward π_ref to get per-component MSEs.
            wbatch = Batch.from_data_list([sw]).to(args.device)
            lbatch = Batch.from_data_list([sl]).to(args.device)
            with torch.no_grad():
                m_t = torch.tensor([float(m_val)], device=args.device)
                y_t = torch.tensor([args.y], device=args.device)
                ml_w, mf_w, ma_w = dpo_helper.per_component_mse(
                    wbatch, t_val, rand_l_w.unsqueeze(0).to(args.device),
                    rand_x_w.to(args.device), rand_t_w.to(args.device), m_t, y_t)
                ml_l, mf_l, ma_l = dpo_helper.per_component_mse(
                    lbatch, t_val, rand_l_l.unsqueeze(0).to(args.device),
                    rand_x_l.to(args.device), rand_t_l.to(args.device), m_t, y_t)
            ref_w = torch.stack([ml_w[0], mf_w[0], ma_w[0]]).cpu()
            ref_l = torch.stack([ml_l[0], mf_l[0], ma_l[0]]).cpu()

            entries.append({
                "winner": sw, "loser": sl,
                "t": t_val, "m": m_val,
                "rand_l_w": rand_l_w, "rand_x_w": rand_x_w, "rand_t_w": rand_t_w,
                "rand_l_l": rand_l_l, "rand_x_l": rand_x_l, "rand_t_l": rand_t_l,
                "ref_loss_w": ref_w, "ref_loss_l": ref_l,
                "reward_gap": float(gap),
                "winner_name": w_row["name"], "loser_name": l_row["name"],
            })
        if (pi + 1) % 50 == 0:
            print(f"  cached {pi+1}/{len(candidates)} pairs")

    # 7. Save
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(entries, args.out_path)
    print(f"[save] {len(entries)} entries → {args.out_path}")
    # Quick summary stats
    ref_w_mat = torch.stack([e["ref_loss_w"] for e in entries])
    ref_l_mat = torch.stack([e["ref_loss_l"] for e in entries])
    gaps = [e["reward_gap"] for e in entries]
    print(f"  ref_loss_w means (lattice/coord/type): {ref_w_mat.mean(0).tolist()}")
    print(f"  ref_loss_l means (lattice/coord/type): {ref_l_mat.mean(0).tolist()}")
    print(f"  reward_gap min/median/max: {min(gaps):.4f} / {sorted(gaps)[len(gaps)//2]:.4f} / {max(gaps):.4f} eV/atom")


if __name__ == "__main__":
    main()
