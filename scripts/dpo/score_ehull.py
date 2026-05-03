"""CHGNet single-point scoring with a structure-hash cache.

Day-1 deliverable: score N CIFs from a directory and produce a dict
{hash: {energy, energy_per_atom, formula, num_atoms, valid, error?}} on disk.

Per the plan: single-point only, no relaxation. E_above_hull lookup is
deferred — using `energy_per_atom` as the stability proxy for now. If the
MP convex hull is wired in later, this script gains an `e_above_hull` key.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def canonical_hash(struct: Structure, sg_tol: float = 0.1) -> str:
    """Composition + reduced cell + spacegroup → 16-char SHA-256 prefix."""
    try:
        sg = SpacegroupAnalyzer(struct, symprec=sg_tol).get_space_group_number()
    except Exception:
        sg = -1
    reduced = struct.copy()
    try:
        reduced = reduced.get_reduced_structure()
    except Exception:
        pass
    key = (
        reduced.composition.reduced_formula,
        f"{reduced.lattice.a:.3f}_{reduced.lattice.b:.3f}_{reduced.lattice.c:.3f}",
        f"{reduced.lattice.alpha:.1f}_{reduced.lattice.beta:.1f}_{reduced.lattice.gamma:.1f}",
        sg,
    )
    return hashlib.sha256(repr(key).encode()).hexdigest()[:16]


def is_valid_structure(struct: Structure, min_dist: float = 0.5) -> bool:
    if struct is None or len(struct) == 0:
        return False
    if not (2.0 < struct.lattice.a < 30 and 2.0 < struct.lattice.b < 30 and 2.0 < struct.lattice.c < 30):
        return False
    try:
        d = struct.distance_matrix
        d = d + np.eye(len(struct)) * 1e6
        if d.min() < min_dist:
            return False
        if struct.volume < 1.0:
            return False
    except Exception:
        return False
    return True


def load_cifs(cif_dir: Path) -> list[tuple[str, Structure | None]]:
    cif_paths = sorted(cif_dir.glob("*.cif"))
    out = []
    for p in cif_paths:
        try:
            s = Structure.from_file(str(p))
        except Exception as e:
            s = None
            print(f"[warn] failed to parse {p.name}: {e}", file=sys.stderr)
        out.append((p.stem, s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif_dir", required=True)
    ap.add_argument("--out", required=True, help="Output JSON path for scores")
    ap.add_argument("--cache", default="dpo_artifacts/score_cache/chgnet_cache.json")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    cif_dir = Path(args.cif_dir)
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"[cache] loaded {len(cache)} prior entries from {cache_path}")

    items = load_cifs(cif_dir)
    print(f"[score] {len(items)} CIFs in {cif_dir}")

    # Pre-compute hashes + validity. Anything invalid gets NaN energy.
    rows = []
    for name, s in items:
        if s is None or not is_valid_structure(s):
            rows.append({
                "name": name, "hash": None, "valid": False,
                "formula": (s.composition.reduced_formula if s else None),
                "num_atoms": (len(s) if s else 0),
                "energy": float("nan"), "energy_per_atom": float("nan"),
            })
        else:
            h = canonical_hash(s)
            rows.append({
                "name": name, "hash": h, "valid": True,
                "formula": s.composition.reduced_formula,
                "num_atoms": len(s),
                "energy": None, "energy_per_atom": None,
                "_struct": s,
            })

    # Lazy CHGNet load — only if we actually need to score something.
    to_score = [r for r in rows if r["valid"] and r["hash"] not in cache]
    print(f"[score] {len(to_score)} new structures to score (rest cached or invalid)")

    if to_score:
        from chgnet.model.model import CHGNet
        device = "cuda" if torch.cuda.is_available() else "cpu"
        chgnet = CHGNet.load()
        chgnet.to(device).eval()
        print(f"[chgnet] loaded on {device}")

        t0 = time.time()
        for i in range(0, len(to_score), args.batch_size):
            chunk = to_score[i:i + args.batch_size]
            structs = [r["_struct"] for r in chunk]
            # task='e' → energy only; skips forces/stress/magmom (which require autograd
            # and would error inside `torch.no_grad`).
            with torch.no_grad():
                preds = chgnet.predict_structure(structs, task="e")
            if not isinstance(preds, list):
                preds = [preds]
            for r, p in zip(chunk, preds):
                e = float(p["e"])  # energy per atom in eV (CHGNet convention)
                cache[r["hash"]] = {"energy_per_atom": e}
            if (i // args.batch_size) % 5 == 0:
                rate = (i + len(chunk)) / max(time.time() - t0, 1e-3)
                print(f"  scored {i+len(chunk)}/{len(to_score)}  ({rate:.1f}/s)")

        cache_path.write_text(json.dumps(cache))
        print(f"[cache] wrote {len(cache)} entries to {cache_path}")

    # Resolve energies for all rows from cache
    for r in rows:
        if r["valid"] and r["hash"] in cache:
            e_pa = cache[r["hash"]]["energy_per_atom"]
            r["energy_per_atom"] = e_pa
            r["energy"] = e_pa * r["num_atoms"]
        r.pop("_struct", None)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=1))
    valid_e = [r["energy_per_atom"] for r in rows if r["valid"]]
    print(f"[done] wrote {out_path}")
    if valid_e:
        valid_e = np.array(valid_e)
        print(f"  valid: {len(valid_e)}/{len(rows)}")
        print(f"  energy_per_atom: mean={valid_e.mean():.3f}  median={np.median(valid_e):.3f}  min={valid_e.min():.3f}  max={valid_e.max():.3f}  eV")
        # Hit-rate proxies (CHGNet single-point E/atom against a few thresholds)
        for thr in (-3.0, -4.0, -5.0, -6.0):
            print(f"  hit-rate (E/atom < {thr:5.1f}): {(valid_e < thr).mean():.3%}")


if __name__ == "__main__":
    main()
