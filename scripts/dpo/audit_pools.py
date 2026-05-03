"""Audit β=0.1 (and the rest of the sweep) for reward-hacking signatures:
  1. Chemistry distribution shift — does β=0.1 collapse to a few chemsys?
  2. Structure uniqueness — does β=0.1 mode-collapse to a few prototypes?
  3. Element frequency shift — does it bias toward heavy / strongly-bonded elements?

All inputs are the cached scored JSONs and CIFs already on disk. No GPU needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def chemsys_of(formula: str) -> str:
    return "-".join(sorted({el.symbol for el in Composition(formula).elements}))


def canonical_hash(struct: Structure, sg_tol: float = 0.1) -> str:
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


def hash_one(p: Path) -> str | None:
    try:
        s = Structure.from_file(str(p))
        return canonical_hash(s)
    except Exception:
        return None


def uniqueness(cif_dir: Path, n_workers: int = 8) -> dict:
    paths = sorted(cif_dir.glob("*.cif"))
    if not paths:
        return {"n": 0, "unique": 0, "rate": 0.0, "top5_collisions": []}
    with mp.Pool(n_workers) as pool:
        hashes = pool.map(hash_one, paths)
    hashes = [h for h in hashes if h is not None]
    counts = Counter(hashes)
    return {
        "n": len(paths),
        "n_hashable": len(hashes),
        "unique": len(counts),
        "rate": len(counts) / max(len(paths), 1),
        "top5_collisions": counts.most_common(5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dpo_artifacts/audit_summary.json")
    ap.add_argument("--n_workers", type=int, default=8)
    args = ap.parse_args()

    pools = [
        ("baseline",   Path("dpo_artifacts/day1_baseline_pool/scores_y10k.json"),
                       Path("dpo_artifacts/day1_baseline_pool/cifs_y10k")),
        ("dpo_b0.1",   Path("dpo_artifacts/eval/scores_dpo_b0.1.json"),
                       Path("dpo_artifacts/eval/cifs_dpo_b0.1")),
        ("dpo_b0.5",   Path("dpo_artifacts/eval/scores_dpo_b0.5.json"),
                       Path("dpo_artifacts/eval/cifs_dpo_b0.5")),
        ("dpo_b1",     Path("dpo_artifacts/eval/scores_dpo_b1.json"),
                       Path("dpo_artifacts/eval/cifs_dpo_b1")),
        ("dpo_b5",     Path("dpo_artifacts/eval/scores_dpo_b5.json"),
                       Path("dpo_artifacts/eval/cifs_dpo_b5")),
        ("dpo_b25",    Path("dpo_artifacts/eval/scores_dpo_b25.json"),
                       Path("dpo_artifacts/eval/cifs_dpo_b25")),
    ]

    summary = {}
    for name, scores_p, cif_dir in pools:
        rows = json.loads(scores_p.read_text())
        valid_rows = [r for r in rows if r["valid"]]

        chemsys_counts = Counter(chemsys_of(r["formula"]) for r in valid_rows)
        formula_counts = Counter(r["formula"] for r in valid_rows)
        element_counts = Counter()
        for r in valid_rows:
            for el in Composition(r["formula"]).elements:
                element_counts[el.symbol] += 1

        print(f"=== {name} ===")
        print(f"  pool size: {len(rows)}; valid: {len(valid_rows)}")
        print(f"  unique chemsys: {len(chemsys_counts)}  (top-10 share: {sum(c for _, c in chemsys_counts.most_common(10))/len(valid_rows):.1%})")
        print(f"  unique formulas: {len(formula_counts)}  (top-10 share: {sum(c for _, c in formula_counts.most_common(10))/len(valid_rows):.1%})")
        print(f"  top-10 chemsys: {chemsys_counts.most_common(10)}")
        print(f"  top-10 formulas: {formula_counts.most_common(10)}")
        print(f"  most common elements: {element_counts.most_common(15)}")

        print(f"  computing structure-hash uniqueness from {cif_dir} ...")
        uniq = uniqueness(cif_dir, n_workers=args.n_workers)
        print(f"    hashable: {uniq['n_hashable']}/{uniq['n']}  unique: {uniq['unique']}  rate: {uniq['rate']:.3%}")
        print(f"    top-5 hash collisions: {uniq['top5_collisions']}")
        print()

        summary[name] = {
            "pool_size": len(rows),
            "valid": len(valid_rows),
            "n_chemsys": len(chemsys_counts),
            "top10_chemsys_share": sum(c for _, c in chemsys_counts.most_common(10)) / len(valid_rows),
            "n_formulas": len(formula_counts),
            "top10_formulas_share": sum(c for _, c in formula_counts.most_common(10)) / len(valid_rows),
            "chemsys_top20": chemsys_counts.most_common(20),
            "formula_top20": formula_counts.most_common(20),
            "elements_top20": element_counts.most_common(20),
            "uniqueness": uniq,
        }

    Path(args.out).write_text(json.dumps(summary, indent=1, default=str))
    print(f"[done] wrote {args.out}")

    # Side-by-side summary table
    print()
    print("===== summary =====")
    print(f"{'pool':<12} {'valid':>6} {'#chemsys':>9} {'top10%cs':>9} {'#formula':>9} {'top10%fm':>9} {'#unique':>8} {'uniq%':>6}")
    print("-" * 80)
    for name, s in summary.items():
        print(f"{name:<12} {s['valid']:>6} {s['n_chemsys']:>9} {s['top10_chemsys_share']:>8.1%} {s['n_formulas']:>9} {s['top10_formulas_share']:>8.1%} {s['uniqueness']['unique']:>8} {s['uniqueness']['rate']:>5.1%}")


if __name__ == "__main__":
    main()
