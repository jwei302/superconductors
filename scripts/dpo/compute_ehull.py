"""Retroactive E_above_hull scoring against the Materials Project convex hull.

Methodology caveat: CHGNet predicts total energies with ~30–50 meV/atom MAE vs
DFT (PBE+U with MP corrections). We use CHGNet energies for our generated
structures and DFT (corrected) energies for the hull entries — so there is a
calibration offset baked in. The 200 meV/atom hit threshold is loose enough to
absorb this noise, but a careful reader should treat the resulting E_hull as an
approximate stability score, not a strict thermodynamic statement.

For each unique chemical system in the input pool:
  1. fetch MP ComputedStructureEntries via MPRester
  2. build a PhaseDiagram
  3. for each generated structure in that chemsys, compute E_above_hull by
     wrapping its CHGNet energy as a ComputedEntry at the structure's composition

Caches per-chemsys hull entries on disk so repeated runs are fast.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

from pymatgen.core import Composition, Structure
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.ext.matproj import MPRester


def chemsys_of(formula: str) -> str:
    return "-".join(sorted(set(el.symbol for el in Composition(formula).elements)))


def load_or_fetch_entries(chemsys: str, mpr: MPRester, cache_dir: Path) -> list[ComputedEntry]:
    cache_path = cache_dir / f"{chemsys}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    print(f"  [mp] fetching {chemsys}...", end="", flush=True)
    t0 = time.time()
    entries = mpr.get_entries_in_chemsys(chemsys.split("-"))
    print(f" {len(entries)} entries in {time.time()-t0:.1f}s")
    with open(cache_path, "wb") as f:
        pickle.dump(entries, f)
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="Input JSON from score_ehull.py")
    ap.add_argument("--cif_dir", required=True, help="Dir of CIFs (need structures for composition)")
    ap.add_argument("--out", required=True, help="Output JSON with e_above_hull added")
    ap.add_argument("--cache_dir", default="dpo_artifacts/score_cache/mp_entries")
    ap.add_argument("--api_key", default=None, help="MP API key; else read MP_API_KEY env")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        print("ERROR: provide --api_key or set MP_API_KEY env var.", file=sys.stderr)
        print("Get a free key at https://next-gen.materialsproject.org/api", file=sys.stderr)
        sys.exit(1)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(Path(args.scores).read_text())
    cif_dir = Path(args.cif_dir)
    print(f"[load] {len(rows)} rows from {args.scores}")

    # Attach Structure for composition lookup; collect chemsys set
    chemsys_to_rows: dict[str, list[dict]] = {}
    for r in rows:
        if not r["valid"] or r["energy_per_atom"] is None or r["formula"] is None:
            continue
        cs = chemsys_of(r["formula"])
        chemsys_to_rows.setdefault(cs, []).append(r)
    print(f"[chemsys] {len(chemsys_to_rows)} unique chemical systems")

    # Fetch entries + build phase diagrams
    pds: dict[str, PhaseDiagram | None] = {}
    with MPRester(api_key) as mpr:
        for i, cs in enumerate(sorted(chemsys_to_rows.keys())):
            entries = load_or_fetch_entries(cs, mpr, cache_dir)
            if len(entries) < len(cs.split("-")):
                # Need at least the elemental references to define a hull
                pds[cs] = None
                print(f"  [skip] {cs}: only {len(entries)} entries (need elementals)")
                continue
            try:
                pds[cs] = PhaseDiagram(entries)
            except Exception as e:
                pds[cs] = None
                print(f"  [skip] {cs}: PhaseDiagram failed: {e}")

    # Compute E_above_hull per row
    n_scored = n_skipped = 0
    for r in rows:
        r["e_above_hull"] = None
        if not r["valid"] or r["energy_per_atom"] is None or r["formula"] is None:
            continue
        cs = chemsys_of(r["formula"])
        pd = pds.get(cs)
        if pd is None:
            n_skipped += 1
            continue
        try:
            comp = Composition(r["formula"])
            n_atoms_per_fu = comp.num_atoms
            # CHGNet returned energy/atom. Multiply by num_atoms in the *cell*
            # to get total energy of that cell.
            total_energy = r["energy_per_atom"] * r["num_atoms"]
            # Build a ComputedEntry with the cell composition
            cell_comp = comp * (r["num_atoms"] / n_atoms_per_fu)
            entry = ComputedEntry(cell_comp, total_energy)
            e_hull = pd.get_e_above_hull(entry)
            r["e_above_hull"] = float(e_hull)
            n_scored += 1
        except Exception as e:
            r["e_above_hull_error"] = str(e)
            n_skipped += 1

    print(f"[done] scored E_hull for {n_scored} structures; skipped {n_skipped}")

    # Summary
    e_hull_vals = [r["e_above_hull"] for r in rows if r.get("e_above_hull") is not None]
    if e_hull_vals:
        import numpy as np
        a = np.array(e_hull_vals)
        print(f"  E_above_hull (eV/atom): median={np.median(a):.3f}  mean={a.mean():.3f}  min={a.min():.3f}  max={a.max():.3f}")
        for thr in (0.05, 0.1, 0.2, 0.5):
            print(f"  hit-rate (E_hull < {int(thr*1000):3d} meV/atom): {(a < thr).mean():.3%}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=1))
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
