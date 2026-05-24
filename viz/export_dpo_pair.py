"""Export one real DPO preference pair + the real σ(z) training trajectory for the
website animation.

  winner = a stable crystal (low CHGNet energy/atom), loser = an unstable one,
  both real samples from the SFT generator's baseline pool, read from their CIFs.
  sigma_z = P(model prefers winner) over training, from the β=0.5 metrics.csv —
  the actual "preference learned" signal the animation plays back.

Output: docs/data/dpo_pair.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
DPO = ROOT / "artifacts" / "dpo"
CIFS = DPO / "day1_baseline_pool" / "cifs_y10k"
METRICS = DPO / "dpo_b0.5" / "dpo" / "version_0" / "metrics.csv"
OUT = ROOT / "docs" / "data" / "dpo_pair.json"

WINNER_NAME = "1737"   # TaRe2W  ~ -12.52 eV/atom  (stable)
LOSER_NAME = "347"     # La4H    ~ -4.70  eV/atom  (unstable)


def crystal(name: str, energy: float) -> dict:
    s = Structure.from_file(CIFS / f"{name}.cif")
    return {
        "formula": s.composition.reduced_formula,
        "energy_per_atom": round(energy, 3),
        "elems": [str(sp.symbol) for sp in s.species],
        "xyz": [[round(float(c), 4) for c in p] for p in s.cart_coords],
        "lattice": [[round(float(c), 4) for c in row] for row in s.lattice.matrix],
    }


def main() -> None:
    scores = {r["name"]: r for r in json.loads((DPO / "day1_baseline_pool" / "scores_y10k.json").read_text())}
    winner = crystal(WINNER_NAME, scores[WINNER_NAME]["energy_per_atom"])
    loser = crystal(LOSER_NAME, scores[LOSER_NAME]["energy_per_atom"])

    df = pd.read_csv(METRICS)
    sub = df[["step", "train/sigma_z_mean"]].dropna()
    steps = sub["step"].values.astype(float)
    sz = sub["train/sigma_z_mean"].values.astype(float)
    # centered rolling mean (correct edge handling) then downsample for the sparkline
    szs = pd.Series(sz).rolling(15, center=True, min_periods=1).mean().values
    idx = np.linspace(0, len(steps) - 1, 60).round().astype(int)
    traj = {
        "steps": [int(steps[i]) for i in idx],
        "values": [round(float(szs[i]), 4) for i in idx],
        "start": round(float(szs[0]), 4),
        "end": round(float(sz[-20:].mean()), 4),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"beta": 0.5, "winner": winner, "loser": loser, "sigma_z": traj}))
    print(f"wrote {OUT}")
    print(f"  winner {winner['formula']} {winner['energy_per_atom']} eV/atom ({len(winner['elems'])} atoms)")
    print(f"  loser  {loser['formula']} {loser['energy_per_atom']} eV/atom ({len(loser['elems'])} atoms)")
    print(f"  sigma_z {traj['start']} -> {traj['end']} over {traj['steps'][-1]} steps")


if __name__ == "__main__":
    main()
