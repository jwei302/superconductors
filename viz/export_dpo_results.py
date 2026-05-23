"""Export the committed DPO β-sweep results to a compact JSON for the website.

Reads the per-structure CHGNet energy scores and the chemistry audit that the DPO
study already produced under artifacts/dpo/, and writes everything the docs page
needs to draw three figures with no runtime dependencies:

  - hit-rate vs β at three baseline thresholds  (+ baseline reference)
  - energy-per-atom distribution shift          (baseline vs each β, as densities)
  - chemistry-diversity audit                   (# unique chemsys, top-10 share)

Output: docs/data/dpo_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DPO = ROOT / "artifacts" / "dpo"
OUT = ROOT / "docs" / "data" / "dpo_results.json"

BETAS = ["0.1", "0.5", "1", "5", "25"]

# Shared histogram support for the distribution-shift figure.
BIN_LO, BIN_HI, NBINS = -13.0, -3.0, 56


def load_pool(path: Path) -> np.ndarray:
    rows = json.loads(path.read_text())
    return np.array([r["energy_per_atom"] for r in rows if r.get("valid")], dtype=float)


def density(e: np.ndarray, edges: np.ndarray) -> list[float]:
    h, _ = np.histogram(e, bins=edges, density=True)
    return [round(float(v), 5) for v in h]


def main() -> None:
    baseline = load_pool(DPO / "day1_baseline_pool" / "scores_y10k.json")
    pools = {b: load_pool(DPO / "eval" / f"scores_dpo_b{b}.json") for b in BETAS}
    audit = json.loads((DPO / "audit_summary.json").read_text())

    T25 = float(np.percentile(baseline, 25))
    T10 = float(np.percentile(baseline, 10))
    T05 = float(np.percentile(baseline, 5))

    edges = np.linspace(BIN_LO, BIN_HI, NBINS + 1)
    centers = [round(float(c), 4) for c in (edges[:-1] + edges[1:]) / 2]

    def pool_stats(e: np.ndarray, audit_key: str) -> dict:
        a = audit[audit_key]
        return {
            "n": int(len(e)),
            "median": round(float(np.median(e)), 4),
            "hr25": round(float((e < T25).mean()), 4),
            "hr10": round(float((e < T10).mean()), 4),
            "hr05": round(float((e < T05).mean()), 4),
            "top500": round(float(np.median(np.sort(e)[:500])), 4),
            "n_chemsys": int(a["n_chemsys"]),
            "top10_share": round(float(a["top10_chemsys_share"]), 4),
            "hist": density(e, edges),
        }

    out = {
        "thresholds": {"T25": round(T25, 4), "T10": round(T10, 4), "T05": round(T05, 4)},
        "bins": {"lo": BIN_LO, "hi": BIN_HI, "centers": centers},
        "baseline": pool_stats(baseline, "baseline"),
        "betas": BETAS,
        "pools": {b: pool_stats(pools[b], f"dpo_b{b}") for b in BETAS},
        # headline result: the strongest β that improves stability *without*
        # chemistry collapse (β=0.1 hits 90% hit-rate but reward-hacks down to
        # 158 chemsys). β=0.5 is the best honest presenting result.
        "primary": "0.5",
        "collapsed": "0.1",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT}")
    print(f"  thresholds T25={T25:.3f} T10={T10:.3f} T05={T05:.3f}")
    for b in BETAS:
        s = out["pools"][b]
        print(f"  β={b:>4}: median={s['median']:.3f}  hr25={s['hr25']:.1%}  "
              f"#chemsys={s['n_chemsys']}  top10={s['top10_share']:.1%}")


if __name__ == "__main__":
    main()
