"""Read the latest GRPO `train_progress.jsonl` and write a compact JSON for the
GitHub-Pages reward-curve chart (docs/data/grpo_progress.json).

Run by the monitoring loop every ~20 min during a GRPO run. Safe to run anytime;
if no run is found it writes a 'waiting' placeholder.
"""
import glob
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "docs" / "data" / "grpo_progress.json"


def find_progress() -> str | None:
    pattern = str(PROJECT_ROOT / "hydra_jobs" / "singlerun" / "*" / "grpo_beeonly" / "train_progress.jsonl")
    cands = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p))
    return cands[-1] if cands else None


def downsample(rows, k=400):
    if len(rows) <= k:
        return rows
    step = len(rows) / k
    return [rows[int(i * step)] for i in range(k)]


def main():
    path = find_progress()
    out = {
        "status": "waiting", "n": 0, "target": None, "source": path,
        "updates": [], "reward": [], "tcad": [], "tcad_max": [], "loss": [],
    }
    if path and os.path.exists(path):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        finite = [r for r in rows if r.get("finite", True)]
        finite = downsample(finite)
        out["updates"] = [r.get("update") for r in finite]
        out["reward"] = [r.get("reward_bee_potential") for r in finite]
        out["tcad"] = [r.get("tcad_pred") for r in finite]
        out["tcad_max"] = [r.get("tcad_max") for r in finite]
        out["loss"] = [r.get("loss") for r in finite]
        targets = [r.get("target_tc") for r in finite if r.get("target_tc") == r.get("target_tc")]
        out["target"] = targets[-1] if targets else None
        out["n"] = len(finite)
        out["status"] = "running" if finite else "starting"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {OUT.name}: {out['n']} points, status={out['status']}, src={path}")


if __name__ == "__main__":
    main()
