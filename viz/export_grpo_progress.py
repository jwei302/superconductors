"""Read GRPO `train_progress.jsonl` logs for one or more runs and write a compact
multi-run JSON for the GitHub-Pages reward-curve charts (docs/data/grpo_progress.json).

Scans every hydra_jobs/singlerun/*/grpo_*/ run dir, so both the stable run
(grpo_beeonly) and the high-LR run (grpo_bee_hilr) show up as separate series.
Run by the monitoring loop every ~20 min; safe to run anytime.
"""
import glob
import json
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "docs" / "data" / "grpo_progress.json"

LABELS = {"grpo_beeonly": "stable", "grpo_bee_hilr": "high-LR"}


def lr_for(run_dir: str):
    cfg = Path(run_dir) / "rl_config.yaml"
    if cfg.exists():
        m = re.search(r"lr:\s*([0-9.eE+-]+)", cfg.read_text())
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def downsample(rows, k=500):
    if len(rows) <= k:
        return rows
    step = len(rows) / k
    return [rows[int(i * step)] for i in range(k)]


def load_run(progress_path: str) -> dict:
    run_dir = os.path.dirname(progress_path)
    key = os.path.basename(run_dir)
    rows = []
    with open(progress_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    finite = downsample([r for r in rows if r.get("finite", True)])
    targets = [r.get("target_tc") for r in finite if r.get("target_tc") == r.get("target_tc")]
    lr = lr_for(run_dir)
    name = LABELS.get(key, key)
    if lr is not None:
        name = f"{name} (lr {lr:g})"
    return {
        "key": key,
        "name": name,
        "lr": lr,
        "n": len(finite),
        "target": targets[-1] if targets else None,
        "updates": [r.get("update") for r in finite],
        "reward": [r.get("reward_bee_potential") for r in finite],
        "tcad": [r.get("tcad_pred") for r in finite],
        "tcad_max": [r.get("tcad_max") for r in finite],
        "loss": [r.get("loss") for r in finite],
    }


def main():
    paths = glob.glob(str(PROJECT_ROOT / "hydra_jobs" / "singlerun" / "*" / "grpo_*" / "train_progress.jsonl"))
    # de-dupe by run-name, keeping the most recent file per name
    by_key = {}
    for p in sorted(paths, key=os.path.getmtime):
        by_key[os.path.basename(os.path.dirname(p))] = p
    runs = [load_run(p) for p in by_key.values()]
    runs = [r for r in runs if r["n"] > 0]
    runs.sort(key=lambda r: (r["key"] != "grpo_beeonly", r["key"]))  # stable first
    out = {"status": "running" if runs else "waiting", "runs": runs}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {OUT.name}: {len(runs)} run(s): " +
          ", ".join(f"{r['key']}={r['n']}pts" for r in runs))


if __name__ == "__main__":
    main()
