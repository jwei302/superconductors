#!/bin/bash
# GRPO run monitor — invoked ~every 20 min during the BEE-only training run.
# Reports job health, refreshes the site's reward-curve JSON from the live
# train_progress.jsonl, and pushes it so the GitHub-Pages charts update.
# Does NOT auto-restart; restart decisions are made by inspecting the .err log.
cd /nfs/roberts/scratch/pi_jks79/jw2933/GuidedMatDiffusion || exit 1
source .venv/bin/activate 2>/dev/null

echo "===== GRPO monitor $(date '+%Y-%m-%d %H:%M:%S') ====="

echo "--- jobs ---"
squeue -u "$USER" -o "%.10i %.14j %.9T %.11M %.12l %.6D %R" 2>/dev/null | grep -iE "grpo|JOBID" || echo "(no GRPO job in queue)"

echo "--- refresh reward-curve JSON ---"
python viz/export_grpo_progress.py 2>/dev/null

LOG=$(ls -t hydra_jobs/singlerun/*/grpo_beeonly/train_progress.jsonl 2>/dev/null | head -1)
echo "--- progress log: ${LOG:-<none>} ---"
if [ -n "$LOG" ]; then
  echo "lines: $(wc -l < "$LOG")"
  echo "latest: $(tail -1 "$LOG")"
fi

echo "--- recent errors (if any) ---"
ERR=$(ls -t hydra_jobs/grpo_bee_*.err 2>/dev/null | head -1)
if [ -n "$ERR" ]; then
  echo "errfile: $ERR"
  grep -iE "error|traceback|exception|cuda|killed|out of memory" "$ERR" | tail -5 || echo "(clean)"
fi

echo "--- push reward curve to site ---"
git add docs/data/grpo_progress.json 2>/dev/null
if git commit -q -m "monitor: refresh GRPO reward curve" 2>/dev/null; then
  git push origin main 2>&1 | tail -1
else
  echo "(no data change to push)"
fi
echo "===== monitor tick done ====="
