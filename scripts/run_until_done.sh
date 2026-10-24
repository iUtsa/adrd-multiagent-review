#!/bin/bash
# Run a config again and again until every article is finished, waiting between tries for free-tier
# daily limits to reset (or for a missing API key to be added). Finished calls come from the cache,
# so each try only does the work that is still open.
#
#   scripts/run_until_done.sh configs/runs/PROF_search_free.yaml            # all articles
#   scripts/run_until_done.sh configs/runs/PROF_search_free.yaml --sample 100
#   RUNNER=adrd_review.phase2 scripts/run_until_done.sh configs/runs/PHASE2_independent.yaml   # phase 2
#
# Settings (environment variables): TRIES (default 240), SLEEP_SECONDS between tries (default 3600 = 1 h),
# RUNNER (default adrd_review.graph.runner; adrd_review.phase2 for phase 2).
# Log: data/logs/until_done_<name>.log. Progress: python -m adrd_review.eval.flow <latest run folder>
# Stop it with:  pkill -f run_until_done.sh ; pkill -f "adrd_review.graph.runner"
cd "$(dirname "$0")/.." || exit 1
CONFIG="$1"; shift
NAME="$(basename "$CONFIG" .yaml)"
LOG="data/logs/until_done_${NAME}.log"
TRIES="${TRIES:-240}"
SLEEP_SECONDS="${SLEEP_SECONDS:-3600}"
RUNNER="${RUNNER:-adrd_review.graph.runner}"
mkdir -p data/logs
for try in $(seq 1 "$TRIES"); do
  echo "=== try $try: $(date '+%Y-%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python -m "$RUNNER" "$CONFIG" "$@" >> "$LOG" 2>&1
  RUN="$(ls -d data/runs/${NAME}/*/ | tail -1)"
  OPEN="$(.venv/bin/python -c "import json; print(len(json.load(open('${RUN}failures.json'))))" 2>/dev/null || echo "?")"
  echo "=== try $try done $(date '+%H:%M'): $OPEN article(s) still open; run folder $RUN ===" >> "$LOG"
  [ "$RUNNER" = "adrd_review.graph.runner" ] && .venv/bin/python -m adrd_review.eval.flow "$RUN" >> "$LOG" 2>&1
  if [ "$OPEN" = "0" ]; then echo "=== COMPLETE: $RUN ===" >> "$LOG"; exit 0; fi
  sleep "$SLEEP_SECONDS"
done
echo "=== stopped after $TRIES tries ===" >> "$LOG"
