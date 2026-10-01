#!/bin/bash
# Finish a run that stopped on a provider's daily limit, unattended.
#   scripts/finish_run.sh configs/runs/E2_groq.yaml
# Re-runs the config (finished calls come from the cache) with patient rate-limit waits, and
# repeats until no article fails or 12 tries have passed. Log: data/logs/finish_<name>.log
# Stop it with:  pkill -f finish_run.sh ; pkill -f "adrd_review.graph.runner"
cd "$(dirname "$0")/.." || exit 1
CONFIG="$1"
NAME="$(basename "$CONFIG" .yaml)"
LOG="data/logs/finish_${NAME}.log"
mkdir -p data/logs
export ADRD_MAX_RATE_WAIT_SECONDS=1500
for try in $(seq 1 12); do
  echo "=== try $try: $(date '+%Y-%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python -m adrd_review.graph.runner "$CONFIG" >> "$LOG" 2>&1
  RUN="$(ls -d data/runs/${NAME}/*/ | tail -1)"
  FAILS="$(.venv/bin/python -c "import json; print(len(json.load(open('${RUN}failures.json'))))")"
  echo "=== try $try done: $FAILS failures, $(date '+%H:%M') ===" >> "$LOG"
  if [ "$FAILS" = "0" ]; then echo "=== COMPLETE: $RUN ===" >> "$LOG"; exit 0; fi
  sleep 1800
done
echo "=== gave up after 12 tries ===" >> "$LOG"
