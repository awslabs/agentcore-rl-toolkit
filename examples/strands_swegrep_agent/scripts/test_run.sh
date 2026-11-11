#!/usr/bin/env bash
# One-step end-to-end test, in the foreground (a few minutes): 2 tasks x 4 rollouts through the deployed
# agent, one policy update and a sampler weight sync. Passes if every rollout was scored.
#   bash scripts/test_run.sh [recipe]
# Needs the SkyRL endpoint running (bash scripts/endpoint.sh start). Log: logs/test-run-*.log
set -uo pipefail
cd "$(dirname "$0")/.." || exit
EXP="test-run-$(date +%Y%m%d-%H%M%S)"
export EXP
mkdir -p logs
.venv/bin/python -u swegrep_train.py --test-run "$@" 2>&1 | tee "logs/$EXP.log"
m="$(grep -h '"rollout/failed_episodes"' "runs/$EXP/metrics.jsonl" 2>/dev/null | tail -1)"
if [[ -n "$m" ]] && jq -e '."rollout/failed_episodes" == 0 and ."rollout/scored_episodes" > 0' <<<"$m" >/dev/null; then
  echo "PASS: $EXP scored all $(jq '."rollout/episodes"' <<<"$m") rollouts"
else
  echo "FAIL: $EXP${m:+ $(jq -c '{episodes: ."rollout/episodes", scored: ."rollout/scored_episodes", failed: ."rollout/failed_episodes"}' <<<"$m")}; see logs/$EXP.log" >&2
  exit 1
fi
