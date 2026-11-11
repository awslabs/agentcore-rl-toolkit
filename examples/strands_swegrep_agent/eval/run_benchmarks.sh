#!/usr/bin/env bash
# Benchmark a model through the deployed agent on SWE-Explore (848 tasks) and ContextBench Lite (500):
#   bash eval/run_benchmarks.sh MODEL_DIR NAME
# Serves MODEL_DIR with vLLM on all 8 GPUs (endpoint/serve_eval.sh), runs both benchmarks, stops the
# server and scores them. Scores: evals/NAME/eval/RESULTS.txt. Rerunning resumes.
set -euo pipefail
MODEL_DIR="$(realpath "$1")" NAME="$2"
cd "$(dirname "$0")/.."
OUT="evals/$NAME/eval" SERVER="swegrep-eval-$NAME"
mkdir -p "$OUT" logs
server_running() { [[ -n "$(docker ps -q --filter "name=^$SERVER\$")" ]]; }

if [[ ! -f "$OUT/swe-explore/COMPLETED" || ! -f "$OUT/contextbench-lite/COMPLETED" ]]; then
  server_running || MODEL_DIR="$MODEL_DIR" SKYRL_CONTAINER="$SERVER" DETACH_LOG="logs/$SERVER.log" \
    bash endpoint/docker.sh bash serve_eval.sh "$NAME"
  echo "waiting for vLLM (log: logs/$SERVER.log)"
  until curl -sf localhost:8399/health >/dev/null; do
    server_running || { echo "FATAL: vLLM exited; see logs/$SERVER.log" >&2; exit 1; }
    sleep 10
  done
  for benchmark in swe-explore contextbench-lite; do
    [[ -f "$OUT/$benchmark/COMPLETED" ]] ||
      .venv/bin/python eval/full_benchmark.py --data "data/benchmarks/$benchmark.jsonl" --out "$OUT/$benchmark" --model "$NAME"
  done
  docker stop -t 60 "$SERVER" >/dev/null
fi
.venv/bin/python eval/score.py "$OUT" | tee "$OUT/RESULTS.txt"
