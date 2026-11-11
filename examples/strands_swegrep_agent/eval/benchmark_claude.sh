#!/usr/bin/env bash
# Benchmark Claude on Amazon Bedrock as the agent's model:
#   bash eval/benchmark_claude.sh [MODEL_ID]    (default global.anthropic.claude-sonnet-5-5)
# Scores: evals/<model>/eval/RESULTS.txt, e.g. evals/claude-sonnet-5-5/eval/RESULTS.txt. Rerunning resumes.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-global.anthropic.claude-sonnet-5-5}"
OUT="evals/${MODEL##*.}/eval"
run() { uv run --frozen --extra claude "$@"; }  # --extra adds the anthropic SDK to the environment
for benchmark in swe-explore contextbench-lite; do
  [[ -f "$OUT/$benchmark/COMPLETED" ]] ||
    run eval/claude_bench.py --model "$MODEL" --data "data/benchmarks/$benchmark.jsonl" --out "$OUT/$benchmark"
done
run eval/score.py "$OUT" | tee "$OUT/RESULTS.txt"
