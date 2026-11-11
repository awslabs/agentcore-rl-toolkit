#!/usr/bin/env bash
# Benchmark the base model (the Qwen3.5-9B snapshot that scripts/endpoint.sh setup downloads):
#   bash eval/benchmark_base.sh
# Needs the GPUs free (bash scripts/endpoint.sh stop). Scores: evals/base/eval/RESULTS.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
bash eval/run_benchmarks.sh "$(source endpoint/environment.sh && echo "$MODEL_DIR")" base
