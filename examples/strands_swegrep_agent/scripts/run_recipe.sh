#!/usr/bin/env bash
# Train a recipe, export and merge the trained LoRA adapter, and benchmark the merged model:
#   bash scripts/run_recipe.sh RECIPE EXP
# Deploy the agent with the same recipe first (bash cdk/deploy.sh -c recipe=RECIPE). Takes hours.
# Rerunning skips finished stages and resumes training from its last checkpoint.
# Outputs: runs/EXP (training), exports/EXP-adapter, exports/EXP (merged model), evals/EXP/eval/RESULTS.txt.
set -euo pipefail
RECIPE="$1" EXP="$2"
cd "$(dirname "$0")/.."
BASE="$(source endpoint/environment.sh && echo "$MODEL_DIR")"

if [[ ! -f "exports/$EXP-adapter/SOURCE.txt" ]]; then
  bash scripts/endpoint.sh start  # does nothing if it is already up
  EXP="$EXP" .venv/bin/python -u swegrep_train.py "$RECIPE"
  .venv/bin/python scripts/export_adapter.py "runs/$EXP" "exports/$EXP-adapter"
  bash scripts/endpoint.sh stop
fi

[[ -f "exports/$EXP/MERGED_FROM.json" ]] ||
  .venv/bin/python scripts/merge_adapter.py "exports/$EXP-adapter" "exports/$EXP" --base "$BASE"
bash eval/run_benchmarks.sh "exports/$EXP" "$EXP"
