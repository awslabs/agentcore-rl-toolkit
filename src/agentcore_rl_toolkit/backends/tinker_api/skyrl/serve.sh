#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source environment.sh
mkdir -p "$TMPDIR" "$SKYRL_STATE_DIR"
# Prevent two launchers from starting servers against the same database.
exec 9>"$SKYRL_STATE_DIR/endpoint.lock"
flock -n 9 || { echo "An endpoint is already running on this instance." >&2; exit 1; }

BACKEND_CONFIG=$(cat backend_config.json)

cd "$SKYRL_DIR"
exec uv run --frozen --extra tinker --extra megatron --extra aws \
  -m skyrl.tinker.api \
  --base-model "$MODEL_DIR" --backend megatron --host 0.0.0.0 \
  --port "${TINKER_PORT:-18080}" \
  --database-url "sqlite:///$SKYRL_STATE_DIR/tinker.db" \
  --checkpoints-base "$SKYRL_STATE_DIR/checkpoints" \
  --external-inference-lora-base "$SKYRL_STATE_DIR/external-lora" \
  --backend-config "$BACKEND_CONFIG"
