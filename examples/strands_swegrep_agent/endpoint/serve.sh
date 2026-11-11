#!/usr/bin/env bash
# Serve the SkyRL Tinker API for Qwen3.5-9B on all 8 local GPUs. Run inside the container:
#   bash endpoint/docker.sh bash serve.sh        (or: bash scripts/endpoint.sh start)
# Adapted from PR #158's serve.sh: 65,536-token context (backend_config.json) and local paths.
# No authentication or TLS: binds 127.0.0.1 by default (the trainer client runs on this host).
set -euo pipefail
cd "$(dirname "$0")"
source environment.sh
mkdir -p "$TMPDIR" "$SKYRL_STATE_DIR"
# Prevent two launchers from starting servers against the same database.
exec 9>"$SKYRL_STATE_DIR/endpoint.lock"
flock -n 9 || { echo "An endpoint is already running on this instance." >&2; exit 1; }
[ -f "$MODEL_DIR/config.json" ] || { echo "FATAL: $MODEL_DIR missing" >&2; exit 1; }

BACKEND_CONFIG=$(sed "s#\${SKYRL_STATE_DIR}#$SKYRL_STATE_DIR#g" "${BACKEND_CONFIG_FILE:-backend_config.json}")

cd "$SKYRL_DIR"
exec uv run --frozen --extra tinker --extra megatron --extra aws \
  -m skyrl.tinker.api \
  --base-model "$MODEL_DIR" --backend megatron --host "${TINKER_HOST:-127.0.0.1}" \
  --port "$TINKER_PORT" \
  --database-url "sqlite:///$SKYRL_STATE_DIR/tinker.db" \
  --checkpoints-base "$SKYRL_STATE_DIR/checkpoints" \
  --external-inference-lora-base "$SKYRL_STATE_DIR/external-lora" \
  --backend-config "$BACKEND_CONFIG"
