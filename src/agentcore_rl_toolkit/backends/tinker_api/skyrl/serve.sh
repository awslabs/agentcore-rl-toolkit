#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
integration_dir="$PWD"
# Existing EC2 defaults remain available; externally managed environments can
# supply all three paths without sourcing the example's machine-specific setup.
if [[ -z "${SKYRL_DIR:-}" || -z "${MODEL_DIR:-}" || -z "${SKYRL_STATE_DIR:-}" ]]; then
  source environment.sh
fi
mkdir -p "${TMPDIR:-/tmp}" "$SKYRL_STATE_DIR"
# Prevent two launchers from starting servers against the same database.
exec 9>"$SKYRL_STATE_DIR/endpoint.lock"
flock -n 9 || { echo "An endpoint is already running on this instance." >&2; exit 1; }

checkpoints_base="${SKYRL_CHECKPOINTS_BASE:-$SKYRL_STATE_DIR/checkpoints}"
mode_args=(--external-inference-lora-base "$SKYRL_STATE_DIR/external-lora")
if [[ -n "${SKYRL_EXTERNAL_INFERENCE_URL:-}" ]]; then
  # Trainer-only GPUs; SkyRL's API sends sample() to the external server (e.g. vLLM-Neuron on Trn2).
  # Requires a SkyRL checkout with the trainer-only sampler export (v3) applied.
  : "${SKYRL_BACKEND_CONFIG:?Set the trainer-only FSDP JSON configuration}"
  backend="${SKYRL_TRAINING_BACKEND:-fsdp}"
  mode_args+=(--runtime-role trainer --external-inference-url "$SKYRL_EXTERNAL_INFERENCE_URL")
  # Required when the sampler listens on a private address rather than a forwarded loopback port.
  if [[ -n "${SKYRL_EXTERNAL_INFERENCE_API_KEY:-}" ]]; then
    mode_args+=(--external-inference-api-key "$SKYRL_EXTERNAL_INFERENCE_API_KEY")
  fi
  bind_host="${TINKER_BIND_HOST:-127.0.0.1}"
else
  backend="${SKYRL_TRAINING_BACKEND:-megatron}"
  bind_host="${TINKER_BIND_HOST:-0.0.0.0}"
fi
backend_config="$(cat "${SKYRL_BACKEND_CONFIG:-$integration_dir/backend_config.json}")"

cd "$SKYRL_DIR"
exec uv run --frozen --extra tinker --extra "$backend" --extra aws \
  -m skyrl.tinker.api \
  --base-model "$MODEL_DIR" --backend "$backend" --host "$bind_host" \
  --port "${TINKER_PORT:-18080}" \
  --database-url "sqlite:///$SKYRL_STATE_DIR/tinker.db" \
  --checkpoints-base "$checkpoints_base" \
  "${mode_args[@]}" --backend-config "$backend_config" "$@"
