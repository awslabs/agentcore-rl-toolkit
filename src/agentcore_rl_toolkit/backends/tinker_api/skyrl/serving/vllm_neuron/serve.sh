#!/usr/bin/env bash
# Start stock `vllm serve` on Neuron as SkyRL's external sampler, with LoRA refresh (README.md).
#
# Required: ART_CHECKPOINT_DIR (immutable local base checkpoint), ART_WEIGHT_REFRESH_ROOT
#   (existing directory for merged checkpoints), ART_WEIGHT_REFRESH_CHECKPOINTS (the trainer's
#   SKYRL_CHECKPOINTS_BASE, s3://bucket/prefix).
# Optional: ART_WEIGHT_REFRESH_S3_REGION, ART_INSTANCE_TYPE (trn2.48xlarge),
#   ART_SERVING_PROFILE (qwen3.6-27b), ART_MODEL_VERSION (initial), VLLM_NEURON_BIND_HOST
#   (127.0.0.1; a private address requires VLLM_API_KEY), VLLM_NEURON_PORT (8000),
#   VLLM_API_KEY (read by vLLM; the trainer sends it), NEURON_LIBTORCH_CACHE_ROOT.
# Additional arguments are passed to `vllm serve`.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
: "${ART_CHECKPOINT_DIR:?Set ART_CHECKPOINT_DIR to the local base checkpoint}"
: "${ART_WEIGHT_REFRESH_ROOT:?Set ART_WEIGHT_REFRESH_ROOT to an existing directory for merged checkpoints}"
: "${ART_WEIGHT_REFRESH_CHECKPOINTS:?Set ART_WEIGHT_REFRESH_CHECKPOINTS to the SKYRL_CHECKPOINTS_BASE of the trainer}"

host="${VLLM_NEURON_BIND_HOST:-127.0.0.1}"
if [[ "$host" != 127.0.0.1 && "$host" != ::1 && -z "${VLLM_API_KEY:-}" ]]; then
  echo "Binding to $host requires VLLM_API_KEY (also set as the trainer's SKYRL_EXTERNAL_INFERENCE_API_KEY)" >&2
  exit 1
fi
instance_type="${ART_INSTANCE_TYPE:-trn2.48xlarge}"
profile="${ART_SERVING_PROFILE:-qwen3.6-27b}"
config="$here/config/$instance_type/$profile.yaml"
[[ -f "$config" ]] || { echo "No serving profile $config" >&2; exit 1; }
# Model- and instance-specific Neuron runtime settings live beside the vLLM profile.
if [[ -f "${config%.yaml}.env" ]]; then source "${config%.yaml}.env"; fi
# Settings left over from other Neuron launches would conflict with the profile's devices.
unset NEURON_RT_VISIBLE_CORES NEURON_RT_ROOT_COMM_ID NKI_SIMULATOR VLLM_NEURON_BACKEND
# Keep the compile cache stable across restarts and refreshes; compiling takes hours.
export NEURON_LIBTORCH_CACHE_ROOT="${NEURON_LIBTORCH_CACHE_ROOT:-$HOME/.cache/art-inference/$instance_type/$profile}"

# The served base is the one every LoRA update is merged into.
export ART_WEIGHT_REFRESH_BASE="$ART_CHECKPOINT_DIR" ART_WEIGHT_REFRESH_ROOT ART_WEIGHT_REFRESH_CHECKPOINTS
# vLLM imports both plugins by name: the middleware in the API server, the extension in each worker.
# One API server process: refresh state (served version, in-flight refresh) lives in its middleware.
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
exec vllm serve "$ART_CHECKPOINT_DIR" \
  --config "$config" \
  --served-model-name "${ART_MODEL_VERSION:-initial}" \
  --host "$host" --port "${VLLM_NEURON_PORT:-8000}" \
  --middleware server.ModelResolver --worker-extension-cls server.WeightRefreshWorker \
  --api-server-count 1 \
  "$@"
