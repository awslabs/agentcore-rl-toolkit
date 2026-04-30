#!/bin/bash
# Lightweight evaluation for strands_math_agent. Skips Megatron/Ray entirely
# and just scores the model behind a standalone SGLang server:
#
#   1. Launch a single SGLang server directly from a HuggingFace checkpoint.
#   2. Wait for /health_generate.
#   3. Run examples/strands_math_agent/evaluate.py against the local server
#      and the deployed ACR agent runtime.
#   4. Tear down the server on exit.
#
# Results are written to examples/strands_math_agent/results/<exp_id>.jsonl.
#
# Required env:
#   MODEL_DIR          HuggingFace checkpoint to serve
#
# ACR/toolkit settings (ARN, bucket, model_id, acr_timeout, acr_tps_limit,
# max_concurrent) are read from config.yaml — the same file train.sh uses.
# Override any of them by exporting the uppercase env-var equivalent
# (ACR_AGENT_RUNTIME_ARN, ACR_S3_BUCKET, MODEL_ID, ACR_TIMEOUT,
# ACR_TPS_LIMIT, MAX_CONCURRENT).
#
# Optional env:
#   SGLANG_HOST        IP to bind SGLang on (default: auto-detected LAN IP).
#                      Must be routable from the ACR VPC — loopback won't work.
#   SGLANG_PORT        Override (default: first free port)
#   TP_SIZE            Tensor-parallel size for SGLang (default: 1)
#   EVAL_SPLIT         test | train (default: test)
#   EVAL_LIMIT         Max examples (default: all)
#   EVAL_EXP_ID        Results filename prefix (default: gsm8k_eval_<date>)
#   EVAL_TEMPERATURE   Sampling temperature (default: 1.0, matches train.sh)
#   EVAL_MAX_TOKENS    Max output tokens per LLM call (default: 1024, matches train.sh)
#
# Usage:
#   MODEL_DIR=/path/to/Qwen2.5-3B-Instruct bash eval.sh

set -euo pipefail

# === Required paths ===
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR (path to HF model checkpoint)}"

# === Optional config ===
TP_SIZE="${TP_SIZE:-1}"
SGLANG_STARTUP_TIMEOUT="${SGLANG_STARTUP_TIMEOUT:-300}"

# SGLang must bind to an IP routable by the ACR agent container (which runs
# in a different VPC), not loopback. Auto-detect the primary LAN IP by asking
# the routing table which interface would egress to 8.8.8.8 — same strategy
# slime uses in slime/utils/http_utils.py. Override with SGLANG_HOST if the
# auto-detection guesses wrong.
if [ -z "${SGLANG_HOST:-}" ]; then
    SGLANG_HOST=$(python3 -c "
import socket
try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(('8.8.8.8', 80))
        print(s.getsockname()[0])
except Exception:
    print('127.0.0.1')
")
fi

EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
EVAL_EXP_ID="${EVAL_EXP_ID:-gsm8k_eval_$(date +%Y%m%d-%H%M%S)}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"          # matches train.sh ROLLOUT_TEMPERATURE
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-1024}"           # matches train.sh MAX_RESPONSE_LEN

# === Read shared settings from config.yaml (env takes precedence) ===
CONFIG="${CONFIG:-$(dirname $0)/config.yaml}"
[ -f "${CONFIG}" ] || { echo "error: ${CONFIG} not found (cp config.yaml.example config.yaml)" >&2; exit 2; }
eval "$(python3 -c '
import os, sys, yaml
c = yaml.safe_load(open(sys.argv[1])) or {}
for key, env in (("agent_runtime_arn", "ACR_AGENT_RUNTIME_ARN"),
                 ("s3_bucket", "ACR_S3_BUCKET"),
                 ("model_id", "MODEL_ID"),
                 ("acr_timeout", "ACR_TIMEOUT"),
                 ("acr_tps_limit", "ACR_TPS_LIMIT"),
                 ("max_concurrent", "MAX_CONCURRENT")):
    # Env var takes precedence; only export from yaml if unset.
    if not os.environ.get(env) and c.get(key) is not None:
        print(f"export {env}={c[key]!r}")
' "${CONFIG}")"
: "${ACR_AGENT_RUNTIME_ARN:?Set agent_runtime_arn in config.yaml (or export ACR_AGENT_RUNTIME_ARN)}"
: "${ACR_S3_BUCKET:?Set s3_bucket in config.yaml (or export ACR_S3_BUCKET)}"
: "${MODEL_ID:?Set model_id in config.yaml (or export MODEL_ID)}"
: "${ACR_TIMEOUT:?Set acr_timeout in config.yaml (or export ACR_TIMEOUT)}"
: "${ACR_TPS_LIMIT:?Set acr_tps_limit in config.yaml (or export ACR_TPS_LIMIT)}"
: "${MAX_CONCURRENT:?Set max_concurrent in config.yaml (or export MAX_CONCURRENT)}"

# === Pick a free port if not specified ===
if [ -z "${SGLANG_PORT:-}" ]; then
    SGLANG_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1])")
fi
SGLANG_BASE_URL="http://${SGLANG_HOST}:${SGLANG_PORT}"

# === Paths ===
EVAL_SCRIPT="$(dirname $0)/../../../../../../examples/strands_math_agent/evaluate.py"
EVAL_SCRIPT="$(python3 -c "import os; print(os.path.abspath('${EVAL_SCRIPT}'))")"
if [ ! -f "${EVAL_SCRIPT}" ]; then
    echo "error: evaluate.py not found at ${EVAL_SCRIPT}" >&2
    exit 1
fi

# === Trap for cleanup ===
SGLANG_PID=""
cleanup() {
    if [ -n "${SGLANG_PID}" ] && kill -0 "${SGLANG_PID}" 2>/dev/null; then
        echo "shutting down SGLang server (pid=${SGLANG_PID})..."
        kill -TERM "${SGLANG_PID}" 2>/dev/null || true
        # Give it a chance to exit gracefully, then force-kill
        for i in $(seq 1 30); do
            kill -0 "${SGLANG_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -9 "${SGLANG_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# === Launch SGLang ===
SGLANG_LOG=$(mktemp -t sglang-eval-XXXXXX.log)
echo "launching SGLang: ${MODEL_DIR} on ${SGLANG_BASE_URL} (log: ${SGLANG_LOG})"

python -m sglang.launch_server \
    --model-path "${MODEL_DIR}" \
    --host "${SGLANG_HOST}" \
    --port "${SGLANG_PORT}" \
    --tp "${TP_SIZE}" \
    --served-model-name "${MODEL_ID}" \
    > "${SGLANG_LOG}" 2>&1 &
SGLANG_PID=$!
echo "SGLang pid: ${SGLANG_PID}"

# === Wait for readiness ===
echo "waiting for SGLang /health_generate (timeout ${SGLANG_STARTUP_TIMEOUT}s)..."
deadline=$(( $(date +%s) + SGLANG_STARTUP_TIMEOUT ))
while true; do
    if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
        echo "error: SGLang server exited during startup. Tail of log:" >&2
        tail -30 "${SGLANG_LOG}" >&2
        exit 2
    fi
    if curl -sf -o /dev/null "${SGLANG_BASE_URL}/health_generate"; then
        echo "SGLang ready."
        break
    fi
    if [ "$(date +%s)" -ge "${deadline}" ]; then
        echo "error: SGLang did not become ready within ${SGLANG_STARTUP_TIMEOUT}s. Tail of log:" >&2
        tail -30 "${SGLANG_LOG}" >&2
        exit 2
    fi
    sleep 2
done

# === Run the evaluation ===
EXTRA_ARGS=()
[ -n "${EVAL_LIMIT}" ] && EXTRA_ARGS+=("--limit" "${EVAL_LIMIT}")

echo "running evaluate.py against ${SGLANG_BASE_URL} (agent ARN: ${ACR_AGENT_RUNTIME_ARN})"
python "${EVAL_SCRIPT}" \
    --agent_arn "${ACR_AGENT_RUNTIME_ARN}" \
    --s3_bucket "${ACR_S3_BUCKET}" \
    --base_url "${SGLANG_BASE_URL}/v1" \
    --model_id "${MODEL_ID}" \
    --exp_id "${EVAL_EXP_ID}" \
    --split "${EVAL_SPLIT}" \
    --max_concurrent "${MAX_CONCURRENT}" \
    --timeout "${ACR_TIMEOUT}" \
    --tps_limit "${ACR_TPS_LIMIT}" \
    --temperature "${EVAL_TEMPERATURE}" \
    --max_tokens "${EVAL_MAX_TOKENS}" \
    "${EXTRA_ARGS[@]}"

echo "eval complete — results in examples/strands_math_agent/results/${EVAL_EXP_ID}.jsonl"
