#!/usr/bin/env bash
# Manage the local SkyRL Tinker endpoint: the background `skyrl-endpoint` container, logging to
# logs/endpoint.log.
#   bash scripts/endpoint.sh setup|start|stop|status|logs
# setup: download the model (host HF cache), build the Ubuntu 22.04 + CUDA 13.0 image and install SkyRL
#        (idempotent, ~10 min first time).
# start: serve Qwen3.5-9B on all 8 GPUs at 127.0.0.1:$TINKER_PORT (default 18080). GPU workers initialise
#        lazily when the first training client connects (several minutes).
set -euo pipefail
cd "$(dirname "$0")/.."
CONTAINER=skyrl-endpoint
running() { [ -n "$(docker ps -q --filter "name=^$CONTAINER\$")" ]; }
# The trainer's port: the environment, else .env (swegrep_train.py and export_adapter.py load it), else 18080.
if [ -z "${TINKER_PORT:-}" ] && [ -f .env ]; then
  TINKER_PORT="$(grep -E '^(export +)?TINKER_PORT=' .env | tail -1 | grep -oE '[0-9]+' | head -1 || true)"
fi
export TINKER_PORT="${TINKER_PORT:-18080}"
LOG="$PWD/logs/endpoint.log"
mkdir -p logs
case "${1:-status}" in
  setup)
    # Into the host's Hugging Face cache, which docker.sh mounts read-only.
    # hf download can hang on a large file without failing; a rerun resumes it, so bound each attempt.
    read -r MODEL_ID MODEL_REVISION < <(source endpoint/environment.sh && echo "$MODEL_ID $MODEL_REVISION")
    for attempt in 1 2 3 4 5 6; do
      timeout 600 .venv/bin/hf download "$MODEL_ID" --revision "$MODEL_REVISION" >/dev/null && break
      [ "$attempt" -lt 6 ] || { echo "FATAL: model download failed after 6 attempts" >&2; exit 1; }
      echo "model download attempt $attempt did not finish; retrying" >&2
    done
    bash endpoint/docker.sh build && bash endpoint/docker.sh bash setup.sh ;;
  start)
    if running; then
      curl -sf "localhost:$TINKER_PORT/api/v1/healthz" >/dev/null && { echo "$CONTAINER already running on port $TINKER_PORT"; exit 0; }
      echo "FATAL: $CONTAINER is running but not healthy on port $TINKER_PORT (started on another port?); see $LOG" >&2; exit 1
    fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "$used" -lt 1000 ] || { echo "FATAL: GPUs busy (${used} MiB)" >&2; exit 1; }
    SKYRL_CONTAINER="$CONTAINER" DETACH_LOG="$LOG" bash endpoint/docker.sh bash serve.sh
    for _ in $(seq 60); do
      curl -sf "localhost:$TINKER_PORT/api/v1/healthz" >/dev/null && { echo "endpoint up"; exit 0; }
      running || break
      sleep 5
    done
    echo "FATAL: endpoint not healthy after 5 min, or its container exited; see $LOG" >&2; exit 1 ;;
  stop) docker stop -t 60 "$CONTAINER" >/dev/null 2>&1 || true ;;
  status)
    if running; then echo "$CONTAINER running"; else echo "$CONTAINER not running"; fi
    curl -s "localhost:$TINKER_PORT/api/v1/healthz" || true; echo ;;
  logs) grep -v uvicorn.access "$LOG" | tail -50 ;;
  *) echo "usage: $0 setup|start|stop|status|logs" >&2; exit 2 ;;
esac
