#!/usr/bin/env bash
# Serve a merged model for benchmarking (vLLM, 8 data-parallel replicas on :8399):
#   MODEL_DIR=<merged model> bash endpoint/docker.sh bash serve_eval.sh <served-name>
# The AgentCore agent calls this server directly, so it listens on HOST (default all interfaces;
# keep 8399 closed to everything but the runtime's VPC in the security group).
# Sampling and thinking are fixed server-side to the training settings (the agent sends none):
# temperature 1, top_p 1, top_k off, 16384 new tokens, enable_thinking on. Without this, vLLM would use
# the model's generation_config.json.
set -euo pipefail
source "$(dirname "$0")/environment.sh"
export VLLM_ENGINE_READY_TIMEOUT_S=1800
exec "$SKYRL_HOME/venv/bin/vllm" serve "$MODEL_DIR" \
 --host "${HOST:-0.0.0.0}" --port 8399 --served-model-name "${1:?usage: serve_eval.sh <served-name>}" \
 --generation-config vllm \
 --override-generation-config '{"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": 16384}' \
 --default-chat-template-kwargs '{"enable_thinking": true}' \
 --language-model-only --gdn-prefill-backend triton \
 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
 --max-model-len 65536 --tensor-parallel-size 1 --data-parallel-size 8 \
 --gpu-memory-utilization 0.8 --max-num-seqs 32 --disable-custom-all-reduce
