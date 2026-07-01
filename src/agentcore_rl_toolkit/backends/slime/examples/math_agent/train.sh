#!/bin/bash
# Train the strands math agent with slime GRPO using ACR-based rollouts.
#
# The agent (examples/strands_math_agent/rl_app.py) must be deployed to ACR first.
# This script runs slime training with our custom rollout function that submits
# tasks to ACR, collects per-turn token ids + logprobs via rllm-model-gateway
# (use_sglang mode, driven by the integration), and feeds Samples into Megatron
# for GRPO training.
#
# Configuration:
#   config.yaml      — ACR deployment + toolkit tunables
#                      (cp from config.yaml.example; gitignored)
#   .wandb.env       — WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT (optional)
#   env vars below   — paths, model type, cluster + CUDA config (override via env)
#
# Usage: bash train.sh
set -euo pipefail

# === Paths (set these via env) ===
SLIME_DIR="${SLIME_DIR:?Set SLIME_DIR (path to the slime repo)}"
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR (path to the HF model checkpoint)}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:?Set TRAIN_DATA_PATH (path to the training JSONL)}"
VAL_DATA_PATH="${VAL_DATA_PATH:?Set VAL_DATA_PATH (path to the validation JSONL)}"
CONFIG="${CONFIG:-$(dirname "$0")/config.yaml}"

# Set your cuda path
CUDA_HOME=/usr/local/cuda-13.0

# Model architecture args sourced from $SLIME_DIR/scripts/models/${MODEL_TYPE}.sh
MODEL_TYPE="${MODEL_TYPE:?Set MODEL_TYPE (slime model-arch name, e.g. qwen3-4B)}"

# Checkpoint output dir (cleared at start; comment the rm to resume).
CKPTS_DIR="${CKPTS_DIR:-exp_agentcore_grpo}"
rm -rf "${CKPTS_DIR}"

# Sequence / batching budget
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-14336}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-14336}"

# Cluster / run config
NUM_GPUS="${NUM_GPUS:-8}"
TP_SIZE="${TP_SIZE:-2}"                                 # tensor model parallel size
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}" # GPUs per SGLang engine
NUM_ROLLOUT="${NUM_ROLLOUT:-1}"                         # total rollout steps (smoke-test default)
SGLANG_TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-qwen}"  # must match the served model

# Load wandb credentials (optional). Define WANDB_API_KEY (and optionally
# WANDB_ENTITY / WANDB_PROJECT) in .wandb.env — do NOT commit real keys.
[ -f "$(dirname "$0")/.wandb.env" ] && source "$(dirname "$0")/.wandb.env"

# === Setup ===
pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 3

# torch_memory_saver fixup for CUDA 13 envs: slime/ray/actor_group.py looks for
# the cu12-named preload .so, but a cu13 build ships *_cu13.abi3.so. Symlink the
# cu13 binary to the cu12 name so slime finds it. No-op on cu12 envs (the cu12
# .so already exists) and idempotent — only the filename is bridged, not contents.
TMS_SP="$(python -c 'import os, torch_memory_saver; print(os.path.dirname(os.path.dirname(torch_memory_saver.__file__)))' 2>/dev/null || true)"
if [ -n "$TMS_SP" ] \
   && [ ! -e "$TMS_SP/torch_memory_saver_hook_mode_preload_cu12.abi3.so" ] \
   && [ -e "$TMS_SP/torch_memory_saver_hook_mode_preload_cu13.abi3.so" ]; then
  ln -s "$TMS_SP/torch_memory_saver_hook_mode_preload_cu13.abi3.so" \
        "$TMS_SP/torch_memory_saver_hook_mode_preload_cu12.abi3.so"
  echo "[setup] linked tms cu13 .so -> cu12 name for slime compatibility"
fi

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
NVIDIA_LIBS=$(python -c "import sysconfig, os, glob; base=os.path.join(sysconfig.get_path('purelib'), 'nvidia'); print(':'.join(sorted(glob.glob(os.path.join(base, '*', 'lib')))))")
LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vE '^/usr/local/cuda(-[0-9.]+)?/' | paste -sd ':' -)"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

export PYTHONUNBUFFERED=1
ray start --head --num-gpus ${NUM_GPUS} --disable-usage-stats

# Source model architecture args (populates MODEL_ARGS)
source ${SLIME_DIR}/scripts/models/${MODEL_TYPE}.sh

# === Launch training ===
export no_proxy=127.0.0.1

# Env vars forwarded to every Ray worker. (ACR ARN + bucket flow through
# config.yaml, not env.) WANDB_API_KEY is appended only when set, so an unset key
# doesn't inject an empty value into the worker env.
RUNTIME_ENV_JSON="{\"env_vars\": {\"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\", \"CUDA_HOME\": \"${CUDA_HOME}\", \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH}\"${WANDB_API_KEY:+, \"WANDB_API_KEY\": \"${WANDB_API_KEY}\"}}}"

# slime creates the Megatron train actors with their OWN runtime_env.env_vars
# (slime/ray/actor_group.py), which does NOT carry CUDA_HOME / LD_LIBRARY_PATH —
# so a train worker would fall back to the cluster CUDA and TransformerEngine
# could hit "Multiple libcudart libraries found". --train-env-vars is merged into
# that actor env, so pass the pinned CUDA paths through it too.
TRAIN_ENV_VARS_JSON="{\"CUDA_HOME\": \"${CUDA_HOME}\", \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH}\"}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 ${SLIME_DIR}/train.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint ${MODEL_DIR} \
  --ref-load ${MODEL_DIR} \
  --prompt-data ${TRAIN_DATA_PATH} \
  --num-rollout ${NUM_ROLLOUT} \
  --tensor-model-parallel-size ${TP_SIZE} \
  --rollout-num-gpus-per-engine ${ROLLOUT_GPUS_PER_ENGINE} \
  --input-key prompt \
  --rollout-batch-size 64 \
  --n-samples-per-prompt 4 \
  --num-steps-per-rollout 1 \
  --rollout-max-response-len ${MAX_RESPONSE_LENGTH} \
  --rollout-temperature 1.0 \
  --eval-interval 10 \
  --eval-prompt-data gsm8k ${VAL_DATA_PATH} \
  --eval-input-key prompt \
  --n-samples-per-eval-prompt 1 \
  --eval-temperature 0.0 \
  --eval-max-response-len ${MAX_RESPONSE_LENGTH} \
  --advantage-estimator grpo \
  --use-kl-loss \
  --kl-loss-type low_var_kl \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --lr 1e-6 \
  --lr-decay-style constant \
  --optimizer-cpu-offload \
  --overlap-cpu-optimizer-d2h-h2d \
  --use-precision-aware-optimizer \
  --sequence-parallel \
  --sglang-mem-fraction-static 0.6 \
  --sglang-cuda-graph-max-bs 32 \
  --sglang-context-length ${MAX_CONTEXT_LENGTH} \
  --sglang-tool-call-parser ${SGLANG_TOOL_CALL_PARSER} \
  --sglang-log-level warning \
  --sglang-log-level-http warning \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --actor-num-gpus-per-node ${NUM_GPUS} \
  --colocate \
  --train-env-vars "${TRAIN_ENV_VARS_JSON}" \
  --megatron-to-hf-mode bridge \
  --rollout-function-path \
      agentcore_rl_toolkit.backends.slime.integration.rollout.generate_rollout \
  --custom-reward-post-process-path \
      agentcore_rl_toolkit.backends.slime.integration.rewards.normalize_episode_rewards \
  --custom-config-path ${CONFIG} \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU} \
  --save ${CKPTS_DIR} \
  --save-interval 100 \
  --save-hf ${CKPTS_DIR}/hf/{rollout_id} \
  ${WANDB_API_KEY:+--use-wandb --wandb-project ${WANDB_PROJECT:-slime-art} --wandb-group gsm8k-slime-grpo}
