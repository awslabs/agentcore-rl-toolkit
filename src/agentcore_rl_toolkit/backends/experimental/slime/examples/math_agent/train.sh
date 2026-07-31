#!/bin/bash
# Train the strands math agent (examples/strands_math_agent/rl_app.py, deployed to
# ACR first) with slime GRPO: our rollout function submits tasks to ACR, captures
# per-turn token ids + logprobs via the rollout gateway, and feeds Megatron.
#
# Config: config.yaml (ACR ARN + tunables; cp from config.yaml.example),
# .wandb.env (optional wandb creds), env vars below.
#
# Usage:
#   export SLIME_DIR=/root/slime \
#          MODEL_DIR=/path/to/Qwen3-0.6B \
#          TRAIN_DATA_PATH=/path/to/gsm8k_tiny.jsonl \
#          MODEL_TYPE=qwen3-0.6B
#   bash train.sh
set -euo pipefail

# === Paths (set these via env) ===
SLIME_DIR="${SLIME_DIR:?Set SLIME_DIR (path to the slime repo, e.g. /root/slime)}"
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR (path to the HF model checkpoint)}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:?Set TRAIN_DATA_PATH (path to the training JSONL)}"
VAL_DATA_PATH="${VAL_DATA_PATH:-${TRAIN_DATA_PATH}}"
MODEL_TYPE="${MODEL_TYPE:?Set MODEL_TYPE (slime model-arch name, e.g. qwen3-0.6B)}"
CONFIG="${CONFIG:-$(dirname "$0")/config.yaml}"

# Set your cuda path. CUDA 13 only — cu12 is not supported.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"

# Checkpoint output dir (cleared at start; comment the rm to resume).
CKPTS_DIR="${CKPTS_DIR:-checkpoints/exp_agentcore_grpo}"
rm -rf "${CKPTS_DIR}"

# GPUs on this node; with --colocate this is also the train+rollout pool.
NUM_GPUS="${NUM_GPUS:-8}"

# Optional wandb creds (WANDB_API_KEY / _ENTITY / _PROJECT) — never commit real keys.
[ -f "$(dirname "$0")/.wandb.env" ] && source "$(dirname "$0")/.wandb.env"

# === Setup ===
pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 3

# The cu12 decoy below only works for processes that inherit LD_LIBRARY_PATH at exec
# time (the loader reads it once at startup), so a leftover Ray cluster would run
# actors that never see it. Fail loudly now, not mid-training on "Multiple libcudart
# libraries found".
if pgrep -f 'raylet|gcs_server' >/dev/null 2>&1; then
  echo "ERROR: Ray is still running after 'ray stop --force'." >&2
  echo "  Kill it before rerunning: pkill -9 -f 'raylet|gcs_server|ray::'" >&2
  exit 1
fi

# slime/ray/actor_group.py hardcodes the cu12-named torch_memory_saver preload .so,
# but a cu13 build ships *_cu13.abi3.so. Bridge the filename (idempotent).
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
LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vE '^/usr/local/cuda(-[0-9.]+)?(/|$)' | paste -sd ':' -)"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Block cu12's libcudart from loading alongside cu13: TE 2.11.0's vendored cuDNN
# Frontend probes both and raises "Multiple libcudart libraries found". Scrubbing
# LD_LIBRARY_PATH isn't enough — /etc/ld.so.cache still resolves cu12. An empty decoy
# at the FRONT of the path makes dlopen fail on that name ("file too short") without
# falling back to the cache. (Newer frontends just warn and honor
# CUDNN_FRONTEND_CUDART_LIB_NAME; TE's vendored copy does neither.)
#
# Path is fixed, not mktemp'd: Ray workers resolve it lazily (first cu12 pull is at
# fused_attn_fwd), so it must still exist whenever an actor gets there, across reruns.
DECOY_DIR="${DECOY_DIR:-/tmp/cudart-decoy-cu13-${USER}}"
mkdir -p "${DECOY_DIR}"
: > "${DECOY_DIR}/libcudart.so.12"
export LD_LIBRARY_PATH="${DECOY_DIR}:${LD_LIBRARY_PATH}"

# Honored by the standalone cudnn-frontend (>=1.26), which then skips the probe.
export CUDNN_FRONTEND_CUDART_LIB_NAME=libcudart.so.13

# Pin cuDNN to the venv wheel. TE loads the main libcudnn by absolute path, globbing
# ${CUDNN_HOME}|${CUDNN_PATH}|${CUDA_HOME} in that order, while its sublibraries
# (libcudnn_graph/_engines_*) come from LD_LIBRARY_PATH — i.e. the wheel. Without
# these vars, CUDA_HOME's own cuDNN wins and the main/sublib version mismatch kills
# fused attention with CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED.
CUDNN_HOME="$(python -c "import sysconfig, os; print(os.path.join(sysconfig.get_path('purelib'), 'nvidia', 'cudnn'))")"
export CUDNN_HOME
export CUDNN_PATH="${CUDNN_HOME}"

export PYTHONUNBUFFERED=1
ray start --head --num-gpus ${NUM_GPUS} --disable-usage-stats

# Source model architecture args (populates MODEL_ARGS)
source ${SLIME_DIR}/scripts/models/${MODEL_TYPE}.sh

# === Launch training ===
export no_proxy=127.0.0.1

# Env forwarded to every Ray worker (ACR ARN + bucket come from config.yaml instead).
# WANDB_API_KEY is appended only when set, to avoid injecting an empty value.
RUNTIME_ENV_JSON="{\"env_vars\": {\"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\", \"CUDA_HOME\": \"${CUDA_HOME}\", \"CUDNN_HOME\": \"${CUDNN_HOME}\", \"CUDNN_PATH\": \"${CUDNN_PATH}\", \"CUDNN_FRONTEND_CUDART_LIB_NAME\": \"${CUDNN_FRONTEND_CUDART_LIB_NAME}\", \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH}\"${WANDB_API_KEY:+, \"WANDB_API_KEY\": \"${WANDB_API_KEY}\"}}}"

# slime gives the Megatron train actors their own runtime_env.env_vars
# (slime/ray/actor_group.py), which drops CUDA_HOME / CUDNN_* / LD_LIBRARY_PATH — those
# workers would fall back to the cluster CUDA and hit the failures above.
# --train-env-vars is merged into that actor env, so re-pin the paths there.
TRAIN_ENV_VARS_JSON="{\"CUDA_HOME\": \"${CUDA_HOME}\", \"CUDNN_HOME\": \"${CUDNN_HOME}\", \"CUDNN_PATH\": \"${CUDNN_PATH}\", \"CUDNN_FRONTEND_CUDART_LIB_NAME\": \"${CUDNN_FRONTEND_CUDART_LIB_NAME}\", \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH}\"}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 ${SLIME_DIR}/train.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint ${MODEL_DIR} \
  --ref-load ${MODEL_DIR} \
  --prompt-data ${TRAIN_DATA_PATH} \
  --eval-prompt-data gsm8k ${VAL_DATA_PATH} \
  --num-rollout 100 \
  --tensor-model-parallel-size 2 \
  --rollout-num-gpus-per-engine 2 \
  --input-key prompt \
  --rollout-batch-size 64 \
  --n-samples-per-prompt 4 \
  --num-steps-per-rollout 1 \
  --sglang-context-length 14336 \
  --max-tokens-per-gpu 14336 \
  --rollout-max-response-len 2048 \
  --eval-max-response-len 2048 \
  --rollout-temperature 1.0 \
  --eval-interval 10 \
  --eval-input-key prompt \
  --n-samples-per-eval-prompt 1 \
  --eval-temperature 0.0 \
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
  --sglang-tool-call-parser qwen \
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
      agentcore_rl_toolkit.backends.experimental.slime.integration.rollout.generate_rollout \
  --custom-reward-post-process-path \
      agentcore_rl_toolkit.backends.experimental.slime.integration.rewards.normalize_episode_rewards \
  --custom-config-path ${CONFIG} \
  --use-dynamic-batch-size \
  --save ${CKPTS_DIR} \
  --save-interval 100 \
  --save-hf ${CKPTS_DIR}/hf/{rollout_id} \
  ${WANDB_API_KEY:+--use-wandb --wandb-project ${WANDB_PROJECT:-slime-art} --wandb-group gsm8k-slime-grpo}
