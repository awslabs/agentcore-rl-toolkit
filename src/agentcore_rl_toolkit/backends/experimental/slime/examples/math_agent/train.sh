#!/bin/bash
# GSM8K GRPO with slime + ACR rollout gateway.
set -euo pipefail

# === Paths (set these via env) ===
SLIME_DIR="${SLIME_DIR:?Set SLIME_DIR (path to the slime repo, e.g. /root/slime)}"
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR (path to the HF model checkpoint)}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:?Set TRAIN_DATA_PATH (path to the training JSONL)}"
VAL_DATA_PATH="${VAL_DATA_PATH:-${TRAIN_DATA_PATH}}"
MODEL_TYPE="${MODEL_TYPE:?Set MODEL_TYPE (slime model-arch name, e.g. qwen3-0.6B)}"
CONFIG="${CONFIG:-$(dirname "$0")/config.yaml}"

# CUDA 13 only — cu12 is not supported.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"

# Cleared at start; comment the rm to resume.
CKPTS_DIR="${CKPTS_DIR:-checkpoints/exp_agentcore_grpo}"
rm -rf "${CKPTS_DIR}"

NUM_GPUS="${NUM_GPUS:-8}"

pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 3

# The cu12 libcudart decoy below only works if LD_LIBRARY_PATH is inherited at exec
# time; a leftover Ray cluster runs actors that never see it.
if pgrep -f 'raylet|gcs_server' >/dev/null 2>&1; then
  echo "ERROR: Ray is still running after 'ray stop --force'." >&2
  echo "  Kill it before rerunning: pkill -9 -f 'raylet|gcs_server|ray::'" >&2
  exit 1
fi

# slime/ray/actor_group.py hardcodes the cu12-named torch_memory_saver .so; bridge it.
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

# Empty decoy stops /etc/ld.so.cache resolving cu12's libcudart alongside cu13:
# TE's vendored cuDNN Frontend raises "Multiple libcudart found". Fixed (not mktemp'd)
# path because Ray actors resolve lazily and must find the decoy on every rerun.
DECOY_DIR="${DECOY_DIR:-/tmp/cudart-decoy-cu13-${USER}}"
mkdir -p "${DECOY_DIR}"
: > "${DECOY_DIR}/libcudart.so.12"
export LD_LIBRARY_PATH="${DECOY_DIR}:${LD_LIBRARY_PATH}"

# Standalone cudnn-frontend >=1.26 honors this and skips the libcudart probe entirely.
export CUDNN_FRONTEND_CUDART_LIB_NAME=libcudart.so.13

# Pin cuDNN to the venv wheel: without this CUDA_HOME's system cuDNN wins and the
# main/sublib version mismatch kills fused attention (CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED).
CUDNN_HOME="$(python -c "import sysconfig, os; print(os.path.join(sysconfig.get_path('purelib'), 'nvidia', 'cudnn'))")"
export CUDNN_HOME
export CUDNN_PATH="${CUDNN_HOME}"

export PYTHONUNBUFFERED=1
ray start --head --num-gpus ${NUM_GPUS} --disable-usage-stats

source ${SLIME_DIR}/scripts/models/${MODEL_TYPE}.sh

RUNTIME_ENV_JSON="{\"env_vars\": {\"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\", \"CUDA_HOME\": \"${CUDA_HOME}\", \"CUDNN_HOME\": \"${CUDNN_HOME}\", \"CUDNN_PATH\": \"${CUDNN_PATH}\", \"CUDNN_FRONTEND_CUDART_LIB_NAME\": \"${CUDNN_FRONTEND_CUDART_LIB_NAME}\", \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH}\"}}"

# slime/ray/actor_group.py gives Megatron train actors their own runtime_env that drops
# CUDA_HOME / CUDNN_* / LD_LIBRARY_PATH; --train-env-vars re-pins them for those actors.
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
  --rollout-temperature 1.0 \
  --eval-interval 10 \
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
  --sglang-log-level warning \
  --sglang-log-level-http warning \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --actor-num-gpus-per-node ${NUM_GPUS} \
  --colocate \
  --train-env-vars "${TRAIN_ENV_VARS_JSON}" \
  --megatron-to-hf-mode bridge \
  --custom-generate-function-path \
      agentcore_rl_toolkit.backends.experimental.slime.integration.rollout.generate \
  --custom-reward-post-process-path \
      agentcore_rl_toolkit.backends.experimental.slime.integration.rewards.normalize_episode_rewards \
  --custom-config-path ${CONFIG} \
  --use-dynamic-batch-size \
  --save ${CKPTS_DIR} \
  --save-interval 100 \
  --save-hf ${CKPTS_DIR}/hf/{rollout_id} \
  --use-wandb \
  --wandb-project slime-art \
  --wandb-group gsm8k-slime-grpo
