#!/bin/bash
# Train strands_math_agent with slime GRPO using ACR-based rollouts.
#
# The agent (examples/strands_math_agent/rl_app.py) must be deployed to ACR first.
# This script runs slime training with our custom rollout function that submits
# tasks to ACR, collects traces via rllm-model-gateway, and feeds per-turn
# Samples into Megatron for GRPO training.
#
# Default smoke-test target is Qwen2.5-3B-Instruct (large enough to produce
# non-trivial GSM8K rewards on the first rollout step, small enough to fit
# on 8×H100 without offload acrobatics). Hyperparameters follow slime's
# Qwen2.5-0.5B GSM8K production config (scripts/run-qwen2.5-0.5B-reproducibility.sh)
# and are inlined in the `python3 train.py` command below.
#
# Configuration:
#   config.yaml      — ACR deployment + toolkit tunables
#                      (cp from config.yaml.example; gitignored)
#   .wandb.env       — WANDB_API_KEY, WANDB_ENTITY (optional)
#   env vars below   — paths, model type, cluster config (override via env)
#
# Usage: bash train.sh
set -euo pipefail

# === Paths (edit these) ===
SLIME_DIR="${SLIME_DIR:?Set SLIME_DIR (path to slime repo)}"
MEGATRON_DIR="${MEGATRON_DIR:?Set MEGATRON_DIR (path to patched Megatron-LM)}"
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR (path to HF model checkpoint)}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH (path to training JSONL)}"
CONFIG="${CONFIG:-$(dirname $0)/config.yaml}"

# Model architecture (change for different models)
MODEL_TYPE="${MODEL_TYPE:-qwen2.5-3B}"

# Cluster / run config (override via env — 3B smoke-test defaults)
NUM_GPUS="${NUM_GPUS:-8}"
TP_SIZE="${TP_SIZE:-2}"                                 # tensor model parallel size (TP=2 keeps 3B activations under the SGLang-mem-fraction=0.7 budget)
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}" # GPUs per SGLang engine
NUM_ROLLOUT="${NUM_ROLLOUT:-1}"                         # total training iterations (smoke-test default)

# Load wandb credentials (optional)
[ -f "$(dirname $0)/.wandb.env" ] && source "$(dirname $0)/.wandb.env"

# === Setup ===
pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 3

export PYTHONUNBUFFERED=1
ray start --head --num-gpus ${NUM_GPUS} --disable-usage-stats

# Source model architecture args
source ${SLIME_DIR}/scripts/models/${MODEL_TYPE}.sh

# === Launch training ===
export no_proxy=127.0.0.1

# Env vars forwarded to every Ray worker. Wandb keys injected only when set.
# (ACR ARN + bucket flow through config.yaml, not env.)
RUNTIME_ENV_JSON=$(python3 -c '
import json, os, sys
env = {
    "PYTHONPATH": sys.argv[1],           # megatron.training is not editable-installed
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",  # required for Megatron TP comm/compute overlap
}
for key in ("WANDB_API_KEY", "WANDB_ENTITY"):
    val = os.environ.get(key)
    if val:
        env[key] = val
print(json.dumps({"env_vars": env}))
' "${MEGATRON_DIR}")

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 ${SLIME_DIR}/train.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint ${MODEL_DIR} \
  --ref-load ${MODEL_DIR} \
  --prompt-data ${DATA_PATH} \
  --num-rollout ${NUM_ROLLOUT} \
  --tensor-model-parallel-size ${TP_SIZE} \
  --rollout-num-gpus-per-engine ${ROLLOUT_GPUS_PER_ENGINE} \
  --input-key prompt \
  --label-key label \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 8 \
  --rollout-max-response-len 1024 \
  --rollout-temperature 1.0 \
  --advantage-estimator grpo \
  --use-kl-loss \
  --kl-loss-type low_var_kl \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta2 0.98 \
  --optimizer-cpu-offload \
  --overlap-cpu-optimizer-d2h-h2d \
  --use-precision-aware-optimizer \
  --sequence-parallel \
  --sglang-mem-fraction-static 0.7 \
  --sglang-cuda-graph-max-bs 32 \
  --sglang-tool-call-parser qwen25 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --actor-num-gpus-per-node ${NUM_GPUS} \
  --colocate \
  --megatron-to-hf-mode bridge \
  --rollout-function-path \
      agentcore_rl_toolkit.backends.slime.integration.rollout.generate_rollout \
  --custom-reward-post-process-path \
      agentcore_rl_toolkit.backends.slime.integration.rewards.normalize_episode_rewards \
  --custom-config-path ${CONFIG} \
  --use-dynamic-global-batch-size \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 9216 \
  ${WANDB_API_KEY:+--use-wandb --wandb-project ${WANDB_PROJECT:-slime-art} --wandb-group gsm8k-grpo}
  # To enable checkpointing, add: --save /path/to/ckpt --save-interval 10 [--save-hf]
