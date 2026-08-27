#!/bin/bash
# GRPO on MigrationBench (Java 8->17) with rollouts on Bedrock AgentCore Runtime,
# using verl's stock main_ppo entrypoint (v1 trainer) and the agentcore_agent loop.
#
# Qwen3-Coder-30B-A3B is a sparse MoE, so this uses Megatron expert parallelism
# and LoRA rather than the math example's FSDP full fine-tune.
#
# Prerequisites:
#   1. Deploy examples/strands_migration_agent/rl_app.py to AgentCore Runtime (see
#      that example's README; it also needs a data bucket of prepared repo tarballs)
#      and export:
#        export AGENT_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/...
#        export ACR_S3_BUCKET=your-results-bucket
#   2. ACR containers must be able to reach this host on the gateway port
#      (auto-assigned; open the trainer CPU nodes' ports to ACR egress).
#   3. python preprocess_migrationbench.py --s3-bucket-name <data-bucket>   (once)
#   4. Megatron deps:
#        uv sync --extra verl --group verl-megatron
#      Use a Python 3.12 venv (`uv venv --python 3.12`).
#
# LoRA without NVIDIA Apex requires gradient_accumulation_fusion=False below.
#
# Tool-call parsing happens in the rollout gateway's renderer, auto-detected from the
# model's chat template (rollout_gateway/response_schemas.py — Qwen3-Coder maps to the
# qwen3_5 XML schema). Engine-level parser flags (vllm tool_call_parser etc.) have no
# effect on this token-in/token-out path and are deliberately not set.
#
set -x

export HYDRA_FULL_ERROR=1

# vLLM 0.23 kernel selection for weight sync on this stack: the symmetric-memory
# allreduce and the FlashInfer TRT-LLM FP16 MoE kernel are both incompatible with
# verl's in-place weight refit.
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_USE_FLASHINFER_MOE_FP16=0

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
AGENT_LOOP_CONFIG=$SCRIPT_DIR/agentcore_agent.yaml

train_files="['$SCRIPT_DIR/migrationbench_agent_train.parquet']"
test_files="['$SCRIPT_DIR/migrationbench_agent_test.parquet']"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-Coder-30B-A3B-Instruct}

# `response_length` is verl storage for the whole multi-turn trajectory, not a
# per-turn generation limit. Give it the full model context so a short opening prompt
# can use the otherwise-unused context. `prompt_length` is a separate padded region;
# remove-padding keeps those nominal widths from increasing the valid model sequence.
MAX_CONTEXT_LENGTH=131072
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=$MAX_CONTEXT_LENGTH

# Megatron parallelism for a 30B-A3B MoE. Rollout uses TP=4 independently (below),
# which gives the most KV cache per GPU for these long agent contexts.
TP=2
EP=2
CP=2
MAX_TOKENS_PER_GPU=$((MAX_CONTEXT_LENGTH / CP))

LORA_RANK=64
LORA_ALPHA=128

PROJECT_NAME=${PROJECT_NAME:-agentcore_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-migrationbench_qwen3_coder_30b}
CKPTS_DIR=${CKPTS_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}

python3 -m verl.trainer.main_ppo \
    --config-name ppo_megatron_trainer \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=32 \
    data.val_batch_size=128 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.custom_cls.path=pkg://agentcore_rl_toolkit.backends.verl.dataset \
    data.custom_cls.name=PayloadDataset \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora.rank=$LORA_RANK \
    actor_rollout_ref.model.lora.alpha=$LORA_ALPHA \
    actor_rollout_ref.model.lora.merge=true \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP \
    actor_rollout_ref.actor.megatron.sequence_parallel=true \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.actor.megatron.param_offload=false \
    actor_rollout_ref.actor.megatron.grad_offload=true \
    actor_rollout_ref.actor.megatron.optimizer_offload=true \
    ++actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    ++actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0 \
    ++actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.prompt_length=$MAX_PROMPT_LENGTH \
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.max_model_len=$MAX_CONTEXT_LENGTH \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=agentcore_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.ref.megatron.param_offload=true \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.ref.megatron.context_parallel_size=$CP \
    actor_rollout_ref.ref.megatron.sequence_parallel=true \
    trainer.critic_warmup=0 \
    trainer.default_local_dir=$CKPTS_DIR \
    trainer.resume_mode=disable \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=15 \
    trainer.total_epochs=1 "$@"
