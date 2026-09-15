#!/bin/bash
# GRPO on MigrationBench (Java 8->17) with rollouts on Bedrock AgentCore Runtime,
# using verl's stock main_ppo entrypoint (v1 trainer) and the agentcore_agent loop.

set -x

export HYDRA_FULL_ERROR=1

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
AGENT_LOOP_CONFIG=$SCRIPT_DIR/agentcore_agent.yaml

# Where preprocess_migrationbench.py wrote the parquets. Defaults to alongside this
# script (the sibling 30B recipe's convention); override for a shared data directory:
#   DATA_DIR=/path/to/migrationbench ./megatron_lora_sync_grpo_qwen3.6-27b.sh
DATA_DIR=${DATA_DIR:-$SCRIPT_DIR}
train_files="['$DATA_DIR/migrationbench_agent_train.parquet']"
test_files="['$DATA_DIR/migrationbench_agent_test.parquet']"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.6-27B}

MAX_CONTEXT_LENGTH=131072
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=$MAX_CONTEXT_LENGTH

TP=4
CP=2
MAX_TOKENS_PER_GPU=$((MAX_CONTEXT_LENGTH / CP))

LORA_RANK=64
LORA_ALPHA=128

PROJECT_NAME=${PROJECT_NAME:-agentcore_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-migrationbench_qwen3.6_27B}
CKPTS_DIR=${CKPTS_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}


python3 -m verl.trainer.main_ppo \
    --config-name ppo_megatron_trainer \
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
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora.rank=$LORA_RANK \
    actor_rollout_ref.model.lora.alpha=$LORA_ALPHA \
    actor_rollout_ref.model.lora.merge=true \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP \
    actor_rollout_ref.actor.megatron.sequence_parallel=true \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
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
    +actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=triton \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false \
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
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.ref.megatron.param_offload=true \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP \
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
