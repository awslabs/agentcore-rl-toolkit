#!/usr/bin/env bash
# Qwen3.5-9B language-only LoRA, synchronous colocated GRPO with verl.
set -euo pipefail
: "${TRAIN_DATA_PATH:?Set the TMax training Parquet path}"
: "${VAL_DATA_PATH:?Set the independent TBLite 2.0 Parquet path}"
: "${AGENT_RUNTIME_ARN:?Set the training harness Runtime ARN}"
: "${CHECKPOINT_DIR:?Set a shared checkpoint directory}"

export HYDRA_FULL_ERROR=1
export VERL_USE_EXTERNAL_MODULES=agentcore_rl_toolkit.backends.verl.trainer
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B}
PROJECT_NAME=${PROJECT_NAME:-agentcore-tmax}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen35_9b_lora64_tmax800_cosine50}
NNODES=${NNODES:-${SLURM_JOB_NUM_NODES:-16}}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
ROLLOUT_N=${ROLLOUT_N:-32}
MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH:-65536}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
TP=${TP:-4}
LORA_TARGETS='[language_model.decoder.layers.*.self_attention.linear_qkv,language_model.decoder.layers.*.self_attention.linear_proj,language_model.decoder.layers.*.self_attention.in_proj,language_model.decoder.layers.*.self_attention.out_proj,language_model.decoder.layers.*.mlp.linear_fc1,language_model.decoder.layers.*.mlp.linear_fc2]'

python -m verl.trainer.main_ppo \
    --config-name ppo_megatron_trainer \
    trainer.v1.trainer_mode=agentcore_sync \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=false \
    algorithm.use_kl_in_reward=false \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    data.train_files="['$TRAIN_DATA_PATH']" \
    data.val_files="['$VAL_DATA_PATH']" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size=100 \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_CONTEXT_LENGTH" \
    data.custom_cls.path=pkg://agentcore_rl_toolkit.backends.verl.dataset \
    data.custom_cls.name=PayloadDataset \
    +data.apply_chat_template_kwargs.enable_thinking=true \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.use_fused_kernels=true \
    actor_rollout_ref.model.lora.type=lora \
    actor_rollout_ref.model.lora.rank=64 \
    actor_rollout_ref.model.lora.alpha=128 \
    actor_rollout_ref.model.lora.merge=true \
    actor_rollout_ref.model.lora.target_modules="$LORA_TARGETS" \
    actor_rollout_ref.model.lora.exclude_modules='[]' \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_decay_style=cosine \
    actor_rollout_ref.actor.optim.min_lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_decay_steps='${trainer.total_training_steps}' \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_CONTEXT_LENGTH" \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra"]' \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="$TP" \
    actor_rollout_ref.actor.megatron.context_parallel_size=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.sequence_parallel=true \
    actor_rollout_ref.actor.megatron.use_mbridge=true \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=false \
    actor_rollout_ref.actor.megatron.param_offload=true \
    actor_rollout_ref.actor.megatron.optimizer_offload=true \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.prompt_length="$MAX_PROMPT_LENGTH" \
    actor_rollout_ref.rollout.response_length="$MAX_CONTEXT_LENGTH" \
    actor_rollout_ref.rollout.max_model_len="$MAX_CONTEXT_LENGTH" \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=true \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$MAX_CONTEXT_LENGTH" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.agent.num_workers=16 \
    actor_rollout_ref.rollout.agent.default_agent_loop=agentcore_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$SCRIPT_DIR/agentcore_agent.yaml" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.rollout_data_dir="$CHECKPOINT_DIR/../rollouts" \
    trainer.validation_data_dir="$CHECKPOINT_DIR/../validation" \
    trainer.resume_mode=disable \
    trainer.resume_from_path=null \
    trainer.del_local_ckpt_after_load=false \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node="$GPUS_PER_NODE" \
    trainer.nnodes="$NNODES" \
    trainer.save_freq=5 \
    trainer.test_freq=10 \
    trainer.total_training_steps=50 \
    trainer.total_epochs=1 "$@"
