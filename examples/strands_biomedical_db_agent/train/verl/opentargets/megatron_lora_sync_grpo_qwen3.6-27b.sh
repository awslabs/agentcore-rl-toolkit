#!/usr/bin/env bash
# Open Targets GRPO with colocated Megatron and vLLM on four 8-GPU p4de/p5 nodes.
set -euo pipefail

: "${TRAIN_DATA_PATH:?Set the Open Targets training Parquet path}"
: "${VAL_DATA_PATH:?Set the Open Targets validation Parquet path}"
: "${AGENT_RUNTIME_ARN:?Set the Open Targets HTTP Runtime ARN}"
: "${CHECKPOINT_DIR:?Set a checkpoint directory}"

export HYDRA_FULL_ERROR=1
export VERL_USE_EXTERNAL_MODULES=agentcore_rl_toolkit.backends.verl.trainer

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.6-27B}
PROJECT_NAME=${PROJECT_NAME:-agentcore_biomni}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-opentargets_qwen3.6_27B}

python3 -m verl.trainer.main_ppo \
    --config-name ppo_megatron_trainer \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=agentcore_sync \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    algorithm.use_kl_in_reward=false \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    data.train_files="['$TRAIN_DATA_PATH']" \
    data.val_files="['$VAL_DATA_PATH']" \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.seed=42 \
    data.max_prompt_length=8192 \
    data.max_response_length=65536 \
    data.custom_cls.path=pkg://agentcore_rl_toolkit.backends.verl.dataset \
    data.custom_cls.name=PayloadDataset \
    +data.payload_prompt_field=query \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.hybrid_engine=true \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.lora.rank=16 \
    actor_rollout_ref.model.lora.alpha=32 \
    actor_rollout_ref.model.lora.merge=true \
    'actor_rollout_ref.model.lora.target_modules=[linear_qkv,linear_proj,linear_fc1,linear_fc2,in_proj,out_proj]' \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    'actor_rollout_ref.actor.checkpoint.save_contents=["model","optimizer","extra"]' \
    actor_rollout_ref.actor.megatron.seed=42 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.context_parallel_size=2 \
    actor_rollout_ref.actor.megatron.sequence_parallel=true \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=false \
    actor_rollout_ref.actor.megatron.use_mbridge=true \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=false \
    actor_rollout_ref.actor.megatron.param_offload=false \
    actor_rollout_ref.actor.megatron.optimizer_offload=true \
    ++actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=true \
    ++actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0 \
    ++actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.prompt_length=8192 \
    actor_rollout_ref.rollout.response_length=65536 \
    actor_rollout_ref.rollout.max_model_len=65536 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    'actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32,64]' \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.checkpoint_engine.backend=naive \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=true \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.agent.num_workers=16 \
    actor_rollout_ref.rollout.agent.default_agent_loop=agentcore_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$SCRIPT_DIR/agentcore_agent.yaml" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.ref.megatron.param_offload=true \
    actor_rollout_ref.ref.megatron.seed=42 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.ref.megatron.context_parallel_size=2 \
    actor_rollout_ref.ref.megatron.sequence_parallel=true \
    trainer.critic_warmup=0 \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.resume_mode=disable \
    trainer.resume_from_path=null \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.save_freq=15 \
    trainer.test_freq=15 \
    trainer.total_epochs=2 \
    trainer.total_training_steps=45 "$@"
