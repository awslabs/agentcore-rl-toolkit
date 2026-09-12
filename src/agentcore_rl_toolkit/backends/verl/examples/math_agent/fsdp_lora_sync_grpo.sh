#!/bin/bash
# FSDP LoRA variant of fsdp_fft_sync_grpo.sh, validated on 8x A100 40GB.
# Setup and usage: ../../README.md and its linked setup guide.
# Load base weights from safetensors to avoid the initial full-model sync OOM.
set -x

export HYDRA_FULL_ERROR=1
export VERL_USE_EXTERNAL_MODULES=agentcore_rl_toolkit.backends.verl.trainer

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
AGENT_LOOP_CONFIG=$SCRIPT_DIR/agentcore_agent.yaml

gsm8k_train_path=gsm8k/gsm8k_agent_train.parquet
gsm8k_test_path=gsm8k/gsm8k_agent_test.parquet
train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

PROJECT_NAME=${PROJECT_NAME:-agentcore_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gsm8k_qwen3_4b}
CKPTS_DIR=${CKPTS_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}

python3 -m verl.trainer.main_ppo \
    trainer.v1.trainer_mode=agentcore_sync \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=64 \
    data.val_batch_size=256 \
    data.max_prompt_length=14336 \
    data.max_response_length=$MAX_MODEL_LEN \
    data.custom_cls.path=pkg://agentcore_rl_toolkit.backends.verl.dataset \
    data.custom_cls.name=PayloadDataset \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_MODEL_LEN \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.prompt_length=14336 \
    actor_rollout_ref.rollout.response_length=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=agentcore_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    trainer.critic_warmup=0 \
    trainer.default_local_dir="$CKPTS_DIR" \
    trainer.resume_mode=disable \
    trainer.logger='["console"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 "$@"
