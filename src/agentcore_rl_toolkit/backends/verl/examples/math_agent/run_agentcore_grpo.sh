#!/bin/bash
set -x

export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Set your cuda path
export CUDA_HOME=/usr/local/cuda-13.0

NVIDIA_LIBS=$(python -c "import sysconfig, os, glob; base=os.path.join(sysconfig.get_path('purelib'), 'nvidia'); print(':'.join(sorted(glob.glob(os.path.join(base, '*', 'lib')))))")
LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vE '^/usr/local/cuda(-[0-9.]+)?/' | paste -sd ':' -)"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# If you want to use wandb to visualize your training curves, set your wandb API key here.
export WANDB_API_KEY="your-wandb-api-key"

# Please run the following data preprocessing script once to get gsm8k dataset parquet files
python preprocess_gsm8k.py --output-dir gsm8k

# Paths to training/validation data
gsm8k_train_path=gsm8k/gsm8k_agent_train.parquet
gsm8k_test_path=gsm8k/gsm8k_agent_test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

CKPTS_DIR=exp_agentcore_grpo
rm -rf $CKPTS_DIR
ray stop

export HYDRA_FULL_ERROR=1

# Please follow the guide in examples/strands_math_agent/README.md to deploy the math agent
# RL app (examples/strands_math_agent/rl_app.py) to AgentCore Runtime.
# Set actor_rollout_ref.rollout.agentcore.agent_runtime_arn to your AgentCore Runtime ARN.
# Set actor_rollout_ref.rollout.agentcore.s3_bucket to your S3 bucket name.
python3 -m agentcore_rl_toolkit.backends.verl.main \
    model_engine=megatron \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=64 \
    data.val_batch_size=256 \
    data.max_prompt_length=14336 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.model.lora.rank=128 \
    actor_rollout_ref.model.lora.alpha=256 \
    actor_rollout_ref.model.lora.merge=true \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_steps=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes \
    actor_rollout_ref.rollout.agentcore.agent_runtime_arn=your-math-agent-arn \
    actor_rollout_ref.rollout.agentcore.s3_bucket=your-s3-bucket \
    actor_rollout_ref.rollout.agentcore.max_rollout_time=180 \
    actor_rollout_ref.rollout.agentcore.gateway_cumulative_token_mode=true \
    actor_rollout_ref.rollout.agentcore.gateway_renderer_model_family=qwen3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.default_local_dir=$CKPTS_DIR \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='agentcore-rl-toolkit' \
    trainer.experiment_name='gsm8k' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.val_before_train=true \
    trainer.total_epochs=1 $@
