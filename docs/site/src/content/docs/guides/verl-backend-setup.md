---
title: verl backend setup
description: Train an AgentCore Runtime-deployed agent with the verl training backend (Megatron / FSDP + vLLM).
---

This doc describes how to train an AgentCore Runtime-deployed agent with the [verl](https://github.com/volcengine/verl) training backend. Note that this is the direct integration with official verl, instead of through other packages like [rllm](https://awslabs.github.io/agentcore-rl-toolkit/guides/rllm-backend-setup/). We implement a thin AgentCore layer on top of official verl's PPO trainer, launched directly via `python -m agentcore_rl_toolkit.backends.verl.main`.

## Prerequisites

- A GPU cluster with **CUDA>=12.8** installed.
- Python 3.10+ and [`uv`](https://docs.astral.sh/uv/).
- AWS credentials with permission to invoke an AgentCore Runtime and read/write an S3 bucket.
- An AgentCore Runtime deployment of your agent — follow the [Prepare agent for RL](/agentcore-rl-toolkit/guides/agent-adaptation/) guide. Save the resulting **runtime ARN** — required as `actor_rollout_ref.rollout.agentcore.agent_runtime_arn` below for launching agent rollout sessions.
- An S3 bucket for rollout result delivery — required as `actor_rollout_ref.rollout.agentcore.s3_bucket` below for acquiring rewards.

## Installation

The verl backend has a heavyweight dependency stack (vLLM, Megatron-Core, Megatron-Bridge, Transformer Engine, Apex, flash-attn). Run the commands below to install environment for verl 0.8.0.

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

uv pip install vllm==0.21.0 --torch-backend=cu130
uv pip install tensordict torchdata --torch-backend=cu130
uv pip install accelerate datasets peft hf-transfer \
    "pyarrow>=19.0.0" pandas \
    codetiming hydra-core pylatexenc qwen-vl-utils wandb dill \
    pybind11 liger-kernel mathruler \
    "ray[default]>=2.41.0" "packaging>=20.0" tensorboard \
    latex2sympy2_extended math_verify \
    pytest py-spy pre-commit ruff --torch-backend=cu130

uv pip install "optree>=0.13.0" "grpcio>=1.62.1" --torch-backend=cu130

export CUDA_HOME=/usr/local/cuda-13.0
uv pip install --no-cache-dir --no-build-isolation \
    flash_attn==2.8.3 --torch-backend=cu130

APEX_CPP_EXT=1 APEX_CUDA_EXT=1 MAX_JOBS=32 uv pip install -v \
    --disable-pip-version-check \
    --no-cache-dir --no-build-isolation \
    "apex @ git+https://github.com/NVIDIA/apex.git@1bcf66dc11c3f614d6a071b1bb39854189ea3110" \
    --torch-backend=cu130

uv pip install --no-cache --no-build-isolation \
    "transformer_engine[pytorch,core_cu13]==2.15.0" \
    --torch-backend=cu130

uv pip install --no-deps "nvidia-cublas==13.5.1.27"

uv pip install --no-deps megatron-core==0.17.0 --torch-backend=cu130

uv pip install opencv-python --torch-backend=cu130
uv pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"

uv pip install 'nvidia-modelopt[torch]>=0.37.0'

uv pip install --no-deps \
    "megatron-bridge @ git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git@15d8eadcde212712d6bc9e271da5fef7d03732d7" \
    --torch-backend=cu130

# flash-linear-attention (required for Qwen3.5 Gated Delta Net attention)
uv pip install flash-linear-attention==0.4.1 --torch-backend=cu130

# Install verl, pinned to a known-good commit
git clone https://github.com/verl-project/verl.git
git -C verl checkout 0c36dee1387243ee0c1c57273dd81cd7902422b6
uv pip install --no-deps -e verl/ --torch-backend=cu130

# Download and install the stable branch vllm (support vLLM serving in RL rollout) of agentcore-rl-toolkit
git clone https://github.com/awslabs/agentcore-rl-toolkit -b vllm
uv pip install -e agentcore-rl-toolkit/

git clone https://github.com/rllm-org/rllm
uv pip install -e rllm/rllm-model-gateway
```

:::note
The above commands assume CUDA 13.0 is installed at `/usr/local/cuda-13.0`; adjust `CUDA_HOME` and `--torch-backend` if yours differs.
:::

## Prepare data

Verl reads training and validation data from parquet files. We make each training example (one row in parquet file) in the dataset processed and forwarded as a raw python dict in the `payload` to the AgentCore Runtime session — no trainer-side tokenization. So you just need to put every field that your implemented agent takes as input from session's `payload` as a top-level column of dataset parquet file.

For a reference example, we provide the data preprocessing script ([`preprocess_gsm8k.py`](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/src/agentcore_rl_toolkit/backends/verl/examples/math_agent/preprocess_gsm8k.py)) used for GSM8K dataset in [math agent](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/examples/strands_math_agent/rl_app.py). By running it, you downloads `openai/gsm8k`, extracts the gold answer from the `#### N` marker, and writes two Parquet files:

```bash
cd src/agentcore_rl_toolkit/backends/verl/examples/math_agent
python preprocess_gsm8k.py --output-dir gsm8k
```

Each row in the parquet file has two columns:

| Column | Purpose |
|--------|---------|
| `prompt` | The question text. Reaches the agent as `payload["prompt"]`. |
| `answer` | The ground-truth final answer. Reaches the agent as `payload["answer"]` and is passed to `GSM8KReward` as `ground_truth`. |

To train your own task, write a script that produces a Parquet file with whatever columns your agent's `payload` expects.

## Training Configuration

Verl uses yaml for training configuration, see verl's [official doc](https://verl.readthedocs.io/en/latest/examples/config.html) for configuration explanation. We make a training config file at [`src/agentcore_rl_toolkit/backends/verl/config/agentcore_grpo.yaml`](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/src/agentcore_rl_toolkit/backends/verl/config/agentcore_grpo.yaml). It inherits verl's full training config and adds the following new arguments:

```yaml
actor_rollout_ref:
  rollout:
    agentcore:
      agent_runtime_arn: ""                   # REQUIRED — The ARN of your deployed agent at AgentCore Runtime
      s3_bucket: ""                           # REQUIRED — S3 bucket for saving rewards and other artifacts of agent rollouts
      reqs_per_sec: 25                        # AgentCore Runtime invoke TPS limit (default 25, per-account)
      max_pool_connections: 10                # boto3 connection pool size
      max_rollout_time: 1800                  # Max running time of an AgentCore Runtime session in seconds
      gateway_port: 9090                      # local model-gateway port
      gateway_store: memory                   # gateway trace store backend
      gateway_cumulative_token_mode: false    # Turn on model gateway's cumulative token mode or not
      gateway_renderer_model_family: auto     # Renderer family used for model gateway's cumulative token mode

  actor:
    ppo_mini_steps: 1                         # The number of ppo mini steps per global training step
```

Please see comments in [`agentcore_grpo.yaml`](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/src/agentcore_rl_toolkit/backends/verl/config/agentcore_grpo.yaml) for more instructions.

You can edit these values directly in the yaml file, or as the next section's example script does, override them on the command line of training launch.

## Launch training

Training is launched with `python -m agentcore_rl_toolkit.backends.verl.main`, which loads `agentcore_grpo.yaml` and accepts standard yaml overrides from command lines.

We use math agent as a reference example, see complete training script at [`src/agentcore_rl_toolkit/backends/verl/examples/math_agent/run_agentcore_grpo.sh`](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/src/agentcore_rl_toolkit/backends/verl/examples/math_agent/run_agentcore_grpo.sh).

Before running the training:

1. Deploy the math agent RL app to AgentCore Runtime following the [guide](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/examples/strands_math_agent/README.md).
2. Get the Bedrock AgentCore Runtime ARN of your deployed agent, and create a S3 bucket for saving rollout rewards.
3. Get your wandb API key for training curve visualization (set `trainer.logger` below to use other visualization platform).

The training can be launched with the following commands:

```bash
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_HOME=/usr/local/cuda-13.0
VENV_CU13_LIB=$(python -c "import sysconfig, os; print(os.path.join(sysconfig.get_path('purelib'), 'nvidia', 'cu13', 'lib'))")
export LD_LIBRARY_PATH=$VENV_CU13_LIB:$LD_LIBRARY_PATH
export WANDB_API_KEY="your-wandb-api-key"

python3 -m agentcore_rl_toolkit.backends.verl.main \
    model_engine=megatron \
    algorithm.adv_estimator=grpo \
    data.train_files="$gsm8k/gsm8k_agent_train.parquet" \
    data.val_files="$gsm8k/gsm8k_agent_test.parquet" \
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
    trainer.default_local_dir=exp_agentcore_grpo \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='agentcore-rl-toolkit' \
    trainer.experiment_name='gsm8k' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.val_before_train=true \
    trainer.total_epochs=1
```

Things worth noting:

- **`model_engine`** — `megatron` (used in the example) or `dp` for FSDP.
- **`ppo_mini_steps`** — This is specific to our verl + AgentCore integration. One agent rollout trajectory can expand into multiple sequences, so the number of PPO mini-batches per global step isn't fixed; this pins it explicitly.
- **`enable_auto_tool_choice` / `tool_call_parser=hermes`** — required for tool-calling agents (the math agent uses a calculator tool). Match the parser to your model family.
- **Tool-call + cumulative tokens** — the example enables gateway's cumulative token mode by setting `gateway_cumulative_token_mode=true` with `gateway_renderer_model_family=qwen3`. See this [PR](https://github.com/rllm-org/rllm/pull/596) for more information about it.

For the full list of tunable fields, consult verl's yaml config [documentation](https://verl.readthedocs.io/en/latest/examples/config.html).
