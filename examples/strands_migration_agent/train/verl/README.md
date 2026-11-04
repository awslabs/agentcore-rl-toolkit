# Training MigrationBench with verl

Train the [Java 8 to 17 migration agent](../../README.md) with synchronous
GRPO, Megatron, and LoRA. Recipes are provided for Qwen3-Coder-30B-A3B-Instruct
and Qwen3.6-27B.

## Install

Install the [verl Megatron dependencies](../../../../src/agentcore_rl_toolkit/backends/verl/README.md#megatron-engine)
on every Ray node. Megatron requires Python 3.12. From the repository root:

```bash
uv sync --python 3.12 --extra verl --group verl-megatron
```

## Prepare data

The agent example is a standalone uv project. From the repository root, prepare
repository tarballs and metadata with the agent's environment, then create
payload-only Parquet files with the root verl environment:

```bash
uv sync --project examples/strands_migration_agent
uv run --project examples/strands_migration_agent \
  python examples/strands_migration_agent/preprocess.py \
  --s3-bucket-name <data-bucket>

uv run python examples/strands_migration_agent/train/verl/preprocess_migrationbench.py \
  --s3-bucket-name <data-bucket> --output-dir /path/to/migrationbench
```

Training excludes repositories without tests; validation includes every prepared
test repository.

## Train

Deploy the agent using its [deployment instructions](../../README.md#deploy-to-agentcore).
Its Runtime sessions must reach the gateway ports on the Ray nodes; see
[rollout networking](../../../../src/agentcore_rl_toolkit/backends/verl/README.md#rollouts-and-trajectory-capture).

From the repository root, with the training environment active on the Ray head:

```bash
export TRAIN_DATA_PATH=/path/to/migrationbench/migrationbench_agent_train.parquet
export VAL_DATA_PATH=/path/to/migrationbench/migrationbench_agent_test.parquet
export AGENT_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/...
export CHECKPOINT_DIR=/path/to/checkpoints
wandb login
bash examples/strands_migration_agent/train/verl/megatron_lora_sync_grpo_qwen3-coder-30b-a3b.sh
```

For Qwen3.6-27B, use `megatron_lora_sync_grpo_qwen3.6-27b.sh` instead.
Rollout settings are in [`agentcore_agent.yaml`](agentcore_agent.yaml).

Override shell variables and pass verl Hydra arguments when launching, for example:

```bash
TRAIN_BATCH_SIZE=16 ROLLOUT_N=8 \
  bash examples/strands_migration_agent/train/verl/megatron_lora_sync_grpo_qwen3.6-27b.sh \
  trainer.total_training_steps=20 trainer.total_epochs=20
```

The agent returns reward 0 for a failed build, 0.5 for a successful Java 17
build, and 1.0 when test preservation also passes.
