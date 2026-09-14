# MigrationBench agent (Megatron + LoRA)

GRPO for the Java 8 to 17
[`strands_migration_agent`](../../../../../../examples/strands_migration_agent),
using Qwen3-Coder-30B-A3B with Megatron expert parallelism and LoRA.

## Install

Megatron-Bridge 0.5.0 requires Python 3.12. From the repository root, create the
root project environment with:

```bash
uv sync --python 3.12 --extra verl --group verl-megatron
```

## Prepare data

The agent example is a standalone uv project. From the repository root, prepare its
repository tarballs and metadata using that project's environment, then return to
this backend example to create the payload-only parquet files using the root verl
environment:

```bash
cd examples/strands_migration_agent
uv sync
uv run python preprocess.py --s3-bucket-name <data-bucket>

cd ../../src/agentcore_rl_toolkit/backends/verl/examples/migration_agent
uv run python preprocess_migrationbench.py --s3-bucket-name <data-bucket>
```

## Train

```bash
export AGENT_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/...
export ACR_S3_BUCKET=your-results-bucket
wandb login
./megatron_lora_sync_grpo.sh
```

The per-turn generation limit is configured in `agentcore_agent.yaml`; the shell
script separately configures the cumulative model context and verl storage limits.
Qwen3-Coder's chat-template hash maps to the gateway's `qwen3_5` response schema for
tool-call parsing.
