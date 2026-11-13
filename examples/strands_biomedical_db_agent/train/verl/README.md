# Training the biomedical database agents with verl

The GWAS Catalog and Open Targets recipes train Qwen3.6-27B with synchronous GRPO and Megatron
LoRA. Training and vLLM sampling share the same GPUs. The recipes target four
p4de/p5 nodes with eight GPUs each.

## Setup

Install the root package's [verl Megatron environment](../../../../src/agentcore_rl_toolkit/backends/verl/README.md#megatron-engine)
on every Ray node, using Python 3.12:

```bash
uv sync --python 3.12 --extra verl --group verl-megatron
```

Deploy the matching image target as an HTTP Runtime using the
[deployment guide](../../deployment/README.md). Its deployment script selects
Runtime V2:

| Recipe | Image target | Database configuration |
| --- | --- | --- |
| GWAS Catalog | `gwas-catalog` | `GWAS_CATALOG_SQL_DATA_ROOT` points to the prepared database snapshot. |
| Open Targets | `opentargets` | `OPEN_TARGETS_GRAPHQL_URL` points to the [Platform GraphQL service](../../deployment/opentargets/README.md), with `OPEN_TARGETS_EXPECTED_RELEASE` matching its release (default `26.06`). |

The Runtime must reach the rollout gateway ports on every Ray node; see
[rollout networking](../../../../src/agentcore_rl_toolkit/backends/verl/README.md#rollouts-and-trajectory-capture).

Export `VERL_USE_EXTERNAL_MODULES=agentcore_rl_toolkit.backends.verl.trainer`
before starting Ray on each node. Stage the model at the same path on all nodes
and make the dataset paths accessible to the workers.

## Data

Supply training and validation Parquet files containing `data_source` and
`payload` columns. The payload uses the
[training entrypoint's contract](../../README.md#training-entrypoints):

```json
{
  "data_source": "gwas_catalog_causal_gene",
  "payload": {
    "query": "A self-contained candidate-gene selection task",
    "expected_answers": ["CALLER_SUPPLIED_ANSWER"]
  }
}
```

verl groups validation metrics by the top-level `data_source`, producing
`val-core/gwas_catalog_causal_gene/reward/mean@1` for GWAS Catalog. For Open
Targets, use the same payload structure and preserve each row's `data_source`
as `causal_gene` or `variant`. The corresponding metrics are
`val-core/causal_gene/reward/mean@1` and `val-core/variant/reward/mean@1`.
If the column is absent, the group defaults to `unknown`.

## Run

With the training environment active on the Ray head, set paths and the
Runtime ARN for the selected agent:

```bash
export TRAIN_DATA_PATH=/path/to/train.parquet
export VAL_DATA_PATH=/path/to/eval.parquet
export AGENT_RUNTIME_ARN=arn:aws:bedrock-agentcore:REGION:ACCOUNT:runtime/AGENT
export CHECKPOINT_DIR=/path/to/checkpoints
export MODEL_PATH=/path/to/Qwen3.6-27B
```

For GWAS Catalog:

```bash
bash examples/strands_biomedical_db_agent/train/verl/gwas_catalog/megatron_lora_sync_grpo_qwen3.6-27b.sh
```

For Open Targets:

```bash
bash examples/strands_biomedical_db_agent/train/verl/opentargets/megatron_lora_sync_grpo_qwen3.6-27b.sh
```

Both recipes use LoRA training with a 64K context length and up to 20 agent turns.

Append Hydra `key=value` arguments to override training settings. Set
`PROJECT_NAME` (default `agentcore_biomni`) and `EXPERIMENT_NAME` to choose the
W&B project and run name.

AgentCore settings are in
[`gwas_catalog/agentcore_agent.yaml`](gwas_catalog/agentcore_agent.yaml) and
[`opentargets/agentcore_agent.yaml`](opentargets/agentcore_agent.yaml).
