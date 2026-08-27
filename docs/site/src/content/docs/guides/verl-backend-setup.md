---
title: verl backend setup
description: Train an AgentCore Runtime-deployed agent with verl, the v1 agent-loop API, and the rollout gateway.
---

The direct verl backend integrates AgentCore rollouts as a custom
`AgentCoreAgentLoop` while using verl's standard
`python -m verl.trainer.main_ppo` entrypoint and v1 trainer. The in-repo rollout
gateway captures token IDs, log probabilities, and loss masks from multi-turn agent
calls and converts each trajectory-tree leaf into a verl training row.

The checked-in recipes have been run end to end with:

- FSDP full fine-tuning of Qwen3-4B on GSM8K.
- Megatron + LoRA fine-tuning of Qwen3-Coder-30B-A3B on MigrationBench.

## Prerequisites

- A GPU cluster supported by the pinned CUDA 13 stack.
- AWS credentials that can invoke the AgentCore Runtime and read/write the result
  S3 bucket.
- A deployed `AgentCoreRLApp` whose OpenAI-compatible model client forwards
  `payload["_rollout"]["api_key"]`.
- Network routing from the AgentCore containers to the trainer nodes' rollout
  gateway ports.

The backend is pinned to a tested verl commit through `tool.uv.sources`, so install
it from a checkout of this repository:

```bash
uv sync --extra verl
```

The full stack uses CUDA 13 wheels and requires driver 580.65.06 or newer and a GPU
with compute capability 7.5 or newer.

For Megatron, use Python 3.12 and install the additional dependency group:

```bash
uv venv --python 3.12
uv sync --extra verl --group verl-megatron
```

## Adapt the agent

The trainer generates one capture-session key per rollout and supplies it as
`_rollout.api_key`. Pass that value to the model client:

```python
@app.rollout_entrypoint
def invoke_agent(payload: dict, context):
    rollout_config = payload["_rollout"]
    api_key = rollout_config.get("api_key") or "EMPTY"
    model = OpenAIModel(
        client_args={
            "api_key": api_key,
            "base_url": rollout_config["base_url"],
        },
        model_id=rollout_config["model_id"],
        params=rollout_config.get("sampling_params", {}),
    )
    # Run the agent and return {"rewards": score}.
```

The fallback `"EMPTY"` keeps local evaluation and unauthenticated inference
endpoints working.

## Prepare data

Each parquet row must contain a `payload` column holding the exact JSON object the
agent expects:

```python
{"payload": {"prompt": "Natalia sold clips to...", "answer": "72"}}
```

Configure verl to use `PayloadDataset`:

```yaml
data:
  custom_cls:
    path: pkg://agentcore_rl_toolkit.backends.verl.dataset
    name: PayloadDataset
```

`PayloadDataset` synthesizes the chat-format `prompt` column required by verl from
`payload["prompt"]`. If the agent uses another field name, set
`+data.payload_prompt_field=<field>`. An explicit chat-format `prompt` column takes
precedence when present.

## Run the GSM8K recipe

```bash
cd src/agentcore_rl_toolkit/backends/verl/examples/math_agent
export AGENT_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/...
export ACR_S3_BUCKET=your-results-bucket
python preprocess_gsm8k.py --output-dir gsm8k
./fsdp_fft_sync_grpo.sh
```

The shell script configures verl and accepts additional Hydra overrides through its
trailing arguments:

```bash
./fsdp_fft_sync_grpo.sh trainer.logger='["console","wandb"]'
```

The accompanying `agentcore_agent.yaml` contains the AgentCore runtime ARN, result
bucket, per-turn token limit, timeout, and gateway settings. Values that vary by run
use OmegaConf environment interpolation.

## Run the MigrationBench recipe

See the
[`migration_agent` example](https://github.com/awslabs/agentcore-rl-toolkit/tree/main/src/agentcore_rl_toolkit/backends/verl/examples/migration_agent)
for the Megatron + LoRA setup and data-preparation commands.

## Important configuration

- `trainer.use_v1=true` is required because one AgentCore rollout may emit multiple
  training rows.
- `actor_rollout_ref.rollout.max_model_len` is the inference model's context
  capacity and must be set explicitly.
- `actor_rollout_ref.rollout.response_length` is both verl's response storage width
  and the gateway's cumulative trajectory budget.
- `max_tokens_per_turn` lives in `agentcore_agent.yaml` and limits each individual
  model call.
- The only supported reward mode is agent-side scoring: the app returns
  `{"rewards": score}`. Trainer-side reward functions are not yet supported.

For the full token-budget model, failure behavior, gateway networking options, and
troubleshooting notes, see the
[`backends/verl` README](https://github.com/awslabs/agentcore-rl-toolkit/tree/main/src/agentcore_rl_toolkit/backends/verl).
