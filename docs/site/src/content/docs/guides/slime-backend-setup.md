---
title: SLIME training backend setup
description: Train an AgentCore Runtime-deployed agent with the SLIME training backend.
---

This doc describes how to train an AgentCore Runtime-deployed agent with the
[slime](https://github.com/THUDM/slime) training backend. The public
user surface is exactly two things: the
[`SlimeRunner`](/agentcore-rl-toolkit/api/backends/slime/runner/)
class (for launching training) and the two SGLang
[patch scripts](/agentcore-rl-toolkit/api/backends/slime/patches/)
(applied once to the SGLang install).

For known issues (Megatron-LM regression on 32B, norm-epsilon
mismatch, etc.) see
[slime troubleshooting](/agentcore-rl-toolkit/troubleshooting/slime/).

## Prerequisites

- Hardware and CUDA requirements: see
  [slime's README](https://github.com/THUDM/slime#installation) and the
  [slime docker README](https://github.com/THUDM/slime/blob/main/docker/README.md)
  for tested GPU configurations per model size.
- Python 3.10+ and [`uv`](https://docs.astral.sh/uv/).
- AWS credentials with permission to invoke an AgentCore Runtime and
  read/write an S3 bucket.
- An AgentCore Runtime deployment of your agent — follow the
  [Prepare agent for RL](/agentcore-rl-toolkit/guides/agent-adaptation/)
  guide. Save the resulting **runtime ARN** — required as the
  `agent_runtime_arn` argument on `SlimeRunner` below for Agent rollouts.
- An S3 bucket for rollout result delivery — required as the
  `s3_bucket` argument on `SlimeRunner` below.

## slime environment

Follow
[slime's own installation docs](https://github.com/THUDM/slime#installation)
— either the container path (`slimerl/slime:latest`) or a bare-metal
install. Everything below runs inside this environment.

Inside the `slimerl/slime:latest` container, slime and Megatron-LM
ship pre-installed at `/root/slime` and `/root/Megatron-LM` — use
those paths for `slime_dir` / `megatron_dir` on `SlimeRunner`. For a
bare-metal install, point at wherever you cloned slime + Megatron-LM.

## Install the toolkit + apply patches

Inside the slime environment:

```bash
# From a clone of this repo
cd /path/to/agentcore-rl-toolkit

# Install the toolkit plus the slime-backend extras
uv pip install -e ".[slime]"
```

Then apply the SGLang `token_ids` patch — it adds
`prompt_token_ids` / `token_ids` fields to chat completion responses
so the gateway can capture RL training trace data. The patch is idempotent:

```bash
python -m agentcore_rl_toolkit.backends.slime.patches.sglang_token_ids

# Verify the patch round-trips under greedy decoding (any HF checkpoint
# works; Qwen2.5-0.5B-Instruct is the fastest to download + load)
python -m agentcore_rl_toolkit.backends.slime.patches.verify_sglang_token_ids \
    --model-path /path/to/Qwen2.5-0.5B-Instruct
# Expect: "OK: 4/4 checks passed"
```

## Prepare data

The training dataset is a JSONL file where each line is one rollout
request. Every line has the shape:

```json
{"prompt": "...", "metadata": { /* whatever your agent expects */ }}
```

- **`prompt`** — top-level string, used by slime for **length filtering** only.
- **`metadata`** — copied **verbatim** as the `payload` dict your
  `@rollout_entrypoint` function receives. Put every per-rollout
  config the agent needs here (user prompt, ground-truth answer,
  task IDs, repo URIs, etc.).

Example (GSM8K):

```json
{"prompt": "How many ...?", "metadata": {"prompt": "How many ...?", "answer": "42"}}
```

## Launch training with `SlimeRunner`

[`SlimeRunner`](/agentcore-rl-toolkit/api/backends/slime/runner/) is
the one and only entry point — a Python class that stops stale
processes, starts a Ray head, submits the slime training job, and
streams output. Defaults target **8 × H100** (`num_gpus=8`,
`tp_size=2`, `rollout_gpus_per_engine=2`); tune them for your
cluster.

```python
from agentcore_rl_toolkit.backends.slime import SlimeRunner

SlimeRunner(
    exp_id="gsm8k-3b-smoke",
    agent_runtime_arn="arn:aws:bedrock-agentcore:...",
    s3_bucket="your-bucket-name",
    model_dir="/path/to/Qwen2.5-3B-Instruct",
    data_path="/path/to/gsm8k_tiny.jsonl",
    model_type="qwen2.5-3B",
).train(num_rollout=1)   # 1 = smoke test; bump to 100 for a real run
```

**Wandb** — set `WANDB_API_KEY` and `WANDB_ENTITY` in your
environment (plus `wandb_project` / `wandb_group` on the
constructor) to log a run. Unset env vars skip wandb entirely.

**Config-file workflow** — dump kwargs to YAML and call
`SlimeRunner.from_yaml("my_run.yaml")` instead.

`SlimeRunner` exposes every field most experiments tune (cluster
shape, training hyperparameters, per-rollout ACR limits,
`extra_flags` for extra arguments to be directly passed to
[SLIME](https://github.com/THUDM/slime)) as constructor arguments.
See the
[API reference](/agentcore-rl-toolkit/api/backends/slime/runner/) or
`help(SlimeRunner)` for the full list.
