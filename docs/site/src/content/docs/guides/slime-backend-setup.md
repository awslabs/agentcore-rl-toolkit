---
title: SLIME training backend setup
description: Train an AgentCore Runtime-deployed agent with the SLIME training backend.
---

This doc describes how to train an AgentCore Runtime-deployed agent with the
[slime](https://github.com/THUDM/slime) training backend. The public
user surface is the
[`SlimeRunner`](/agentcore-rl-toolkit/api/backends/slime/runner/)
class for launching training.

For known issues (e.g. the norm-epsilon mismatch on
Qwen2.5-32B-Instruct) see
[slime troubleshooting](/agentcore-rl-toolkit/troubleshooting/slime/).

## Prerequisites

- A GPU cluster with **CUDA>=12.9** installed.
- Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).
- AWS credentials with permission to invoke an AgentCore Runtime and
  read/write an S3 bucket.
- An AgentCore Runtime deployment of your agent — follow the
  [Prepare agent for RL](/agentcore-rl-toolkit/guides/agent-adaptation/)
  guide. Save the resulting **runtime ARN** — required as the
  `agent_runtime_arn` argument on `SlimeRunner` below for Agent rollouts.
- An S3 bucket for rollout result delivery — required as the
  `s3_bucket` argument on `SlimeRunner` below.

## Installation

Choose **one** of the two paths below to install slime, then install
the toolkit into the same environment.

### Option A: Official slime docker

Follow
[slime's own installation docs](https://github.com/THUDM/slime#installation)
and use the container image (`slimerl/slime:latest`). Inside the
container, slime and Megatron-LM ship pre-installed at `/root/slime`
and `/root/Megatron-LM` — use those paths for `slime_dir` /
`megatron_dir` on `SlimeRunner`.

Install the toolkit with the slime-backend extras inside the container:

```bash
uv pip install -e ".[slime]"
```

:::note
We have only tested against official slime at commit
[`fa3c990`](https://github.com/THUDM/slime/commit/fa3c990af6f18efd3fd9922698bf4bf4048d1263).
:::

### Option B: Bare-metal install script

Install slime and its heavyweight dependency stack (Megatron-LM,
Transformer Engine, Apex, flash-attn, sglang, torch_memory_saver)
with the provided script, which clones slime + Megatron-LM into the
current directory and applies slime's official patches. Run it inside
your activated python environment.

:::note
Both CUDA 12.9 and CUDA 13 are supported. The following commands assume CUDA 13.0 is installed at `/usr/local/cuda-13.0`; adjust `CUDA_HOME` and `cu13` if yours differs.
:::

```bash
uv pip install -e ".[slime]"
export CUDA_HOME=/usr/local/cuda-13.0
bash src/agentcore_rl_toolkit/backends/slime/scripts/install_slime.sh cu13
```

Point `slime_dir` / `megatron_dir` on `SlimeRunner` at the `slime` and
`Megatron-LM` directories the script cloned.

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
