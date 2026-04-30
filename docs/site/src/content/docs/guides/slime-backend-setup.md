---
title: SLIME training backend setup
description: Train an AgentCore Runtime-deployed agent with the SLIME training backend.
---

How to train an AgentCore Runtime-deployed agent with the
[slime](https://github.com/THUDM/slime) training backend. The public
user surface is exactly two things: the
[`SlimeRunner`](/agentcore-rl-toolkit/api/backends/slime/runner/)
class (for launching training) and the two SGLang
[patch scripts](/agentcore-rl-toolkit/api/backends/slime/patches/)
(applied once to the SGLang install).

For known issues (Megatron-LM regression on 32B, norm-epsilon
mismatch, etc.) see
[Slime troubleshooting](/agentcore-rl-toolkit/troubleshooting/slime/).

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
  guide. Save the resulting **runtime ARN** — required `SlimeRunner` arguments below for Agent rollouts.

## Slime environment

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
request. Every line has the same shape:

```json
{"prompt": "...dummy placeholder", "metadata": { /* whatever your agent expects */ }}
```

- **`prompt`** — a top-level string used by slime for **length filtering** only.
- **`metadata`** — an object copied **verbatim** into the body of
  the AgentCore Runtime invocation. Every field you put here lands
  directly in the `payload` dict your `@rollout_entrypoint`
  function receives — the slime backend does no filtering,
  renaming, or schema validation on it. Put every per-rollout
  config the agent needs in here: the user prompt, ground-truth
  answers, task IDs, repo URIs, tool-specific settings — anything.
  Each example can carry different keys.

Because `metadata` is forwarded as-is, each agent declares its own
payload shape without any slime-side changes. A few illustrative
shapes from the examples in this repo:

```json
// math — question + ground-truth answer
{"prompt": "How many ...?",
 "metadata": {"prompt": "How many ...?", "answer": "42"}}

// migration — repo pointers + per-run switches
{"prompt": "Migrate this Java 8 repo to Java 17.",
 "metadata": {"repo_uri": "s3://...", "metadata_uri": "s3://...",
              "apply_static_update": true}}
```

A concrete script (building a 64-row GSM8K JSONL) might look like:

```python
from datasets import load_dataset
import json

ds = load_dataset("openai/gsm8k", "main", split="train")
with open("/path/to/gsm8k_tiny.jsonl", "w") as f:
    for i, row in enumerate(ds):
        question = row["question"]
        answer = row["answer"].split("####")[-1].strip()
        f.write(json.dumps({
            "prompt": question,
            "metadata": {"prompt": question, "answer": answer},
        }) + "\n")
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

## Tested versions

For reproducibility, here's the exact environment this integration
was validated against:

| Component | Version / SHA |
|---|---|
| Instance type | 8 × NVIDIA H100 80GB HBM3 |
| CUDA | `12.9` |
| PyTorch | `2.9.1+cu129` |
| Docker image | `slimerl/slime@sha256:0100c933f1f63e7c4acdb9ec575e769839d59de4a648551e09e3fe0e7885631b` (built 2026-04-28) |
| slime | commit `f3e7bd7f3091d3be05c20977eefb31a785d6221d` (2026-04-28) |
| SGLang | `v0.5.9` |
| Megatron-LM | commit `3714d81d418c9f1bca4594fc35f9e8289f652862` ⚠ see note |

:::caution[Megatron-LM pin]
The image bundles Megatron-LM at `1dcf0dafa` (~500 commits ahead of
slime's stable pin), which breaks 32B training — see
[Slime troubleshooting](/agentcore-rl-toolkit/troubleshooting/slime/).
We downgrade to `3714d81d` (slime's documented stable sha) via
`git checkout` inside `/root/Megatron-LM`. The table above reflects
the downgraded sha, not the one baked into the image.
:::
