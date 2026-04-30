---
title: Strands Math Agent
description: An RL-trainable GSM8K math agent built with Strands and the AgentCore RL Toolkit — the canonical "hello world" example.
---

An RL-trainable [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
math agent using Bedrock AgentCore RL Toolkit. The agent solves
grade-school word problems by invoking a `calculator` tool
for arithmetic steps and emitting a final answer after `####`.

This is the "hello world" for the toolkit — minimal
prompt and single tool. Start here.

## Quickstart

```bash
cd examples/strands_math_agent
uv venv --python 3.13 && source .venv/bin/activate
uv pip install -e .

# Run the RL-adapted app as a local HTTP server
python rl_app.py
```

Deploy the same entrypoint to AgentCore Runtime via the
[Prepare agent for RL → Deploy](/agentcore-rl-toolkit/guides/agent-adaptation/)
flow, then feed the resulting runtime ARN into
[`SlimeRunner`](/agentcore-rl-toolkit/guides/slime-backend-setup/)
to train.

## What's in the example

- **`basic_app.py`** — deployment-ready reference using `BedrockModel`.
- **`rl_app.py`** — RL-adapted version; reads `base_url` / `model_id`
  from `_rollout`, returns `{"rewards": ...}`.
- **`reward.py`** — `GSM8KReward`: extract the `####` answer, exact-match against ground truth.
- **`evaluate.py`** — batch-evaluate a deployed ACR agent via `RolloutClient`.

For local setup (Bedrock credentials, local vLLM, S3 bucket,
sending `curl` requests), Docker builds, ECR push, IAM setup, and
full deploy instructions, see the
[full README on GitHub](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/examples/strands_math_agent/README.md).
