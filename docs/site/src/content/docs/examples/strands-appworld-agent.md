---
title: Strands AppWorld Agent
description: An RL-trainable ReAct code agent that solves AppWorld tasks through a Python REPL.
---

An RL-trainable [AppWorld](https://github.com/StonyBrookNLP/appworld)
agent using Bedrock AgentCore RL Toolkit. The agent solves
day-to-day tasks by interacting with simulated app APIs (Spotify,
Venmo, Gmail, etc.) via a Python REPL.

The agent uses a **ReAct code agent** pattern: a single `execute`
tool runs Python code inside an AppWorld sandbox. APIs are
discovered on-demand through the pre-loaded `apis` object, so tool
context stays small regardless of how many APIs exist.

## Quickstart

```bash
cd examples/strands_appworld_agent
uv venv --python 3.12 && source .venv/bin/activate
UV_GIT_LFS=1 uv pip install -e .
uv pip install -e ../../ --force-reinstall --no-deps
appworld install && appworld download data

# Start the RL app (needs a local vLLM server running on :4000)
python rl_app.py
```

Deploy via the
[Prepare agent for RL → Deploy](/agentcore-rl-toolkit/guides/agent-adaptation/)
flow, then evaluate with `evaluate.py` or train with
[`SlimeRunner`](/agentcore-rl-toolkit/guides/slime-backend-setup/).

## What's in the example

- **`rl_app.py`** — the rollout entrypoint; spins up an
  `AppWorldServers` + `AppWorld` context per-session and binds the
  `execute` tool as a closure over the task's `world`.
- **`reward.py`** — `AppWorldReward`: pass-rate from the task's
  `test_tracker`.
- **`evaluate.py`** — async batch evaluation over the AppWorld
  splits (`train` / `dev` / `test_normal` / `test_challenge`).
- **`few_shot_example.py`** — few-shot demo messages seeded into
  the agent's context.
- **`Dockerfile`** + **`deploy.py`** + **`config.toml`** —
  container build + programmatic ACR deploy.

For AppWorld data download, Docker + ECR setup, full VPC deploy
config, and evaluation options, see the
[full README on GitHub](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/examples/strands_appworld_agent/README.md).
