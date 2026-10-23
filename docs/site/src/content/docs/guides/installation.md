---
title: Installation
description: Install the SDK dependencies for agent apps, rollout clients, and training, plus the AgentCore deployment CLI.
---

## Install the SDK

The core SDK supports Python 3.11+. Install it in the Python environment where
you run your agent or rollout client:

```bash
pip install 'agentcore-rl-toolkit[core]'
```

Or add it to a uv project:

```bash
uv add 'agentcore-rl-toolkit[core]'
```

The `core` extra includes agent apps and rollout clients. Install your agent
framework and its tools alongside the SDK; the
[examples](/agentcore-rl-toolkit/examples/) declare these dependencies in their
own `pyproject.toml` files.

For existing v0.1.x deployments, follow the [migration guide](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/MIGRATION.md).

## Choose dependencies for your environment

For environments that need only part of the SDK, select the corresponding extra:

| What you run | Extra | Includes |
|---|---|---|
| Agent apps and rollout clients together | `core` | Both `server` and `rollout` |
| An agent app on AgentCore Runtime | `server` | `AgentCoreRuntimeApp` dependencies |
| Rollout or sandbox clients | `rollout` | HTTP rollout clients, rollout sessions, Sandbox SDK clients, and AWS helpers |
| A rollout gateway for token capture | `gateway` | Token rendering, protocol adapters, and trajectory capture dependencies |

For example, install the client dependencies in an evaluation environment:

```bash
pip install 'agentcore-rl-toolkit[rollout]'
```

The bare package has no external dependencies; extras install the dependencies
needed by each component.

For training, follow your
[backend's setup guide](/agentcore-rl-toolkit/guides/overview/#training-backends).
The `tinker_api`, `tinker_skyrl`, and `verl` extras include `rollout` and `gateway`.
Training environments can have additional Python and GPU requirements; Megatron
and the SkyRL service require Python 3.12.

## Install the deployment CLI

The example deployment recipes use the Python `agentcore` CLI, installed
separately from agent dependencies:

```bash
uv tool install bedrock-agentcore-starter-toolkit
```

The Python CLI is no longer supported upstream. Migration to the recommended
npm CLI is tracked in [aws/agentcore-cli#2671](https://github.com/aws/agentcore-cli/issues/2671).

To try deploying the Math example, use the commands in its
[README](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/examples/strands_math_agent/README.md#deploy-with-the-agentcore-cli).

## Install from source

From a checkout of this repository, install the SDK in editable mode:

```bash
pip install -e '.[core]'
```

Run this command from the repository root. Choose the extra for your environment,
such as `.[rollout]` for clients or `.[tinker_api]` for the Tinker integration.
