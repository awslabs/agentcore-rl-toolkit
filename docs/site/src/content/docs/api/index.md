---
title: API Reference
description: Public Python API for agentcore-rl-toolkit.
sidebar:
  hidden: true
---

:::note[Skeleton page]
The API reference pages are generated from source docstrings via
`pydoc-markdown` (wired in M2 of the
[docs roadmap](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/docs/plan/roadmap/astro-starlight-docs-setup.md)).
Run `pnpm gen:api` to regenerate `app.md`, `client.md`, and
`reward.md` from the Python source.
:::

## Modules

- **[app](./app/)** — `AgentCoreRLApp`, the `@rollout_entrypoint` decorator.
- **[client](./client/)** — `RolloutClient`, `RolloutFuture`, `BatchResult` family.
- **[reward](./reward/)** — `RewardFunction` base class.
