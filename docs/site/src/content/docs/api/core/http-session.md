---
title: HTTP rollout session
description: AgentCoreHttpSession for invocation and per-rollout cleanup.
sidebar:
  order: 2
---


_module: `agentcore_rl_toolkit.rollout_session.agentcore_http_session`_

## `class AgentCoreHttpSession(RolloutSession)`

One Runtime session. The caller owns the application payload.

**Constructor**

```python
AgentCoreHttpSession(
    session_id: str,
    session_state: PersistentDict,
    *,
    agentcore_runtime_arn: str,
)
```

### Methods

#### `async run(task: dict) -> RolloutDumpResponse`

#### `async setup(task: dict) -> None`

No-op for this adapter.

#### `async shutdown() -> None`

Stop this session once; cleanup failures must not hide rollout failures.

#### `with_client(session_id: str, client: AgentCoreHttpClient, session_state: PersistentDict | None = None) -> AgentCoreHttpSession`

A session over a caller-owned client, which the caller also closes.
