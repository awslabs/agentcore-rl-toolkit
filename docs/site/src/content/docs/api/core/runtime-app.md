---
title: HTTP server
description: AgentCoreRuntimeApp and the HTTP invocation lifecycle.
sidebar:
  order: 0
---


_module: `agentcore_rl_toolkit.runtime.app`_

## `class AgentCoreRuntimeApp(BedrockAgentCoreApp)`

Add persisted start/get invocations to ``@app.entrypoint``.

**Constructor**

```python
AgentCoreRuntimeApp(
    *,
    background: bool = False,
    state_dir: str | Path | None = None,
    **kwargs,
)
```
