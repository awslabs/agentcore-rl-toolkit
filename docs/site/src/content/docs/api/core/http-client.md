---
title: HTTP client
description: AgentCoreHttpClient, InvocationHandle, and InvocationError.
sidebar:
  order: 1
---


_module: `agentcore_rl_toolkit.runtime.client`_

## `class AgentCoreHttpClient`

Invoke HTTP agents using a shared asynchronous AWS client.

**Constructor**

```python
AgentCoreHttpClient(
    runtime_arn: str,
    *,
    max_pool_connections: int = 100,
    read_timeout: float = 900,
    request_rate_limiter: RateLimiter | None = None,
    aws_client: Callable[[], Awaitable[Any]] | None = None,
)
```

### Methods

#### `async aclose() -> None`

Close the connection pool after all callers have finished using it.

#### `get_invocation(*, session_id: str, invocation_id: str) -> InvocationHandle`

Reconstruct a RIP handle without submitting work or making a network call.

#### `async invoke(payload: dict, *, session_id: str, invocation_id: str | None = None, background: bool = False) -> Any`

Retain invocation_id to retry or reattach after a lost response.

#### `async stop_session(session_id: str) -> None`

Stop the specified Runtime session.

### Attributes

- `runtime_arn`

## `class InvocationHandle`

A handle for one invocation.

**Constructor**

```python
InvocationHandle(
    client: AgentCoreHttpClient,
    session_id: str,
    invocation_id: str,
    snapshot: InvocationState | None = None,
)
```

### Methods

#### `async result(timeout: float | None = None) -> Any`

Wait for and return the original JSON result, or raise InvocationError.

#### `async status() -> InvocationState`

Return the invocation state, reusing an already completed result.

### Attributes

- `client`

- `invocation_id`

- `session_id`

## `class InvocationError(RuntimeError)`

A remote invocation failed, was interrupted, or could not be found.

**Constructor**

```python
InvocationError(
    session_id: str,
    snapshot: InvocationState,
)
```

### Attributes

- `invocation_id`

- `session_id`

- `snapshot`
