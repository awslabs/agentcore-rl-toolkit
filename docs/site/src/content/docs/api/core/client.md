---
title: client
description: RolloutClient, RolloutFuture, and the BatchResult family.
sidebar:
  order: 1
---


_module: `agentcore_rl_toolkit.client`_

Client for invoking agents and collecting rollouts via S3 polling.

## `class RolloutClient`

Client for invoking agents and collecting rollouts with full lifecycle management.

Provides both sync and async APIs. Sync methods (``invoke``, ``run_batch``)
block the caller. Async methods (``invoke_async``, ``run_batch_async``) are
suitable for use inside ``asyncio`` event loops.

**Constructor**

```python
RolloutClient(
    agent_runtime_arn: str,
    s3_bucket: str,
    exp_id: str,
    max_retry_attempts: int = 5,
    tps_limit: int = 25,
    max_pool_connections: int = 10,
    base_url: str = None,
    model_id: str = None,
    extra_config = {},
)
```

Initialize RolloutClient for invoking agents and collecting rollouts.

**Parameters**

- `agent_runtime_arn` *(str)*: ARN of the ACR agent runtime (region is inferred from ARN)

- `s3_bucket` *(str)*: S3 bucket for storing rollout results

- `exp_id` *(str)*: Experiment ID for organizing results

- `max_retry_attempts` *(int)* — default `5`: Max retries for transient errors (default: 5)

- `tps_limit` *(int)* — default `25`: ACR invocation rate limit (default: 25)

- `max_pool_connections` *(int)* — default `10`: Max urllib3 connection pool size per boto3 client (default: 10).
Set to at least the max number of concurrent requests to avoid
"Connection pool is full, discarding connection" warnings.

- `base_url` *(str)* — default `None`: Optional vLLM/SGLang server URL

- `model_id` *(str)* — default `None`: Optional model ID for inference

- `**extra_config` — default `{}`: Additional config passed to _rollout (e.g., temperature, top_p)

### Methods

#### `invoke(payload: dict, session_id: str = None, input_id: str = None, rollout_overrides = {}) -> RolloutFuture`

Single invocation, returns Future for the result.

**Parameters**

- `payload` *(dict)*: The payload to send to the agent

- `session_id` *(str)* — default `None`: Optional session ID (default: auto-generated UUID)

- `input_id` *(str)* — default `None`: Optional input ID (default: auto-generated UUID)

- `**rollout_overrides` — default `{}`: Per-invocation overrides merged into _rollout config
(e.g., base_url, model_id, temperature). These take precedence over
client-level defaults.

**Returns**

-  *(RolloutFuture)*: RolloutFuture that can be awaited or polled for the result

#### `invoke_async(payload: dict, session_id: str = None, input_id: str = None, rollout_overrides = {}) -> RolloutFuture`

Async version of ``invoke()``. Returns a ``RolloutFuture``.

**Parameters**

- `payload` *(dict)*: The payload to send to the agent

- `session_id` *(str)* — default `None`: Optional session ID (default: auto-generated UUID)

- `input_id` *(str)* — default `None`: Optional input ID (default: auto-generated UUID)

- `**rollout_overrides` — default `{}`: Per-invocation overrides merged into _rollout config
(e.g., base_url, model_id, temperature). These take precedence over
client-level defaults.

**Returns**

-  *(RolloutFuture)*: RolloutFuture that can be awaited or polled for the result

Usage::

    future = await client.invoke_async({"prompt": "...", "answer": "42"})
    result = await future.result_async(timeout=60)
    # or simply: result = await future

    # With per-invocation overrides:
    future = await client.invoke_async(payload, base_url="http://other-server")

#### `run_batch(payloads: list[dict], max_concurrent_sessions: int, timeout: float = 900.0, initial_interval: float = 0.5, max_interval: float = 30.0, backoff_factor: float = 1.5, log_interval: float = 30.0) -> BatchResult`

Run batch with full lifecycle management.

Handles:
- TPS rate limiting (default 25/sec)
- Session concurrency limiting
- Automatic completion polling via S3 HEAD with exponential backoff
- Yielding results as they complete
- Per-request timeout
- Periodic status logging

**Parameters**

- `payloads` *(list[dict])*: List of payloads to process

- `max_concurrent_sessions` *(int)*: Max ACR sessions to run concurrently. Set based
on your ACR session quota and model API quota, etc.

- `timeout` *(float)* — default `900.0`: Max seconds to wait for each request (default 900s / 15 min).
Requests exceeding this yield a BatchItem with success=False.

- `initial_interval` *(float)* — default `0.5`: Starting poll interval (default 0.5s)

- `max_interval` *(float)* — default `30.0`: Cap on poll interval (default 30s)

- `backoff_factor` *(float)* — default `1.5`: Multiply interval by this each poll (default 1.5x)

- `log_interval` *(float)* — default `30.0`: Seconds between status log messages (default 30s)

**Returns**

-  *(BatchResult)*: BatchResult iterator that yields BatchItem for each payload

#### `run_batch_async(payloads: list[dict], max_concurrent_sessions: int, timeout: float = 900.0, initial_interval: float = 0.5, max_interval: float = 30.0, backoff_factor: float = 1.5, log_interval: float = 30.0) -> AsyncBatchResult`

Async version of ``run_batch()``. Returns an async iterator.

Same semantics as ``run_batch`` but submissions and polling use
``asyncio`` so cold starts on one request don't block others.

Usage::

    async for item in client.run_batch_async(payloads, max_concurrent_sessions=10):
        if item.success:
            process(item.result)
        else:
            log.warning(f"Payload {item.index} failed: {item.error}")

### Attributes

- `agent_runtime_arn`

- `agentcore_client`

- `base_url`

- `exp_id`

- `extra_config`

- `model_id`

- `region`

- `s3_bucket`

- `s3_client`

- `tps_limit`

## `class RolloutFuture`

Future representing an async rollout result, polled via S3 HEAD.

**Constructor**

```python
RolloutFuture(
    s3_client,
    s3_bucket: str,
    result_key: str,
    initial_interval: float = 0.5,
    max_interval: float = 30.0,
    backoff_factor: float = 1.5,
    session_id: str = None,
    input_id: str = None,
    agentcore_client = None,
    agent_runtime_arn: str = None,
    rate_limiter: ACRRateLimiter = None,
)
```

### Methods

#### `cancel() -> bool`

Cancel the underlying ACR session (best-effort, rate-limited).

Sets cancelled to True once called, even if the API call fails or
the client/session_id are unavailable. Use ``cancelled`` to check
whether cancellation was *attempted*, not whether the session was
successfully stopped. Returns True only when the stop API call succeeds.

#### `cancel_async() -> bool`

Async version of ``cancel()`` with rate limiting.

Uses the shared ACR rate limiter to avoid bursting stop calls
that compete with invoke calls for the same service-side rate limit.

#### `done() -> bool`

Check if result is ready (non-blocking). Updates backoff state.

Returns True if the future is in a terminal state: either the result
is available in S3 or the future was cancelled.

#### `done_async() -> bool`

Check if result is ready (non-blocking, async). Updates backoff state.

Like ``done()`` but wraps the S3 HEAD call in ``asyncio.to_thread()``
so it doesn't block the event loop.

#### `elapsed() -> float`

Returns seconds since this future was created.

#### `ready_to_poll() -> bool`

Returns True if enough time has passed since last poll.

#### `result(timeout: float = None) -> dict`

Block until result is ready, polling S3 HEAD with exponential backoff.

**Parameters**

- `timeout` *(float)* — default `None`: Max time to wait in seconds. If None, waits indefinitely
until the result appears. For long-running tasks, consider
setting a timeout to avoid infinite waits if the server fails
to save the result.

**Returns**

-  *(dict)*: The rollout result dictionary from S3

**Raises**

-  *(CancelledError)*: If the future was cancelled before a result was available.

-  *(TimeoutError)*: If timeout is reached before result is ready.

#### `result_async(timeout: float = None) -> dict`

Async version of ``result()``.

The underlying ACR session is automatically cancelled once the result
is fetched, so callers don't need to manage session cleanup.

**Parameters**

- `timeout` *(float)* — default `None`: Max time to wait in seconds. If None, waits indefinitely.

**Returns**

-  *(dict)*: The rollout result dictionary from S3.

**Raises**

-  *(CancelledError)*: If the future was cancelled before a result was available.

-  *(TimeoutError)*: If timeout is reached before result is ready.

#### `time_until_next_poll() -> float`

Returns seconds until this future should be polled again.

### Attributes

- `agent_runtime_arn`

- `agentcore_client`

- `cancelled` *(bool)* — True if cancellation was attempted (may not have succeeded).

- `input_id`

- `result_key`

- `s3_bucket`

- `s3_client`

- `session_id`

## `class BatchResult`

Iterator that manages batch execution lifecycle with adaptive polling.

**Constructor**

```python
BatchResult(
    client: RolloutClient,
    payloads: list[dict],
    max_concurrent: int,
    timeout: float = 900.0,
    initial_interval: float = 0.5,
    max_interval: float = 30.0,
    backoff_factor: float = 1.5,
    log_interval: float = 30.0,
)
```

### Attributes

- `backoff_factor`

- `client`

- `initial_interval`

- `log_interval`

- `max_concurrent`

- `max_interval`

- `payloads`

- `timeout`

## `class AsyncBatchResult`

Async iterator that manages batch execution lifecycle with adaptive polling.

Submissions are dispatched as concurrent tasks so that cold starts on one
request don't block submission of others.

**Constructor**

```python
AsyncBatchResult(
    client: RolloutClient,
    payloads: list[dict],
    max_concurrent: int,
    timeout: float = 900.0,
    initial_interval: float = 0.5,
    max_interval: float = 30.0,
    backoff_factor: float = 1.5,
    log_interval: float = 30.0,
)
```

### Attributes

- `backoff_factor`

- `client`

- `initial_interval`

- `log_interval`

- `max_concurrent`

- `max_interval`

- `payloads`

- `timeout`

## `class BatchItem`

Result wrapper for batch execution, distinguishing success from error.

**Constructor**

```python
BatchItem(
    success: bool,
    result: dict = None,
    error: str = None,
    index: int = None,
    elapsed: float = None,
)
```

### Attributes

- `elapsed` *(float)*

- `error` *(str)*

- `index` *(int)*

- `result` *(dict)*

- `success` *(bool)*
