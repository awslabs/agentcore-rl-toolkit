# AgentCoreRuntimeApp: recoverable invocations over the HTTP contract

| Field | Value |
| --- | --- |
| Status | Accepted |
| Implementation | In progress |
| Date | 2026-09-25 |

## Summary

`AgentCoreRuntimeApp` adapts agents that already use the AgentCore Runtime HTTP
contract through `BedrockAgentCoreApp`. Replacing the app class preserves the
HTTP deployment and `@app.entrypoint` handler while adding addressable
foreground and background invocations, persisted state, and result retrieval.
It is the long-term replacement for `AgentCoreRLApp` on this HTTP path.

[Runtime Invocation Protocol (RIP)](./runtime_invocation_protocol.md) defines
the invocation contract shared by this app adapter and the Sandbox process
adapter: identities, `start/get`, states, retry semantics, and persistence
guarantees. This document owns the HTTP app's envelope mapping, execution
ownership, filesystem storage, and compatibility.

## Scope and boundaries

The adapter lives in
[`agentcore_rl_toolkit.runtime`](../src/agentcore_rl_toolkit/runtime/).
`AgentCoreRuntimeApp` is exported from that package and from
`agentcore_rl_toolkit`.

| Layer | Responsibility |
| --- | --- |
| `BedrockAgentCoreApp` | HTTP serving, request context, sync/async handler dispatch, and health tracking |
| `AgentCoreRuntimeApp` | Protocol dispatch, live invocation ownership, and persisted results |
| Application | Agent logic, setup, conversation state, and result contents |
| Workload client | Invocation IDs, retries, polling, timeouts, and session policy |
| Runtime deployment | Session routing and isolation, storage mounts, and compute lifecycle |

Native A2A agents use a separate adaptation path. The
[Rollout consumer design](./runtime_invocation_protocol.md#consumer-1-rollout-sdk)
defines the HTTP/A2A session adapters, client responsibilities, and migration.

The protocol path accepts JSON return values. Custom HTTP responses, generators,
stream replay, invocation cancellation, and compute-instance identification are
outside this adapter's scope.

## Entrypoint compatibility

The application retains `@app.entrypoint`:

```python
from agentcore_rl_toolkit import AgentCoreRuntimeApp

app = AgentCoreRuntimeApp()

@app.entrypoint
async def invoke(payload, context):
    return {"answer": payload["question"], "reward": 1.0}
```

Requests without `_agentcore_runtime` follow upstream HTTP behavior, including
streaming. Protocol requests use the same handler, but persist its JSON return
value inside `result`. The app treats that value as opaque: `reward`, agent
output, and other application fields are saved without renaming or injection.
It does not automatically persist the input payload.

Intercepting the envelope at the HTTP endpoint lets `get` bypass application
logic and keeps protocol responses separate from ordinary response handling.
Both synchronous and asynchronous handlers use upstream handler dispatch.

## HTTP mapping

Both operations use `InvokeAgentRuntime`, routed to `/invocations` with the
same `runtimeSessionId`. A start request carries application input alongside a
reserved envelope:

```json
{
  "_agentcore_runtime": {
    "version": 1,
    "operation": "start",
    "invocation_id": "inv-123",
    "background": true
  },
  "question": "What is 2 + 2?"
}
```

The adapter removes only `_agentcore_runtime` before calling the handler.
Application fields, including `_rollout`, pass through unchanged. Keeping the
envelope separate prevents execution controls from becoming rollout-specific
configuration.

The envelope requires integer `version: 1`, `operation`, and `invocation_id`.
IDs are 1–128 ASCII letters, digits, underscores, or hyphens, starting with a
letter or digit. `background` is a boolean valid only for `start`; it defaults
to `false`. Unsupported versions, unknown fields, and invalid values return
HTTP 400 with error code `InvalidRequest`. Explicit version validation prevents
a caller and server from silently interpreting different envelope formats.
Version 1 has no protocol-level conversation-ID field; applications can carry
their own conversation identity in the application payload.

Retrieval uses the same envelope without application input or `background`:

```json
{
  "_agentcore_runtime": {
    "version": 1,
    "operation": "get",
    "invocation_id": "inv-123"
  }
}
```

Every state response contains `version`, `invocation_id`, and `status`.
`completed` additionally contains either `result` (any JSON value, including
`null`) or `error: {"code": "...", "message": "..."}`.

Foreground starts wait for the invocation's terminal response; background
starts return its current state after acceptance. Repeated foreground starts
wait for the same live execution. Repeated starts with an existing record
never run the handler again, even if the supplied payload differs.

## Execution ownership and concurrency

One app process serves one Runtime session; AgentCore isolates different
sessions in separate execution environments. The process-local registry
therefore retains live tasks keyed only by `invocation_id`. The filesystem
retains identity and results across requests and app restarts. Multiple serving
processes for one session are unsupported because one process cannot establish
the liveness of another process's tasks.

Acceptance and lookup share one `asyncio.Lock`. A new start:

1. Keeps a reference to the live task, then reads the store under the lock.
2. Persists the start record if the ID is absent.
3. Registers AgentCore async-task tracking and retains the execution task
   before releasing the lock.
4. Waits for that task in foreground mode or returns `in_progress` in
   background mode.

The lock prevents concurrent retries from starting the same invocation twice.
It also prevents `get` from seeing the interval between start publication and
task registration as an interruption. Handler execution runs outside the lock,
so different invocation IDs can execute concurrently. The application owns
coordination of shared agent state.

The app retains and shields dispatch tasks from HTTP request cancellation,
including during acceptance. Otherwise, losing a connection after the start
record is written could abandon task registration. Foreground waiting also
shields the execution task, so disconnecting does not stop the handler.

Filesystem operations run through `asyncio.to_thread` to keep blocking writes,
flushes, or mount latency off the event loop. AgentCore async-task tracking
keeps `/ping` busy through execution and terminal publication. Tracking is
released only after publication succeeds or fails; detached task failures are
consumed and logged.

## Filesystem storage and recovery

`state_dir` selects the store root. Its default is
`Path(tempfile.gettempdir()) / ".agentcore_runtime"`, honoring `TMPDIR`. The app
creates the directory and checks writability at startup. A local default avoids
requiring a bucket or mount for ordinary use; deployments needing longer
retention can select a mounted path without changing the invocation protocol.
Selecting the path does not configure the mount.

HTTP `start/get` retrieves results whether `state_dir` points to local storage
or a compatible mount.

Records use this layout:

```text
<state_dir>/<sha256(runtime_session_id)>/<invocation_id>/
  started.json
  result.json
```

Session IDs come from the Runtime request context. Hashing treats them as
opaque strings rather than filesystem paths. Calls without a session header
share the empty-string namespace, suitable for a single local app. Independent
apps sharing a filesystem need separate roots.

The app writes each record to a temporary file in the same directory, then
atomically replaces the target file once the write is complete. This prevents
`get` from reading half-written JSON. The filesystem must support atomic
replacement.

`get` keeps a reference to the live task before reading files. It returns any
stored result first. If the task finishes during the read, `get` may return
`in_progress` once more, but will not mistake completion for `interrupted`.

| Evidence | Response |
| --- | --- |
| Terminal record | `completed`, with the stored result or error |
| Live task in the current process | `in_progress` |
| Start record without a live task or terminal record | `interrupted` |
| No record or live task | `not_found` |

These meanings do not depend on the storage location. After an app process
exits, surviving terminal records remain readable; start-only records become
`interrupted`. If the store disappears with compute replacement, a subsequent
lookup returns `not_found`. That state also covers an unknown ID or a lookup
in a different session; it is not proof that execution never occurred.

The root must remain stable across app restarts for records to be recoverable.
Managed session storage extends record lifetime across compute stop/resume,
subject to its retention and replication guarantees. It does not restore live
tasks or application memory. The
[shared persistence contract](./runtime_invocation_protocol.md#persistence-and-retrieval)
describes these limits, including asynchronous replication and graceful flush.

A session directory prevents accidental collisions but is not an access-control
boundary on shared storage, including an S3 Files mount. The deployment must
supply isolation and compatible filesystem semantics. Same-session `get`
requires a reachable app and may resume compute to retrieve records.

## Failure handling

Invocation states, including `interrupted`, `not_found`, and completed handler
errors, return HTTP 200. They describe an execution outcome rather than a failed
lookup operation. Clients should poll only `in_progress`; terminal errors,
interruption, and absence must end the wait without automatic resubmission.

If the handler raises an exception, the app saves the error for later retrieval.
If saving the result fails, the app tries to save that failure as an error instead.
If that also fails, the operation fails with HTTP 500 and the surviving start
record resolves to `interrupted` after the task is released. Storage read
failures also return HTTP 500, never `not_found`. Success is reported only after
the terminal record is published.

## Session and workload integration

The HTTP `RolloutSession` adapter submits and retrieves work through this
app's `start/get` operations and converts results to the rollout result format.

Invocation completion and foreground disconnection never stop the Runtime
session. Session cleanup belongs to the workload client, which may retain a
session for follow-up work.

Setup and run can be two invocations in the same Runtime session, each with
its own invocation ID. The handler can use an application payload field such
as `phase: "setup"` or `phase: "run"` to choose what to do.
