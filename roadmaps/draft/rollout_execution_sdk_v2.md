# Rollout Execution SDK v2: foreground and background Runtime invocation

## Summary

Rollout Execution SDK v1 was designed for long-running agent executions that
should not depend on one client connection remaining open. It detaches the user
handler, keeps the AgentCore session busy, writes the terminal result to S3, and
lets a client retrieve that result asynchronously.

Rollout Execution SDK v2 retains that asynchronous execution model while adding
two connection modes through the same handler:

- `background=false`: keep the initial `InvokeAgentRuntime` connection open and
  return the final handler result on that connection.
- `background=true`: return an acknowledgement immediately, continue the
  handler in the background, and let the client retrieve the result later.

Every logical call has an invocation ID that is distinct from the AgentCore
Runtime session ID. The Runtime session identifies the execution environment
and isolation boundary; the invocation ID identifies one addressable execution
within that environment. A caller may also attach an opaque conversation ID,
but conversation history and follow-up semantics remain the application's
responsibility.

For background delivery, AgentCore managed session storage is the initial
default for execution markers and terminal results. S3 remains a valid
alternative storage backend for applications that need different isolation,
retention, artifact-size, or cross-session access properties. Storage choice
and result retrieval are separate internal concerns and do not change the
public invocation API.

The Rollout Execution SDK is the training- and evaluation-facing product layer.
It is built on a generic managed-invocation protocol for AgentCore Runtime;
rollout-specific configuration, reward handling, trajectory capture, and
trainer integrations remain separate concerns.

## Rollout Execution SDK v1

This proposal uses **Rollout Execution SDK v1** to refer to the behavior
implemented today by `AgentCoreRLApp`, `@app.rollout_entrypoint`,
`RolloutClient`, and `RolloutFuture`. It was not previously defined as a formal
product version.

### Problem it solves

Agent rollouts can run for minutes or hours while making model calls and using
tools. A training engine may need to launch hundreds or thousands of these
executions concurrently.

Keeping one synchronous client connection open for every rollout is brittle:

- callers may restart or lose network connections;
- long-held connections consume client and intermediary resources;
- cold starts or one slow rollout can interfere with concurrent submission;
- the training engine needs a completion signal that is independent of the
  original request.

V1 solves two core problems:

1. **Long-running execution**: the agent continues after the initial
   `InvokeAgentRuntime` request returns.
2. **Asynchronous result delivery**: the trainer can retrieve a successful or
   failed result later without maintaining the original connection.

### Current server-side behavior

`@app.rollout_entrypoint`:

1. Dispatches both synchronous and asynchronous user handlers through
   `BedrockAgentCoreApp._invoke_handler`.
2. Detaches the handler from the initial request with `asyncio.create_task`.
3. Registers an AgentCore async task so `/ping` reports `HealthyBusy` while the
   handler runs.
4. Writes successful or failed results to a customer-managed S3 bucket.
5. Returns an immediate acknowledgement containing the result location.

### Current client-side behavior

`RolloutClient` and `RolloutFuture`:

1. Generate one `runtimeSessionId` per rollout.
2. Submit the invocation and receive the immediate acknowledgement.
3. Poll S3 for the terminal result.
4. Apply concurrency limits, rate limiting, timeout handling, and session
   cleanup.
5. Support both blocking Python calls and asyncio-compatible calls.

### Scope

V1 is the invocation and result-delivery layer. It does not capture token-level
training trajectories itself; that is handled separately by the rollout
gateway and trainer integrations. It also does not require a particular result
schema beyond a JSON-serializable dictionary. Rewards are a convention used by
training integrations, not a responsibility of the app runtime.

## Reference pattern: OpenAI Responses API

The [OpenAI Responses API background mode](https://developers.openai.com/api/docs/guides/background/)
is a useful existing pattern for separating connection lifetime from execution
lifetime:

- one invocation API accepts a `background` flag;
- foreground requests keep the original connection open until a terminal
  result is available;
- background requests return an addressable in-progress execution;
- status and final output belong to a server-side execution resource rather
  than the original TCP connection;
- the client can retrieve or cancel that execution later.

The relevant idea for this proposal is simple: `background` controls response
delivery and connection lifetime, not the user handler's execution semantics.
Foreground and background calls use the same input and produce the same logical
result.

Responses also provides a useful identity separation:

- a Conversation is an optional logical history container;
- every Response is a distinct execution with its own ID and lifecycle;
- compute placement is not exposed as part of the public API.

AgentCore requires one additional infrastructure-level identity:

```text
Runtime session
  ├── conversation A (optional, application-managed)
  │     ├── invocation 1
  │     └── invocation 2
  └── conversation B (optional, application-managed)
        └── invocation 3
```

One Runtime session may host multiple application-managed conversations. The
SDK treats the conversation ID as opaque metadata and does not manage
conversation state.

The analogy is therefore `conversation_id` to Conversation and
`invocation_id` to Response ID. `runtimeSessionId` has no direct public
Responses equivalent because it describes compute placement, isolation,
storage scope, and lifecycle rather than logical conversation state.

This proposal does not attempt to reproduce the complete Responses resource or
Conversation model. The persisted invocation manifest is the closest
equivalent to the Response resource, retrieval is initially mediated by
session compute, and the SDK treats `conversation_id` only as optional opaque
correlation metadata.

The design borrows the foreground/background connection pattern, not the
Responses control plane.

## Why v2

V1 established a working async-first path with minimal changes to agent code.
V2 addresses limitations that became clearer as the SDK gained evaluation and
trainer integrations, and as AgentCore added
[managed session storage](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-filesystem-configurations.html)
and updated Runtime quotas.

| V1 behavior or limitation | Consequence | V2 direction |
| --- | --- | --- |
| Every rollout is detached from the initial request. | Short calls must pay the background-storage and polling cost even when the original connection could return the result. | Add `background=false` for foreground delivery while preserving `background=true`. |
| Background execution, result storage, and S3 delivery are combined in `rollout_entrypoint`. | Execution mode cannot evolve independently from the storage and retrieval mechanism. | Separate connection mode, execution state, result persistence, and client retrieval. |
| Results are written to a customer-managed S3 bucket. | Users must configure a bucket, IAM permissions, and result prefixes even when session-scoped storage is sufficient. | Use AgentCore managed session storage as the default while retaining S3 as an explicit alternative backend. |
| All sessions use the same Runtime IAM role for the result bucket. | An S3 prefix is a convention rather than a session-isolation boundary. | Scope result data to the AgentCore session's managed storage. |
| S3 is the only completion and retrieval channel. | The client and future are coupled to one external storage backend. | Retrieve through the same session initially and hide retrieval behind a replaceable internal abstraction. |
| `runtimeSessionId`, `input_id`, and the result key collectively act as an implicit execution handle. | Follow-up calls and multiple executions in one sticky session cannot be addressed independently. | Give every logical call an explicit invocation ID, generated before submission and used for status, result, cancellation, and retry deduplication. |
| There is no server-side `start` versus `get` protocol. | A same-session polling design could accidentally invoke the user handler again after process-local state is lost. | Add explicit internal operations so retrieval never enters the user handler. |
| `RolloutFuture` stops the Runtime session after result retrieval, timeout, or cancellation. | Invocation completion is treated as session completion, preventing sticky-session follow-up. | Separate wait timeout, invocation cancellation, and explicit or policy-driven session termination. |
| The client applies one 25 TPS limiter to invocation and session stop. | Existing-session data-plane traffic is unnecessarily restricted under the current quota model. | Model the shared 1,000 TPS data-plane quota separately from the 25 TPS new-session quota. |
| Result files include the full original payload for debugging. | Credentials and sensitive task data may be persisted unnecessarily. | Persist only the minimal execution manifest and result metadata by default. |
| The public names are `AgentCoreRLApp` and `rollout_entrypoint`. | A general Runtime capability appears specific to reinforcement learning. | Generalize the app and decorator names after the protocol and delivery behavior are established. |

The rest of this document specifies these changes in detail.

## Goals

- Support foreground and background delivery through one entrypoint.
- Keep Python sync/async handler support orthogonal to the connection mode.
- Preserve compatibility with the current one-rollout-per-session training
  model without encoding one invocation per session into the protocol.
- Give each logical execution an invocation ID distinct from
  `runtimeSessionId`.
- Allow an optional application-managed conversation ID without implementing
  conversation history or follow-up policy in the Runtime SDK.
- Use managed session storage as the default background-state backend while
  allowing alternative durable backends such as S3.
- Prevent polling or duplicate requests from running the user handler again.
- Preserve background results across idle execution-environment termination.
- Hide the internal start/get protocol behind the client and future.
- Remove the requirement for users to configure a result S3 bucket and policy.
- Keep retrieval replaceable so a future direct session-storage API does not
  require a public API change.
- Separate generic Runtime behavior from RL-specific payload configuration and
  trainer integration.

## Non-goals

- Reproducing the full OpenAI Responses resource model.
- Creating, storing, listing, or interpreting conversation history.
- Defining how an application merges, branches, or serializes conversation
  turns.
- Providing a global list/search API for invocations.
- Guaranteeing concurrent execution of multiple invocations within one Runtime
  session in the first version. The identity and storage model must not prevent
  a later concurrency policy.
- Providing strong distributed exactly-once execution.
- Guaranteeing retention beyond the lifecycle and retention policy of the
  selected storage backend.
- Persisting or resuming streaming responses in background mode.
- Protecting SDK metadata from code with arbitrary filesystem access inside the
  same session.
- Replacing `BedrockAgentCoreApp` or forking its Runtime implementation.

## Terminology

This proposal uses `foreground` and `background` to describe connection and
result-delivery behavior:

- **Foreground**: the initial client connection waits for the final result.
- **Background**: the initial connection returns before the handler finishes.

These terms do not describe whether the Python user handler is synchronous or
asynchronous. `BedrockAgentCoreApp._invoke_handler` already supports sync
functions, async functions, sync generators, and async generators.

This proposal distinguishes three identities:

- **Runtime session ID**: the AgentCore routing, isolation, storage, resource,
  and lifecycle boundary.
- **Conversation ID**: optional opaque application metadata identifying a
  logical history or interaction thread. The SDK neither creates nor manages
  that history.
- **Invocation ID**: one top-level execution of the registered handler. Every
  follow-up is a new invocation, even when it uses the same Runtime session and
  conversation.

A retry intended to refer to the same managed execution reuses its invocation
ID. Submitting the same payload with a new invocation ID is a distinct
execution and must not be classified as a duplicate.

## Proposed user experience

The illustrative client API is:

```python
# The initial InvokeAgentRuntime connection waits for completion.
result = client.invoke(
    payload,
    background=False,
    conversation_id="conv-123",  # optional, opaque to the SDK
)

# The initial connection returns after the background task is accepted.
future = client.invoke(
    payload,
    background=True,
    conversation_id="conv-123",
)
print(future.invocation_id)
result = future.result(timeout=600)
```

The async Python API remains a separate axis:

```python
# Does not block the caller's asyncio event loop, but the remote connection
# remains open until the handler completes.
result = await client.invoke_async(payload, background=False)

# Submission and result retrieval are both non-blocking to the caller's loop.
future = await client.invoke_async(payload, background=True)
result = await future.result_async(timeout=600)
```

The concrete return typing can use overloads:

```python
invoke(..., background=False) -> dict
invoke(..., background=True) -> RolloutFuture
```

The client generates an invocation ID before sending the network request. Most
callers do not need to provide or manage it. Advanced callers may supply one
explicitly when they need to persist an identity before submission, hand an
execution across processes, or retry after losing local client state:

```python
future = client.invoke(
    payload,
    background=True,
    invocation_id="inv-stable-123",
)
```

The initial managed-background protocol deliberately uses the client-generated
invocation ID for both execution identity and submission deduplication. These
are conceptually separate: an invocation ID identifies the execution resource,
while an idempotency key identifies repeated attempts to create that resource.
Using one value for both avoids requiring users to manage two fields before the
SDK has a separate control plane.

Whether `background` defaults to `true` for compatibility with the current
rollout client or to `false` to match ordinary request/response behavior remains
an open API decision.

## Server-side design

### One entrypoint, two connection modes

Both modes execute the same registered user handler through the upstream
`_invoke_handler` implementation.

For `background=false`:

1. Validate the request.
2. Await the user handler.
3. Return its result on the initial connection.
4. Let upstream `BedrockAgentCoreApp` preserve its normal error and streaming
   behavior.

For `background=true`:

1. Resolve the current `runtimeSessionId` and invocation ID.
2. Claim the invocation ID and persist its execution marker in managed session
   storage.
3. Register the background task before returning, so `/ping` is already
   `HealthyBusy`.
4. Store a strong reference to the created task.
5. Return an `in_progress` acknowledgement.
6. Execute the user handler.
7. Atomically persist the terminal result or error.
8. Only after persistence succeeds, call `complete_async_task`.

The ordering in steps 7 and 8 is required. Once the task is removed from
AgentCore async-task tracking, the session may become idle and its execution
environment may be terminated. The terminal result must already be durable
before that can happen.

### Internal invocation operations

The SDK needs three logical operations:

- `start(invocation_id)`: begin an invocation if that ID has not already been
  claimed.
- `get(invocation_id)`: read status or result without invoking the user
  handler.
- `cancel(invocation_id)`: cooperatively cancel the targeted invocation without
  terminating the Runtime session.

The concrete transport for each operation is an implementation detail. When an
operation is carried through the app invocation payload, it should use a
reserved namespace separate from the user's payload. In particular, it should
not reuse `_rollout`, which carries RL and inference configuration such as
`base_url`, `model_id`, API keys, and sampling parameters. An optional
conversation ID is passed to the user handler as opaque invocation context; the
wrapper may record it for correlation but does not load or mutate conversation
history.

A `get` operation must never call the user handler, even if the in-memory state
was lost. Whether it is implemented as an internal app operation or by a
deterministic command helper, explicitly distinguishing `start` from `get`
prevents a polling request from being mistaken for a new invocation after an
execution environment restart.

### State machine

Each invocation has an independent minimum observable state machine:

```text
absent
  |
  | start
  v
in_progress ------> completed
      |
      +-----------> failed
      |
      +-----------> cancelled
      |
      +-----------> interrupted
```

`interrupted` is derived during recovery. If persistent storage contains a
start marker but the new execution environment has no corresponding live task
and no terminal result, the previous execution did not finish cleanly.

The SDK should not automatically rerun an interrupted handler. Automatic rerun
would be unsafe for handlers with side effects and would make execution
semantics depend on infrastructure failures. A future explicit retry policy can
be designed separately.

Invocation cancellation and Runtime session termination are different
operations. Cancelling an invocation should cooperatively cancel the active
task and persist a `cancelled` terminal state while leaving the sticky session
available for follow-up work, including new invocations in the same
conversation. `StopRuntimeSession` should remain an explicit session-lifecycle
operation or an optional one-shot rollout policy; it should not be the
implementation of invocation cancellation in the generic layer.

### Storage backend and live task tracking

The selected storage backend is the authoritative recovery boundary. It must
persist:

- one start marker per invocation, written before the initial background
  acknowledgement;
- one terminal result per invocation containing the handler result, a
  serialized error, or cancellation metadata.

Managed session storage is the initial default because it scopes data to the
Runtime session without requiring a customer-managed bucket or bucket policy.
An illustrative layout for that backend is:

```text
<session-storage-mount>/.agentcore-runtime/
  invocations/
    inv-001/
      started.json
      result.json
    inv-002/
      started.json
      result.json
```

The exact filenames are private. Terminal result publication must be atomic so
concurrent readers never observe a partially written result.

The marker may include:

- protocol version;
- creation time;
- invocation ID;
- optional opaque conversation ID;
- sanitized correlation metadata;
- initial status.

The terminal file may include:

- `status`: `completed`, `failed`, or `cancelled`;
- the JSON-serializable result;
- a structured error or cancellation reason;
- completion time;
- protocol version.

Payloads may contain credentials or sensitive task data. The SDK should not
persist the complete input payload by default merely for debugging, unlike the
current S3 result format.

The active process will necessarily retain operational objects such as the
current task reference and AgentCore async-task tracking ID. It may also cache
manifest state for faster reads. This process-local state is an implementation
detail, not an authoritative storage backend or part of the public contract.
All background states needed after idle termination or process replacement must
be reconstructible from the selected durable storage backend.

S3 is also a valid storage backend. It trades the default backend's
session-scoped isolation and zero bucket configuration for direct external
access, cross-session sharing, independent retention, and support for larger
artifacts. Selecting S3 must not change `start`, `get`, `cancel`, or the public
client. V2 does not specify polling a Python global variable as a supported
result-delivery mode.

### Duplicate requests and concurrency

The app must claim the invocation ID before the first await that could allow
another request for the same ID to enter:

1. Check persistent terminal state.
2. Atomically create the persistent start marker if it does not exist.
3. Install `in_progress` in memory.
4. Register AgentCore async-task tracking.
5. Create and retain the task.
6. Return the acknowledgement.

When a duplicate `start` arrives:

- return the existing terminal result if completed;
- return `in_progress` if the task is live;
- return `interrupted` if only a stale marker remains;
- never execute the user handler a second time implicitly.

Within one active execution environment, state mutation should remain on the
app's worker event loop so the claim transition can be made without a
cross-thread race. The filesystem claim should use an atomic primitive such as
exclusive file or directory creation rather than a separate existence check
and write. Persistent markers improve recovery and idempotency but do not
constitute a distributed transaction or a strict exactly-once guarantee.

Duplicate detection is based on invocation identity, not payload equality. Two
requests with identical payloads and different invocation IDs are two valid
executions. Reusing an invocation ID with a different request should return a
conflict; an implementation may persist a sanitized request fingerprint to
detect this without retaining the full payload.

The storage layout and wire protocol permit multiple invocation records within
one Runtime session. The first implementation may serialize active
invocations, reject overlapping work, or support concurrency according to
verified Runtime and application constraints. That policy must be separate
from deduplication and must not redefine the Runtime session as an invocation
ID.

Similarly, two invocations may carry the same conversation ID to represent
follow-up turns. Ordering, concurrent turns, branching, and history updates are
application concerns. A conservative application may serialize invocations
within one conversation while allowing work in independent conversations.

### Foreground streaming

Foreground mode should preserve upstream support for generators,
async generators, and direct Starlette `Response` objects.

Background mode requires a bounded, serializable terminal result. The initial
version should reject streaming/generator results in background mode with a
clear error rather than silently buffering an unbounded stream or returning a
result that cannot be reconstructed.

## Client-side design

### Foreground

For `background=false`, the client:

1. Generates or accepts a `runtimeSessionId`.
2. Generates or accepts an invocation ID before network submission.
3. Optionally attaches an opaque conversation ID.
4. Sends one `InvokeAgentRuntime` request.
5. Maintains the connection until completion.
6. Parses and returns the final response.

If the connection is interrupted, the first version does not promise that a
foreground result can be recovered. Persisted foreground results can be added
later without changing the connection-mode API. Until then, a foreground
invocation ID is correlation metadata rather than a durable retrieval or
deduplication handle.

### Background

For `background=true`, the client:

1. Generates or accepts a `runtimeSessionId`.
2. Generates or accepts an invocation ID before network submission.
3. Optionally attaches an opaque conversation ID.
4. Sends an internal `start` request.
5. Receives an `in_progress` acknowledgement.
6. Returns a `RolloutFuture` exposing the invocation ID.
7. Implements `done` and `result` through internal `get` operations using the
   same `runtimeSessionId` and invocation ID.

Generating the invocation ID before the network call means an ambiguous
transport failure does not erase the execution identity. Internal SDK retries
reuse that ID. A caller that persists the handle can later reattach using the
pair of Runtime session ID and invocation ID.

The public future should not expose which same-session data-plane operation is
used to read invocation state, or whether a future direct session-storage API
is available.

### Waiting, cancellation, and session lifecycle

The client should keep three operations separate:

- **Stop waiting**: `result(timeout=...)` stops waiting when its timeout
  expires. By default, the remote invocation continues running.
- **Cancel the invocation**: a cooperative internal operation cancels the
  background task and persists `cancelled` without terminating the session.
- **Stop the session**: `StopRuntimeSession` terminates the current execution
  environment. This is a separate, explicit lifecycle action.

The RL rollout client may offer a policy such as "stop the session after a
terminal result" because its current training model allocates one session per
rollout. That policy should be configurable and should not define the generic
Runtime behavior. A caller using sticky routing for follow-up turns must be
able to retrieve one result and continue invoking the same session.

### Retrieval path

Storage backend and retrieval path are separate decisions:

```text
Storage backend
  - managed session storage
  - S3
  - future alternatives

Retrieval path
  - same-session data-plane operation
  - direct S3 read
  - future direct session-storage API
```

For the default managed session-storage backend, the initial result path is:

```text
RolloutFuture
  -> same-session data-plane operation(
       runtimeSessionId=<same session>,
       invocationId=<target invocation>)
  -> read the invocation manifest
  -> return status/result
```

Managed session storage is exposed to the application as a filesystem mount
during an invocation. There is currently no documented external API for a
client to read an arbitrary session-storage path directly.

The initial same-session transport remains an implementation decision:

- an internal `InvokeAgentRuntime` operation intercepted before the user
  handler; or
- `InvokeAgentRuntimeCommand` running a deterministic SDK helper that reads the
  invocation manifest and emits a structured result.

The selected transport must be validated against the completed-before-polling
case: after idle termination, it must either reactivate the same Runtime session
and read its managed storage or provide another documented path to the same
state.

The client should keep this mechanism internal. If AgentCore later exposes
direct session-storage reads, `RolloutFuture` can switch to direct access
without changing the public client API or persisted result format.

When S3 is selected as the storage backend, the client can retrieve invocation
state directly from S3 without activating the Runtime session. This is an
alternative backend and retrieval path, not merely a v1 compatibility mode.

Retrieval uses an internal handle containing at least:

```text
runtime_session_id
invocation_id
```

An optional conversation ID may be retained as correlation metadata, but it is
not sufficient to retrieve or cancel a particular invocation.

### Rate limiting

The current client uses one 25 TPS limiter for invocation and session stop.
That reflects an older Runtime quota model.

The current [AgentCore Runtime quotas](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/bedrock-agentcore-limits.html#runtime-service-limits)
separate:

- a shared data-plane request rate of 1,000 TPS across
  `InvokeAgentRuntime`, command APIs, WebSocket APIs,
  `StopRuntimeSession`, and related data-plane operations;
- a shared new Runtime session creation rate of 25 TPS.

The client should model both constraints:

- every data-plane request consumes the data-plane limiter;
- a request that may provision a new session also consumes the
  session-creation limiter;
- polling a known-active session normally consumes only the data-plane limiter.

Both `InvokeAgentRuntime` and command-style APIs can participate in session
creation or resume paths depending on session state. The limiter decision
should be based on whether the client knows the session to be active, not only
on the API method name.

Local client limiters are still best-effort because AWS quotas are shared across
processes and callers in the account. SDK retry and backoff behavior remains
necessary.

## Session lifecycle and recovery

### Completed before polling with managed session storage

If the default managed session-storage backend is selected and a background
invocation finishes before the client polls:

1. The terminal result has already been persisted.
2. `complete_async_task` allows the execution environment to become idle.
3. AgentCore may terminate that environment.
4. A later retrieval operation targets the same `runtimeSessionId` and
   reactivates the session through the selected transport.
5. The retrieval path reads the terminal file for the requested invocation ID
   and returns the result.

This is the primary reason terminal results cannot live only in memory. With an
S3 backend, the client can instead read the persisted state directly without
reactivating the Runtime session.

### Crash before terminal persistence

If the execution environment or process fails after writing the start marker
but before writing the terminal result, a later `get` observes:

- no live in-memory task;
- a persistent start marker for the requested invocation ID;
- no terminal result.

The app returns `interrupted`. It does not rerun the handler automatically.

### Failure to persist the terminal result

Persistence failure is itself a terminal-delivery failure. The app must not
mark the AgentCore async task complete and report success before durable storage
succeeds.

The exact behavior when the selected storage backend becomes unavailable needs
an implementation decision:

- keep the task busy while retrying persistence within a bounded policy;
- persist a minimal failure record if possible;
- otherwise log the delivery failure and allow client timeout/recovery
  semantics to surface it.

### Storage retention

Retention follows the selected storage backend. Managed session storage has its
own service lifecycle and is not a permanent application database. Applications
needing archival, cross-session analytics, or retention independent of the
Runtime lifecycle may select S3 or export results to another data system.

## Security model

Managed session storage improves the current result-delivery security model:

- one session cannot browse another session's mounted files;
- the app no longer needs broad access to a shared customer result bucket;
- users do not need to construct per-session S3 prefix policies;
- result retrieval is scoped by `runtimeSessionId` and the targeted invocation
  ID.

These properties apply to the default managed session-storage backend. An S3
backend uses the configured bucket, IAM policy, and key structure as its
security and isolation boundary; the SDK should not imply that S3 objects gain
AgentCore session isolation automatically.

The boundary is session isolation, not process-internal trust. A user handler,
tool, or shell with arbitrary filesystem access inside the same session may
read or alter SDK metadata. The proposal should not claim tamper resistance
against code executing within the session.

Sensitive fields should be minimized in persisted manifests. In particular,
model API keys and full payloads should not be copied into result metadata by
default.

## Generalizing the app surface

The durable invocation behavior is not specific to reinforcement learning.
The current class and decorator names should therefore be generalized after
the core protocol is established.

### Proposed class name

`AgentCoreRuntimeApp` describes the scope more accurately:

- it extends AgentCore Runtime invocation behavior;
- it does not claim to wrap AgentCore Memory, Gateway, Identity, or other
  services;
- it remains a narrow subclass of upstream `BedrockAgentCoreApp`, not an
  "advanced" replacement SDK.

Illustrative usage:

```python
from agentcore_rl_toolkit import AgentCoreRuntimeApp

app = AgentCoreRuntimeApp()


@app.entrypoint
def handler(payload: dict, context):
    return {"result": "..."}
```

### Keep the decorator consistent with upstream

The public decorator should be `@app.entrypoint`, matching
`BedrockAgentCoreApp`.

`AgentCoreRuntimeApp.entrypoint` can wrap the user handler with the internal
start/get and foreground/background protocol, then register that wrapper via
`super().entrypoint`.

The invocation mode is selected per request. It is not a property of the
decorated function, so a separate `rollout_entrypoint` name is unnecessary.

### Compatibility

The migration can preserve existing applications:

- `AgentCoreRLApp` remains as a compatibility alias or thin subclass of
  `AgentCoreRuntimeApp`.
- `rollout_entrypoint` remains temporarily as an alias for `entrypoint`.
- `RolloutClient` continues to inject `_rollout` model and trainer
  configuration.
- examples and documentation move to `AgentCoreRuntimeApp` and
  `@app.entrypoint` gradually.

The repository as a whole remains RL-oriented because it also contains reward
helpers, rollout gateways, and trainer integrations. Only the Runtime
application and durable invocation protocol become generic.

### Potential upstream destination

Long term, this capability may belong in
[`bedrock-agentcore-sdk-python`](https://github.com/aws/bedrock-agentcore-sdk-python)
because foreground/background invocation and managed session-storage delivery
are general Runtime features.

This repository can first provide the design and a reference implementation.
If the capability moves upstream later, `AgentCoreRuntimeApp` can become a
compatibility export and RL integrations can consume the upstream
implementation.

## Relationship to existing rollout components

### Generic managed-invocation protocol

- foreground/background connection behavior;
- internal start/get protocol;
- invocation identity and lifecycle;
- optional opaque conversation correlation;
- state machine;
- managed session-storage persistence;
- result retrieval and cancellation;
- Runtime quota handling;
- batch invocation primitives.

### Rollout Execution SDK concerns

- `_rollout` payload fields such as model endpoint, model ID, sampling
  parameters, and trajectory-capture session key;
- reward conventions;
- rollout gateway trajectory capture;
- training backend integrations;
- grouping multiple rollouts by training input;
- trainer-specific timeout and failure handling.

The generic layer should not require a `rewards` key or know how a trainer
interprets the returned result.

Conversation history is neither a managed-invocation protocol responsibility
nor a rollout-specific feature of this repository. Applications and agent
frameworks remain free to keep it in managed session storage, another memory
service, a database, or their own payloads. The protocol only ensures that its
invocation model does not prevent multiple conversations, follow-up turns, or
application-defined branching.

## Implementation phases

### Phase 1: protocol and storage

- Define the internal start/get request and response envelopes.
- Add a client-generated invocation ID to every logical call.
- Add managed session-storage configuration and path handling.
- Implement per-invocation persistent start markers and terminal results.
- Implement live task tracking and persistent state recovery.
- Add focused tests for duplicate start, polling, completion, failure, and
  interrupted recovery.
- Verify that identical payloads with distinct invocation IDs execute
  independently and that reusing one ID with a conflicting request fails.

### Phase 2: foreground/background client

- Add `background` to sync and async invocation methods.
- Accept an optional opaque conversation ID without adding conversation
  storage or history APIs.
- Implement foreground direct return.
- Implement same-session background retrieval.
- Separate data-plane and session-creation rate limiting.
- Separate wait timeout, cooperative invocation cancellation, and optional
  session cleanup policies.

### Phase 3: generalize naming

- Introduce `AgentCoreRuntimeApp`.
- Override `entrypoint` with the managed invocation wrapper.
- Keep `AgentCoreRLApp` and `rollout_entrypoint` compatibility aliases.
- Move internal protocol fields out of `_rollout`.
- Update examples and documentation.

### Phase 4: alternative storage and export

- Implement S3 as an optional storage backend with direct result retrieval.
- Define optional result export after client retrieval.
- Document storage-backend trade-offs and migration from required S3
  configuration to the managed session-storage default.
- Validate Runtime version, session timeout, and storage-retention behavior in
  live AgentCore integration tests.

### Phase 5: possible upstreaming

- Evaluate whether the generic app and client behavior should be proposed to
  `bedrock-agentcore-sdk-python`.
- Keep RL-specific adapters and trainer integrations in this repository.

## Open questions

- Should `background` default to `true` for compatibility or `false` for
  ordinary request/response semantics?
- Should successful foreground non-streaming results also be persisted for
  recovery after a client disconnect?
- What is the exact internal protocol namespace and versioning scheme?
- What should the public API be for cooperative invocation cancellation versus
  explicit session termination?
- Should `RolloutClient` retain automatic session stop as its default for
  one-shot training rollouts, or make it opt-in?
- Should the first implementation serialize active invocations per Runtime
  session, reject overlap, or permit concurrency?
- Should initial same-session retrieval use an internal `InvokeAgentRuntime`
  operation or an `InvokeAgentRuntimeCommand` helper?
- What bounded retry policy should apply when terminal-result persistence
  fails?
- Should the optional S3 storage backend ship in the first v2 release or a
  later phase?
- How should the app verify that the configured mount is managed session
  storage rather than an arbitrary writable directory?
- When a previously stopped session is resumed for retrieval, which quota
  buckets does AgentCore apply in practice?
- Does the first generic client remain named `RolloutClient`, or should a
  generic client surface be introduced only if the capability moves upstream?
