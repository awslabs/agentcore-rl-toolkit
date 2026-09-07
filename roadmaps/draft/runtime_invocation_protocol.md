# Runtime Invocation Protocol: an execution lifecycle for stateful, long-running workloads

## Summary

Stateful, long-running workloads separate three lifetimes that a simple
request/response API can treat as one:

- the client connection;
- one logical invocation; and
- the compute session in which that invocation runs.

AgentCore Runtime sessions provide sticky routing, isolation, storage, and
state continuity across invocations. An individual invocation may outlive its
initial client connection, while the Runtime session may outlive that
invocation and accept follow-up work. A `runtimeSessionId` therefore cannot
also serve as the identity and lifecycle of every execution inside the session.

This proposal defines an explicit invocation lifecycle nested within a Runtime
session. Every invocation has its own ID, status, terminal result, retry
semantics, and cancellation target. Foreground and background are two ways to
interact with that same lifecycle:

- `background=false` starts the invocation and waits on the initial
  connection; and
- `background=true` starts the invocation and returns a handle before it
  finishes.

Background state and terminal results must survive the loss of process-local
state. AgentCore managed session storage is the proposed default persistence
backend. S3 remains a valid alternative when an application needs direct
external retrieval, independent retention, larger artifacts, or cross-session
access.

The protocol is not an RL-specific abstraction. It can support both:

- the Rollout SDK, which submits long-running agent rollouts for training and
  evaluation; and
- the Sandbox SDK, which executes commands in isolated task environments and
  needs the same foreground, detached, polling, cancellation, and recovery
  semantics.

The two SDKs retain their workload-specific APIs. The protocol supplies the
shared execution contract underneath them.

## Motivation: the missing invocation lifecycle

The protocol is most valuable where stateful compute and long-running work
meet. Two simpler compute models often avoid the problem by collapsing or
externalizing the invocation lifecycle:

| Compute model | Typical lifecycle |
| --- | --- |
| Request-scoped compute | The request, execution, and compute lifetime largely coincide. A short foreground response is often sufficient. |
| Jobs or long-lived services | Work is already represented by a detached job, process, or application-defined resource whose lifetime is independent of the submission connection. |
| AgentCore Runtime | A stateful, sticky Runtime session can host multiple invocations, while its elastic execution environment can idle, stop, and resume. Neither the connection nor the session uniquely identifies one execution. |

AgentCore Runtime sits between request-scoped serverless compute and a
traditional long-lived service or job system. Its session is a useful
stateful-compute boundary, but it is too broad to be an execution handle:

- one session may contain sequential follow-up invocations;
- one session may contain multiple application-managed conversations;
- an invocation may finish while the session remains available;
- a client may stop waiting while the invocation continues;
- the execution environment may become idle before the result is retrieved;
  and
- retrying a network request must not implicitly create a second execution.

The missing abstraction is therefore:

> An invocation with a lifecycle independent of both its client connection and
> its Runtime session.

Foreground and background delivery follow from that abstraction. They are not
two execution lifecycles. Foreground means start and wait; background means
start and return the invocation handle.

## Reference pattern: OpenAI Responses

The
[OpenAI Responses API background mode](https://developers.openai.com/api/docs/guides/background/)
provides an existing example of this separation:

- foreground and background requests create the same kind of logical
  execution;
- a background request returns an addressable response before processing is
  complete;
- the client can retrieve its status and output later;
- cancellation targets that response rather than the conversation or the
  underlying compute; and
- connection lifetime is separate from execution lifetime.

In this pattern, `background` controls result delivery and whether the initial
connection waits. It does not select a different user handler or a different
application result.

Responses also separates logical history from execution identity:

- a Conversation is an optional history container; and
- each Response is a distinct execution with its own ID and lifecycle.

This proposal borrows that separation:

- `conversation_id` optionally tells the application which conversation or
  thread an invocation belongs to; and
- `invocation_id` identifies one execution.

RIP passes the conversation ID through but does not store or interpret the
conversation's messages or history.

AgentCore additionally exposes `runtimeSessionId`, which has no direct public
Responses equivalent. It controls sticky routing and represents an
infrastructure scope rather than a logical application turn.

One way to think about a Runtime session is as a stateful machine that a user
can reconnect to and run different workloads on. Similarly,
`runtimeSessionId` routes work to the same execution environment, while
`invocation_id` identifies one particular execution within that environment.
The analogy is about the separation of identities, not permanence: an
AgentCore execution environment remains elastic and may stop and resume.

This proposal borrows the execution-resource pattern, not the full Responses
API or control plane. ART does not have a separate Responses-style control
plane. Its initial implementation must persist invocation state inside the
selected storage backend and retrieve it through an available AgentCore
data-plane path.

The durable invocation record is the closest equivalent to a Response
resource: it has an ID, lifecycle status, terminal output, and cancellation
target. It is not a new conversation database or a claim that ART already has
an independently queryable control-plane resource.

## Intended destination and the name

ART is a suitable place to incubate this protocol because the Rollout and
Sandbox SDKs both need it now. ART should not be its permanent ownership
boundary.
Invocation identity, foreground/background delivery, durable status, result
retrieval, and cancellation are general AgentCore Runtime capabilities rather
than training or evaluation concepts.

The intended destination is therefore the
[official AgentCore Python SDK](https://github.com/aws/bedrock-agentcore-sdk-python/tree/main)
or the AgentCore service itself. ART can provide the initial design,
implementation, and workload evidence. Once equivalent semantics exist
upstream, the Rollout and Sandbox SDKs should consume that capability instead
of preserving an ART-specific task manager.

This intended lifecycle motivates the name **Runtime Invocation Protocol**,
shortened from this point onward to **RIP**:

- **Runtime** limits the scope to execution on AgentCore Runtime rather than
  AgentCore Memory, Gateway, Identity, or other services.
- **Invocation** includes both `InvokeAgentRuntime` and
  `InvokeAgentRuntimeCommand`: one logical handler or process execution.
- **Protocol** means the shared identities, operations, state transitions,
  persistence rules, and request and response envelopes observed across
  clients and execution adapters. It does not require one AWS API or one
  byte-level wire format.

The acronym is deliberate: if the generic capability is absorbed upstream, the
ART-local implementation should be allowed to RIP. Its Rollout and Sandbox
consumers should survive; the duplicated protocol implementation should not.

## Existing ART evidence

ART already has two execution-facing SDK surfaces:

| Consumer | Execution target | Current behavior | Missing or coupled behavior |
| --- | --- | --- | --- |
| Rollout SDK | An application handler reached through `InvokeAgentRuntime` | Detaches every handler, writes results to S3, and returns a `RolloutFuture` | Foreground delivery, per-invocation identity, storage-independent retrieval, sticky-session follow-up, and separation from RL naming |
| Sandbox SDK | A shell command reached through `InvokeAgentRuntimeCommand` | Streams one command synchronously and returns `ExecResult` | Detached execution, durable handles, polling, cancellation, reconnection, and recovery |

These surfaces use different AgentCore APIs, but need the same answers to:

- What identifies one execution independently of its Runtime session?
- Does the initial connection wait for the result?
- How does a client retrieve a result after disconnecting?
- What happens when a submission is retried?
- How are status, failure, cancellation, and interruption represented?
- When may the Runtime session stop?
- Which state must survive execution-environment replacement?

RIP defines those semantics once. An app-handler adapter may carry RIP through
`InvokeAgentRuntime`; a command adapter may carry it through
`InvokeAgentRuntimeCommand`.

### Rollout SDK

The current **Rollout SDK** behavior is implemented by:

- `AgentCoreRLApp`;
- `@app.rollout_entrypoint`;
- `RolloutClient`; and
- `RolloutFuture`.

It solves two important rollout problems:

1. A long-running agent rollout continues after its initial
   `InvokeAgentRuntime` connection returns.
2. A trainer can retrieve the result later without maintaining one TCP
   connection per rollout.

The server-side decorator detaches the handler, marks the app busy through
AgentCore async-task tracking, writes a terminal result to a customer-managed
S3 bucket, and immediately returns the result location. The client submits
rollouts, polls S3, handles concurrency and timeouts, and normally stops the
Runtime session after retrieving the result.

This was an effective way to establish background rollout execution, but it
combined several concerns:

```text
Rollout SDK today
  ├── generic Runtime app wrapper
  ├── background task lifecycle
  ├── S3 persistence and polling
  ├── Runtime session lifecycle
  ├── rollout payload configuration
  └── trainer-facing batch behavior
```

Only the last two concerns are inherently rollout-facing. The rest describe a
general Runtime execution lifecycle.

The Rollout SDK does not itself capture token-level training trajectories.
Training backend integrations create and finish Rollout Gateway capture
sessions and correlate the resulting `TraceRecord`s with trainer samples.
`RolloutClient` only transports rollout configuration and opaque metadata
needed by those integrations.

### Sandbox SDK

The Sandbox SDK provides `SandboxClient`, `Sandbox`, and `ExecResult`. It starts
or attaches to a Runtime session and executes shell commands through
`InvokeAgentRuntimeCommand`.

Today, `Sandbox.exec()` is foreground-only: the client maintains the command
stream until the process exits or times out. Planned detached execution needs a
handle that can survive client disconnects and support status, result,
cancellation, and reattachment.

That is the same lifecycle problem as a background rollout, even though the
execution target is a process rather than an application handler. The command
API itself is also an invocation API: `InvokeAgentRuntimeCommand` begins a
logical unit of work in a Runtime session.

The implementations need not be identical. For example, app cancellation may
cancel an asyncio task, while command cancellation may terminate a process
group. RIP defines their common observable semantics.

## Problems addressed

| Current behavior or gap | Consequence | RIP direction |
| --- | --- | --- |
| Rollout execution is always detached. | Short or interactive calls pay persistence and polling costs unnecessarily. | Support foreground and background delivery as modes of the same execution. |
| Sandbox commands are always attached to their initial stream. | Long commands cannot be safely detached and reattached. | Give detached commands the same durable invocation handle. |
| `runtimeSessionId` also acts as the rollout execution identity. | Multiple calls and follow-ups in one sticky session cannot be addressed independently. | Add one invocation ID per logical execution. |
| S3 storage, polling, and background execution are combined in `rollout_entrypoint`. | Connection mode cannot evolve independently from storage or retrieval. | Separate execution mode, persistent state, storage backend, and retrieval path. |
| Rollout results require a customer-managed bucket and policy. | Users configure shared storage even when session-scoped state is sufficient. | Default to managed session storage while retaining S3 as an alternative backend. |
| S3 prefixes are conventions under one Runtime role. | They do not provide session isolation from code running with broad bucket access. | Scope the default execution state to managed session storage. |
| There is no explicit `start` versus `get` operation. | A polling request could accidentally execute the workload again after process-local state is lost. | Make retrieval an operation that can never enter the user workload. |
| Payload equality is easily confused with duplicate execution. | Two intentionally identical requests may be rejected, or retries may run twice. | Deduplicate by invocation identity, never by payload equality alone. |
| Invocation completion currently implies rollout session cleanup. | Automatic `StopRuntimeSession` prevents sticky-session follow-up. | Separate invocation completion, wait timeout, cancellation, and session termination. |
| Runtime-specific mechanics are named `AgentCoreRLApp` and `rollout_entrypoint`. | A general capability appears specific to RL. | Move generic behavior into a Runtime-level app and protocol layer. |
| Rollout and Sandbox clients would otherwise implement detached lifecycle independently. | Identity, recovery, and cancellation semantics can drift. | Reuse the RIP contract through workload-specific adapters. |

## Goals

- Define one invocation lifecycle for app handlers and commands.
- Support foreground and background result delivery without changing workload
  semantics.
- Keep Python sync versus async APIs independent from remote foreground versus
  background execution.
- Give every execution an invocation ID distinct from its Runtime session.
- Allow optional conversation correlation without managing conversation
  history.
- Make retries idempotent with respect to invocation identity.
- Persist all background state required after process or execution-environment
  replacement.
- Use managed session storage as the default persistence backend while
  allowing S3 and future alternatives.
- Keep storage backend and retrieval path replaceable.
- Separate invocation cancellation from Runtime session termination.
- Let Rollout and Sandbox expose domain-appropriate clients and handles over
  the same lifecycle contract.
- Make the generic capability suitable for eventual adoption by the official
  AgentCore SDK.

## Non-goals

- Reproducing the complete OpenAI Responses or Conversations APIs.
- Creating, storing, listing, or interpreting conversation history.
- Defining how applications merge, branch, or serialize conversation turns.
- Defining rollout trajectory capture, reward computation, or trainer sample
  construction.
- Automatically rerunning interrupted workloads with side effects.
- Persisting or replaying unbounded streaming output in background mode.
- Treating process-local Python variables as a supported background result
  backend.
- Protecting protocol files from arbitrary code running inside the same
  Runtime session.
- Solving Sandbox image provisioning, health-shim port conflicts, interactive
  shell design, or file transfer.

## Identity model

RIP distinguishes three identities:

- **Runtime session ID**: the AgentCore compute, sticky-routing, isolation,
  storage, resource, and lifecycle scope.
- **Conversation ID**: optionally tells the application which conversation or
  thread an invocation belongs to. RIP passes the ID through but does not store
  or interpret its messages or history.
- **Invocation ID**: one execution of an application handler or command.

The default conversation rule is:

```python
effective_conversation_id = conversation_id or runtime_session_id
```

This preserves the common AgentCore mental model in which one Runtime session
contains one implicit conversation while leaving room for multiple
application-managed conversations:

```text
Runtime session S
  ├── default conversation namespace (conversation ID = S)
  │     ├── invocation 1
  │     └── invocation 2
  └── explicit conversation namespaces (optional)
        ├── conversation A
        │     └── invocation 3
        └── conversation B
              └── invocation 4
```

Every follow-up is a new invocation, even if it reuses the same Runtime session
and conversation ID. Reusing an invocation ID means "this is another attempt
to address or submit the same execution."

Invocation identity and idempotency are conceptually different:

- the invocation ID names the resulting execution resource; and
- an idempotency key identifies repeated attempts to create that resource.

The first implementation may use the client-generated invocation ID for both
purposes. This avoids exposing two identifiers before ART has a separate
control plane, while preserving the conceptual distinction for future
evolution.

Duplicate detection is based on invocation identity, not payload equality.
Two identical payloads with different invocation IDs are two valid executions.
Reusing one invocation ID with a conflicting request should return a conflict.

## Core protocol

### Foreground and background modes

Foreground and background describe connection and result-delivery behavior:

- **Foreground**: the initial connection waits for a terminal result.
- **Background**: the initial connection returns before execution completes.

They do not describe whether the local Python client is synchronous or
asynchronous, whether an app handler is a sync or async function, or whether
the execution target is an app handler or command.

The generic contract is:

```python
# Illustrative shape, not a final class name.
result = client.invoke(request, background=False)

handle = client.invoke(request, background=True)
result = handle.result(timeout=600)
```

An asyncio client can expose the same remote modes:

```python
# The caller's event loop remains non-blocking, while the remote connection
# waits for completion.
result = await client.invoke_async(request, background=False)

# Submission and later waiting are both non-blocking to the caller's loop.
handle = await client.invoke_async(request, background=True)
result = await handle.result_async(timeout=600)
```

Each workload-facing SDK can choose domain-appropriate names. For example,
Rollout may retain `invoke()` and `RolloutFuture`, while Sandbox may expose
`exec()` for foreground commands and a future `spawn()` or detached-execution
handle. Those API names do not change the RIP lifecycle.

### Logical operations

RIP needs three logical invocation operations:

- `start(invocation_id)`: claim and begin an execution if that ID does not
  already exist.
- `get(invocation_id)`: return current status or terminal output without
  entering the user workload.
- `cancel(invocation_id)`: request cancellation of that execution without
  terminating the Runtime session.

Stopping a Runtime session is a fourth, separate lifecycle operation. It is not
invocation cancellation.

The concrete AWS operation carrying each logical operation is an adapter
decision. A protocol envelope carried through app payloads should use a
reserved namespace separate from application data and from `_rollout`, which
is rollout-specific configuration. Appendix A sketches how ART can initially
implement these operations without making that implementation part of the
normative protocol.

### State machine

Every invocation exposes at least:

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

`interrupted` means durable state shows that execution started, but no live
execution and no terminal result can be recovered. RIP must not automatically
rerun an interrupted workload because app handlers and commands may have side
effects.

Adapters may expose additional transient detail, but consumers must be able to
reason using this minimum state model.

### Start and retry behavior

The execution adapter must claim an invocation ID before beginning the user
workload. A repeated `start` for the same ID:

- returns the existing result when terminal;
- returns `in_progress` when the execution is still live;
- returns `interrupted` when only a stale start record remains; and
- never implicitly starts the workload a second time.

The initial client generates the invocation ID before the network request.
Therefore, an ambiguous connection failure does not erase the intended
execution identity. Internal retries reuse the same ID.

Most users should not need to provide invocation IDs manually. Advanced callers
may supply or persist one when they need to transfer a handle between
processes, retry after losing local state, or reattach from another client.

### Waiting, cancellation, and session lifecycle

RIP keeps three actions distinct:

- **Stop waiting**: a local timeout returns control to the caller. The remote
  execution continues by default.
- **Cancel the invocation**: the execution adapter cooperatively stops the
  targeted app task or command and persists `cancelled`.
- **Stop the Runtime session**: `StopRuntimeSession` terminates the execution
  environment and may affect every invocation or conversation in that
  session.

A workload-specific SDK may add lifecycle policy. A one-shot rollout client can
optionally stop its session after terminal result retrieval. That policy must
not define generic RIP behavior or prevent sticky-session follow-up.

### Multiple invocations

RIP allows multiple active invocations within the same Runtime session. For
example, two concurrent requests may belong to separate conversation threads
while sharing the same compute environment. Different invocation IDs identify
them as distinct executions. RIP does not serialize requests or isolate
application state: preventing concurrent invocations from corrupting shared
files, in-memory objects, or conversation history remains the application's
responsibility.

### Streaming

Foreground adapters should preserve the native streaming behavior of their
underlying operation where practical.

Background mode requires bounded, reconstructable output. The first
implementation should reject unsupported unbounded streams rather than
silently buffering them. A Sandbox command adapter may persist bounded stdout
and stderr or store larger output as artifacts under an explicit policy.

## Persistence and retrieval

### Durable background state

Process-local tasks, process IDs, and caches are useful operational state, but
they cannot be the authority for background execution. A Runtime environment
may become idle, terminate, or be replaced before the client polls.

The selected persistence backend must store:

- a start record written before the background acknowledgement; and
- one terminal record containing the result, structured failure, or
  cancellation metadata.

An illustrative managed-storage layout is:

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

The exact paths and filenames are private implementation details. Readers must
not observe partially published terminal state.

A start record can include:

- protocol version;
- invocation ID;
- creation time;
- effective conversation ID and whether it was explicit;
- execution-adapter type; and
- sanitized correlation metadata.

A terminal record can include:

- terminal status;
- completion time;
- a JSON-serializable result or structured error;
- cancellation or interruption metadata; and
- protocol version.

The complete input payload should not be persisted by default. Rollout payloads
may contain model credentials, and Sandbox commands may contain sensitive
arguments or environment values.

### Storage backend versus retrieval path

Persistence and retrieval are separate choices:

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

AgentCore
[managed session storage](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-filesystem-configurations.html)
is the proposed default because it scopes state to a Runtime session and avoids
requiring a customer-managed result bucket and policy. It is currently a
Preview feature on microVM runtimes: the documented lifecycle survives
stop/resume, expires after 14 idle days, and resets on a Runtime version update.
RIP must therefore treat it as durable execution state, not permanent archival
storage.

There is currently no documented external API for reading an arbitrary managed
session-storage path directly. The initial managed-storage retrieval path
therefore needs a same-session data-plane adapter. Candidate implementations
include:

- an internal `InvokeAgentRuntime` operation intercepted before the user
  handler; or
- `InvokeAgentRuntimeCommand` running a deterministic helper that reads and
  emits the invocation record.

This proposal intentionally does not select between them without live
validation. In particular, the chosen path must work when execution completes
and the environment becomes idle before the first poll.

S3 object storage remains a valid alternative backend, not merely a
compatibility path for the current Rollout SDK. It enables direct external
reads without reactivating session compute and may better serve large
artifacts, archival retention, or cross-session workflows.

The public handle should expose neither the storage backend nor the retrieval
transport. If AgentCore later provides a direct session-storage API, clients
can switch retrieval paths without changing invocation identity or workload
APIs.

### Recovery

If a client polls after the original execution environment has become idle,
the retrieval adapter targets the same Runtime session and reads its durable
invocation record. If the backend contains a terminal result, it is returned
without rerunning the workload.

If the backend contains a start record but no live task or process and no
terminal result, the invocation becomes `interrupted`.

Terminal persistence must complete before an app adapter releases AgentCore
async-task tracking or a command adapter releases its equivalent busy lease.
Otherwise, the environment may become idle before the result is durable.

The behavior when terminal persistence repeatedly fails requires a bounded
retry and failure policy. The adapter must not report successful delivery
before durable publication succeeds.

### Security boundary

Managed session storage improves isolation relative to a shared result bucket:

- invocation state is scoped to the Runtime session;
- applications do not need broad access to one shared bucket; and
- users do not need to construct per-session S3 prefix policies.

This is session isolation, not protection from code inside the session. A user
handler or shell command with arbitrary filesystem access in the same
environment may read or alter protocol files. RIP must not claim tamper
resistance against the workload it hosts.

An S3 backend instead relies on its configured IAM policy, bucket, and key
layout. Selecting S3 does not give those objects AgentCore session isolation.

### Quota model

The current
[AgentCore Runtime quota table](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/bedrock-agentcore-limits.html#runtime-service-limits)
defines:

- a shared data-plane request rate of 1,000 TPS across
  `InvokeAgentRuntime`, `InvokeAgentRuntimeCommand`, WebSocket and shell
  operations, `StopRuntimeSession`, and related data-plane APIs; and
- a shared new Runtime session creation rate of 25 TPS.

RIP clients should model these as separate constraints:

- every request consumes data-plane capacity; and
- a request that creates a new session also consumes session-creation
  capacity.

Whether a request can create a session is not determined solely by whether its
API name contains `Command`. Both app and command adapters begin with an
`Invoke...` operation and can target a new session. Exact resume and quota
behavior should be verified in live integration tests rather than inferred
from the transport name.

Client-side limiters remain best-effort because quotas are shared across
processes and callers. Retry and backoff are still required.

## Execution adapters

### App-handler adapter

The app-handler adapter carries RIP through `InvokeAgentRuntime`. A wrapper
around the registered `BedrockAgentCoreApp` entrypoint intercepts protocol
operations before they reach the user handler:

```text
InvokeAgentRuntime
  -> RIP envelope
  -> start / get / cancel dispatch
  -> registered application handler for start only
```

Foreground execution waits for the registered handler. Background execution
retains the handler task, persists its terminal state, and uses AgentCore
async-task tracking to keep the session busy. Both modes continue to use the
upstream sync/async handler dispatch. Appendix A describes the initial
in-process registry, persistence ordering, and cancellation limits.

### Command adapter

The command adapter carries RIP through `InvokeAgentRuntimeCommand` and an
in-container execution manager:

```text
InvokeAgentRuntimeCommand
  -> foreground command stream
  or
  -> RIP-aware detached launcher
       -> child process
       -> durable status and output
```

Foreground commands continue using the native command stream. Detached
commands need a process owner that persists status and output and maps
cancellation to process-group termination. The current health shim may support
ART validation, but it is not the permanent protocol boundary. Appendix A
describes the prototype shape and the live validation it requires.

## Consumer 1: Rollout SDK

### Definition and scope change

After RIP is separated, the **Rollout SDK** is the ART training- and
evaluation-facing adapter that configures, submits, groups, and collects agent
rollouts over RIP.

Its scope changes as follows:

| Concern | Current owner | Proposed owner |
| --- | --- | --- |
| Runtime app wrapper and entrypoint dispatch | Rollout SDK | RIP app adapter |
| Foreground/background execution | Background-only Rollout SDK behavior | RIP |
| Invocation identity, status, result, retry, and cancellation | Implicit across `runtimeSessionId`, S3 key, and `RolloutFuture` | RIP |
| Background persistence and retrieval | Rollout SDK S3 implementation | RIP storage and retrieval adapters |
| Runtime quota handling | `RolloutClient` | Shared RIP client machinery |
| Default session cleanup policy | `RolloutFuture` always stops after retrieval or timeout | Rollout-specific configurable policy over RIP |
| Model endpoint, model ID, and sampling configuration | `_rollout` payload | Rollout SDK |
| Batch submission and grouping by training input | `RolloutClient` | Rollout SDK |
| Trainer-facing timeout and failure adaptation | `RolloutClient` and backend integrations | Rollout SDK and the relevant training backend |
| Trajectory capture and correlation | Rollout Gateway and training backends | Rollout Gateway and training backends |
| Reward computation and conversation history | Application or training integration | Application or training integration |

The resulting layering is:

```text
Training or evaluation integration
  -> Rollout SDK
       -> RIP client and app adapter
            -> AgentCore Runtime

Training backend
  <-> Rollout Gateway for trajectory capture
```

The Rollout SDK may pass opaque rollout metadata, but it does not create the
Rollout Gateway capture session, finish it, stamp `TraceRecord` identity, or
join captured trajectories to trainer samples.

### Concrete rollout flow

A background training rollout can use RIP as follows:

1. The training backend creates any Rollout Gateway capture state it needs.
2. The Rollout SDK builds the agent payload, including model and sampling
   configuration plus opaque metadata supplied by the backend.
3. The SDK starts a RIP invocation with `background=true`.
4. The app adapter persists `in_progress`, detaches the handler, and returns an
   invocation handle.
5. The agent performs model calls. If trajectory capture is enabled, the
   training backend and Rollout Gateway correlate those calls using their own
   capture identity.
6. RIP persists and exposes the agent handler's terminal result.
7. The training integration awaits the handle and independently finishes or
   drains the capture session.
8. The training backend combines the result, reward, and captured records
   according to its own sample contract.

RIP does not require its invocation ID to equal a Runtime session ID, gateway
session ID, rollout ID, training input ID, or conversation ID. An integration
may correlate them explicitly, but the protocol does not collapse their
meanings.

### Benefits to rollout users

- Training keeps the current fire-and-retrieve background model.
- Evaluation and interactive use can call the same handler in foreground mode.
- Managed session storage can remove mandatory S3 bucket and policy setup.
- A single sticky Runtime session can support follow-up invocations without
  treating result retrieval as session completion.
- Retries can address one execution without guessing from payload or S3 key.
- `RolloutClient` can focus on rollout configuration, batching, grouping, and
  training-facing policy instead of implementing a private task manager.

The default `background` and session-cleanup policies may remain rollout-
oriented for migration. For example, one-shot training can default to
background execution and optional stop-after-result while the generic RIP
default remains foreground and preserves the session.

## Consumer 2: Sandbox SDK

### Definition and boundary

The **Sandbox SDK** is the workload-facing client for starting or attaching to
an isolated task environment and executing commands, shells, files, and
processes within it.

RIP applies specifically to command execution lifecycle. It does not absorb:

- image adaptation or Runtime provisioning;
- Sandbox session creation and attachment APIs;
- interactive shell semantics;
- file transfer;
- task and verifier contracts; or
- the platform work needed to remove the current health-shim port conflict.

### Concrete Sandbox flows

Foreground execution remains simple:

```python
result = sandbox.exec("pytest -q", timeout=900)
```

The client holds the native command stream and returns `ExecResult`. RIP still
provides a distinct invocation identity for correlation and future recovery,
but the common path need not persist every foreground result.

Detached execution can be exposed through a Sandbox-specific API:

```python
# Illustrative naming only.
handle = sandbox.spawn("pytest -q", timeout=900)

status = handle.status()
result = handle.result(timeout=1200)
```

Underneath:

1. the Sandbox client creates an invocation ID;
2. the command adapter starts a RIP-aware detached command in the target
   Runtime session;
3. the command continues after the initial client connection returns;
4. status and terminal output are persisted in the session-scoped backend;
5. another client can reattach using the Runtime session ID and invocation ID;
   and
6. cancellation targets the command process group without necessarily stopping
   the Sandbox session.

### Benefits to Sandbox users

- Long builds, tests, and coding-agent harnesses do not depend on one command
  stream remaining connected.
- A detached command has a durable, transferable handle.
- Polling cannot accidentally rerun the command.
- Timeout can stop local waiting without destroying the environment.
- Command cancellation and Sandbox termination remain distinct.
- Rollout orchestration can use the same execution state model whether the
  workload is an agent handler or an in-sandbox harness process.

Sandbox does not become an RL SDK by using RIP. It remains a general task-
environment abstraction that can be consumed by evaluation, RL, or standalone
workloads.

## Ownership boundaries

| Layer | Owns | Does not own |
| --- | --- | --- |
| RIP | Invocation identity, foreground/background delivery, lifecycle state, durable background result, retry semantics, wait/cancel, storage and retrieval seams | Conversation history, rollout semantics, shell UX, trainer samples |
| Rollout SDK | Rollout payload and model configuration, batch submission, grouping, rollout-oriented defaults and result policy | Generic app task management, trajectory capture, conversation storage |
| Sandbox SDK | Sandbox sessions, command/shell/file UX, structured process results, Sandbox-specific handles | Agent payloads, rewards, trajectory capture |
| Rollout Gateway | Token-level trajectory capture and trace construction | Runtime invocation lifecycle and result delivery |
| Training backends | Capture-session orchestration, reward and trace joining, trainer-native sample construction | Generic Runtime execution protocol |
| AgentCore Runtime | Session routing, isolation, compute lifecycle, data-plane operations, managed storage substrate | ART workload semantics |

This boundary intentionally leaves conversation implementation to the
application or agent framework. RIP accepts an optional conversation ID so it
does not prevent multi-turn or multi-conversation designs, but it does not
manage messages or memory.

## Public surface and migration

The app-side generic class can initially be `AgentCoreRuntimeApp`, retaining
the upstream `@app.entrypoint` name. `AgentCoreRLApp` and
`rollout_entrypoint` can remain temporary migration aliases while examples
move to the generic app surface.

The generic client and handle names do not need to be finalized in this
proposal. `RolloutFuture` and a future Sandbox execution handle can wrap the
same internal RIP handle without exposing a new public `Retriever` abstraction.

## Implementation phases

### Phase 1: specify and validate the protocol

- Define versioned `start`, `get`, and `cancel` envelopes.
- Define invocation identity, conflict behavior, and the minimum state model.
- Define the persistent start and terminal records.
- Validate managed session-storage behavior across idle termination and
  session reactivation.
- Validate both candidate same-session retrieval transports.
- Test two identical requests with distinct invocation IDs and conflicting
  reuse of one invocation ID.

### Phase 2: app adapter and Rollout migration

- Introduce the generic app adapter and `AgentCoreRuntimeApp`.
- Add foreground/background modes to the client machinery.
- Add managed session storage as the default background backend.
- Preserve S3 as an optional backend and direct retrieval path.
- Separate wait timeout, invocation cancellation, and session cleanup.
- Refactor `RolloutClient` and `RolloutFuture` into rollout-facing wrappers
  over RIP.
- Keep rollout payload fields and trainer-facing batching outside the generic
  layer.

### Phase 3: command adapter and detached Sandbox execution

- Define the detached command launcher and process-group lifecycle.
- Persist command status, exit code, and bounded output.
- Add Sandbox-specific detached handles, polling, reattachment, and
  cancellation.
- Verify lifecycle behavior with the current health shim while keeping native
  platform support as the intended destination.

### Phase 4: hardening and upstreaming

- Separate data-plane and new-session rate limiting.
- Add live integration coverage for interruption, reactivation, persistence
  failure, and storage retention.
- Define artifact policies for results that exceed the manifest format.
- Evaluate moving the generic app, client, and protocol behavior into
  `bedrock-agentcore-sdk-python`.
- Keep Rollout, Sandbox, Rollout Gateway, and trainer-specific integrations in
  ART.

## Open questions

- Should the generic RIP default be `background=false` while Rollout preserves
  a `background=true` default for existing training behavior?
- Should successful foreground non-streaming results also be persisted for
  recovery after client disconnect?
- Should initial managed-storage retrieval use an internal
  `InvokeAgentRuntime` operation or an `InvokeAgentRuntimeCommand` helper?
- What is the public cancellation API for an app task versus a command process?
- What mechanism keeps a detached Sandbox command's session alive without
  making the current health shim the permanent design?
- What bounded retry policy applies when terminal-result persistence fails?
- Should S3 ship with the first RIP implementation or follow after managed
  session storage?
- Which result and artifact size limits belong in the protocol versus each
  workload adapter?
- When a stopped or idle session is targeted for retrieval, how does AgentCore
  account for new-session creation and resume quotas in practice?
- Which generic classes should be public in ART before the capability has an
  upstream home?

## Appendix A: Initial ART implementation sketch

This appendix is non-normative. It describes one way ART can validate RIP with
the current AgentCore Runtime and SDK surfaces. The protocol does not require
these private envelope names, file paths, or helper processes, and they should
be replaceable by upstream support.

### Client mapping

The logical operations do not need to become new public methods:

| Workload-facing operation | RIP operation |
| --- | --- |
| `client.invoke(...)` | `start` |
| `handle.status()`, `done()`, or `result()` | `get` |
| `handle.cancel()` | `cancel` |
| `StopRuntimeSession` or `sandbox.terminate()` | Session lifecycle, outside RIP invocation cancellation |

The client generates the invocation ID before submission and retains an
internal handle containing at least:

```text
runtime_session_id
invocation_id
execution_adapter
```

### Private app envelope

The app adapter can carry protocol metadata in a reserved top-level namespace
while leaving the rest of the application payload unchanged:

```json
{
  "_agentcore_runtime": {
    "version": 1,
    "operation": "start",
    "invocation_id": "inv-123",
    "background": true,
    "conversation_id": "conversation-a"
  },
  "prompt": "..."
}
```

`get` and `cancel` use the same envelope without application input. The wrapper
removes the reserved field before calling the registered handler. The exact
namespace is private and must not reuse `_rollout`.

### Shared implementation components

Both execution adapters can share three internal concepts:

- an invocation store that atomically claims IDs and reads or publishes
  persistent records;
- a process-local registry that retains live tasks, cancellation signals, or
  process handles; and
- an adapter that starts and cancels the workload through the appropriate
  Runtime mechanism.

For managed session storage, an initial claim can use atomic directory or file
creation:

```text
<mount>/.agentcore-runtime/invocations/inv-123/
  started.json
  result.json
```

The store is authoritative across process replacement. The live registry is
only an optimization and a way to control work in the current process.

### App-handler `start`

The app wrapper can implement `start` in this order:

1. Read `runtimeSessionId` from the upstream request context.
2. Validate the private envelope and resolve the effective conversation ID.
3. Atomically claim the invocation ID in the selected store.
4. If the ID already exists, return its observable state without entering the
   user handler.
5. Register AgentCore async-task tracking for background work.
6. Create and retain a live task that calls the upstream `_invoke_handler`, so
   existing sync and async handler support remains unchanged.
7. For foreground delivery, await the task and return its terminal response.
8. For background delivery, return `in_progress` after the task and tracking
   state are installed.
9. Publish the terminal result before releasing async-task tracking.

Persisting terminal state for foreground and background invocations gives both
modes the same recovery semantics. A first implementation can choose not to
persist foreground results, but then it cannot promise result recovery after a
foreground connection is lost.

### App-handler `get`

The app wrapper intercepts `get` before the user handler and resolves status in
this order:

```text
terminal record exists       -> completed / failed / cancelled
live registry contains ID    -> in_progress
start record exists only     -> interrupted
no record exists             -> not_found
```

The client initially performs this operation through another
`InvokeAgentRuntime` call using the same Runtime session ID. This path must be
validated after the original execution environment has become idle. A future
direct session-storage API can replace it without changing the handle.

### App-handler `cancel`

For a live asynchronous handler, the adapter can retain its task, set a
cooperative cancellation signal, and request task cancellation. A synchronous
handler running in a Python worker thread cannot be safely force-stopped.

Cancellation should therefore be modeled as a request:

- return the existing state when the invocation is already terminal;
- return `cancellation_requested` while cooperative shutdown is pending; and
- publish `cancelled` only after execution actually stops.

The application may need access to a cancellation signal through invocation
context for handlers that perform long synchronous or external operations.
Cancellation never calls `StopRuntimeSession`.

### Sandbox command operations

Foreground `Sandbox.exec()` can continue to consume the native
`InvokeAgentRuntimeCommand` stream directly.

Detached execution requires a long-lived in-container process owner. The
initial ART implementation can extend the Sandbox helper so a short command
request asks that owner to:

- atomically claim the invocation ID;
- start the workload in a controllable process group;
- persist bounded stdout, stderr, exit status, and timeout state;
- return status without rerunning the command; and
- terminate the process group on cancellation.

Simply launching `nohup <command> &` through `InvokeAgentRuntimeCommand` is not
a sufficient design until live testing proves that descendants survive the
command stream and that the Runtime session remains active. The current
`agentcore-sandboxd` process can temporarily own detached children or expose a
local control channel during ART validation. Native Sandbox support should
ultimately replace this mechanism without requiring a workload-visible server
or reserved port.

### Duplicate and recovery behavior

For either adapter, a repeated `start` with the same invocation ID:

- returns `in_progress` when the execution is live;
- returns the existing terminal state when complete;
- returns `interrupted` when only the persistent start record remains; and
- never starts the workload implicitly a second time.

Different invocation IDs remain distinct executions even when their payloads
are identical or they share one Runtime session.
