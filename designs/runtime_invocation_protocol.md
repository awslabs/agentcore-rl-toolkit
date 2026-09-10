# Runtime Invocation Protocol: an execution lifecycle for stateful, long-running workloads

| Field | Value |
| --- | --- |
| Status | Proposed |
| Implementation | Not started |
| Date | 2026-09-09 |
| Pull request | [#120](https://github.com/awslabs/agentcore-rl-toolkit/pull/120) |

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
semantics, and result-retrieval path. Foreground and background are two ways
to interact with that same lifecycle:

- `background=false` starts the invocation and waits on the initial
  connection; and
- `background=true` starts the invocation and returns a handle before it
  finishes.

Invocation records are written outside process-local memory in both modes. A
microVM-local filesystem is the baseline store. Placing the same file layout on
a managed session-storage mount extends its lifetime across compute
stop/resume, while S3 remains an alternative for external retrieval.

The protocol is not an RL-specific abstraction. It can support both:

- the Rollout SDK, which submits long-running agent rollouts for training and
  evaluation; and
- the Sandbox SDK, which executes commands in isolated task environments and
  needs the same foreground, detached, polling, and recovery semantics.

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
- normal network retries must not implicitly create a second execution.

The missing abstraction is therefore:

> An invocation with a lifecycle independent of both its client connection and
> its Runtime session.

Foreground and background delivery follow from that abstraction. They are not
two execution lifecycles. Foreground means start and wait; background means
start and return the invocation handle.

Keeping a task running after the client disconnects is not enough. Invocation
records must also live outside request-local memory so a later `get` can recover
status and results. The selected store determines whether that recovery is
limited to the current compute or extends across compute replacement.

## Reference pattern: OpenAI Responses

The
[OpenAI Responses API background mode](https://developers.openai.com/api/docs/guides/background/)
provides an existing example of this separation:

- foreground and background requests create the same kind of logical
  execution;
- a background request returns an addressable response before processing is
  complete;
- the client can retrieve its status and output later;
- connection lifetime is separate from execution lifetime.

In this pattern, `background` controls result delivery and whether the initial
connection waits. It does not select a different user handler or a different
application result.

The Responses API also separates logical history from execution identity:

- a Conversation is an optional history container; and
- each Response is a distinct execution with its own ID and lifecycle.

This proposal borrows that separation:

- `conversation_id` optionally tells the application which conversation or
  thread an invocation belongs to; and
- `invocation_id` identifies one execution.

The protocol passes the conversation ID through but does not store or interpret
the conversation's messages or history.

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
plane. Its initial implementation persists invocation state in the selected
store and retrieves it through an available AgentCore Runtime invocation API.

The persisted invocation record is the closest equivalent to a Response
resource: it has an ID, lifecycle status, and terminal output. It is not a new
conversation database or a claim that ART already has an independently
queryable control-plane resource.

## Intended destination and the name

ART is a suitable place to incubate this protocol because the Rollout and
Sandbox SDKs both need it now. However, invocation identity,
foreground/background delivery, persisted status, and result retrieval are
general AgentCore Runtime capabilities rather than training- or
evaluation-specific concepts.

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
- **Invocation** includes one logical application-handler or process
  execution, independently of the AgentCore transport used to reach it.
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
| Sandbox SDK | A process in an isolated Runtime session | Reaches one command through `InvokeAgentRuntimeCommand`, consumes its stream synchronously, and returns `ExecResult` | Detached execution, invocation handles, polling, reconnection, and recovery |

These surfaces use different AgentCore APIs, but need the same answers to:

- What identifies one execution independently of its Runtime session?
- Does the initial connection wait for the result?
- How does a client retrieve a result after disconnecting?
- What happens when a submission is retried?
- How are status, failure, and interruption represented?
- When may the Runtime session stop?
- What persistence scope does the workload require?

RIP defines those semantics once. An app-handler adapter may carry RIP through
`InvokeAgentRuntime`. For Sandbox, ART still needs to validate whether the
client should reach the process manager through `InvokeAgentRuntime` or through
`InvokeAgentRuntimeCommand` calling an in-container helper.

## Problems addressed

| Current behavior or gap | Consequence | RIP direction |
| --- | --- | --- |
| Rollout execution is always detached. | Short or interactive calls pay persistence and polling costs unnecessarily. | Support foreground and background delivery as modes of the same execution. |
| Sandbox commands are always attached to their initial stream. | Long commands cannot be safely detached and reattached. | Give detached commands their own invocation handle. |
| `runtimeSessionId` also acts as the rollout execution identity. | Multiple calls and follow-ups in one sticky session cannot be addressed independently. | Add one invocation ID per logical execution. |
| S3 storage, polling, and background execution are combined in `rollout_entrypoint`. | Connection mode cannot evolve independently from storage or retrieval. | Separate execution mode, persistent state, storage backend, and retrieval path. |
| Rollout results require a customer-managed bucket, while other workloads may need only compute-scoped state or session-scoped recovery. | One mandatory persistence scope either adds setup or weakens recovery. | Use the same filesystem store on local or managed roots and retain S3 for external retrieval. |
| There is no explicit `start` versus `get` operation. | A polling request could accidentally execute the workload again after process-local state is lost. | Make retrieval an operation that can never enter the user workload. |
| Payload equality is easily confused with duplicate execution. | Two intentionally identical requests may be rejected, or retries may run twice. | Deduplicate by invocation identity, never by payload equality alone. |
| Invocation completion currently implies rollout session cleanup. | Automatic `StopRuntimeSession` prevents sticky-session follow-up. | Separate invocation completion, wait timeout, and session termination. |
| Runtime-specific mechanics are named `AgentCoreRLApp` and `rollout_entrypoint`. | A general capability appears specific to RL. | Move generic behavior into a Runtime-level app and protocol layer. |
| Rollout and Sandbox clients would otherwise implement detached lifecycle independently. | Identity and recovery semantics can drift. | Reuse the RIP contract through workload-specific adapters. |

## Goals

- Define one invocation lifecycle, independent of connection and Runtime
  session lifetime, for application handlers and processes.
- Treat foreground and background as delivery modes over the same execution,
  independently from local sync or async client APIs.
- Give each invocation explicit identity, retry, status, and terminal-result
  semantics.
- Persist lifecycle state through a filesystem-backed store with an explicit
  compute- or session-scoped lifetime, while retaining replaceable storage and
  retrieval adapters.
- Keep Runtime session termination separate from invocation completion and
  waiting.
- Let Rollout and Sandbox retain workload-specific APIs over a generic
  capability suitable for eventual upstream adoption.

## Non-goals

- Reproducing the complete OpenAI Responses control plane or managing
  conversation history.
- Defining rollout trajectory capture, reward computation, or trainer sample
  construction.
- Automatically rerunning interrupted workloads with side effects, or
  protecting protocol state from arbitrary code inside the same Runtime
  session.
- Guaranteeing recovery or idempotency after the selected store is lost.
- Defining generic invocation cancellation in the initial protocol.
- Replaying foreground stream contents after a disconnect in the initial
  implementation.

## Identity model

RIP distinguishes three identities:

- **Runtime session ID**: the AgentCore compute, sticky-routing, isolation,
  storage, resource, and lifecycle scope.
- **Conversation ID**: optionally tells the application which conversation or
  thread an invocation belongs to. RIP passes the ID through but does not store
  or interpret its messages or history.
- **Invocation ID**: one execution of an application handler or command.

By default, the Runtime session serves as the implicit conversation namespace.
An explicit `conversation_id` is only needed to distinguish multiple
conversations within one Runtime session:

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

The client generates the invocation ID before its first network request and
reuses that ID for retries. Two identical payloads with different invocation
IDs are two valid executions. Once an invocation ID exists, a repeated `start`
addresses that existing execution; any newly supplied application payload is
not evaluated or executed.

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

An asynchronous client can expose the same method names:

```python
# The caller's event loop remains non-blocking, while the remote connection
# waits for completion.
result = await async_client.invoke(request, background=False)

# Submission and later waiting are both non-blocking to the caller's loop.
handle = await async_client.invoke(request, background=True)
result = await handle.result(timeout=600)
```

Each workload-facing SDK can choose domain-appropriate names. For example,
Rollout may retain `invoke()` and `RolloutFuture`, while Sandbox may use
`exec()` for both modes and return either a result or a handle. Those API names
do not change the RIP lifecycle.

### Logical operations

RIP needs two logical invocation operations:

- `start(invocation_id)`: record and begin an execution if that ID does not
  already exist.
- `get(invocation_id)`: return current status or terminal output without
  entering the user workload.

Stopping a Runtime session is a separate lifecycle operation outside RIP.

App-handler requests carry these operations in a reserved payload field
separate from application data and from `_rollout`, which is rollout-specific
configuration. Appendix A shows an illustrative envelope.

### State machine

Every invocation exposes at least:

```text
absent
  |
  | start
  v
in_progress ------> completed
      |
      +-----------> interrupted
```

A completed invocation contains either a result or a structured error.

`interrupted` means a persisted start record shows that execution began, but no
live execution and no terminal result can be recovered. RIP must not
automatically rerun an interrupted workload because app handlers and commands
may have side effects. If the selected store itself is lost, the invocation
instead becomes `not_found`.

Adapters may expose additional transient detail, but consumers must be able to
reason using this minimum state model.

### Start and retry behavior

The execution adapter must record an invocation ID before beginning the user
workload. While that record exists, a repeated `start` for the same ID:

- returns the stored result or error when completed;
- returns `in_progress` when the execution is still live;
- returns `interrupted` when only a stale start record remains; and
- never implicitly starts the workload a second time.

The initial client generates the invocation ID before the network request.
Therefore, an ambiguous connection failure does not erase the intended
execution identity. Internal retries reuse the same ID.

Most users should not need to provide invocation IDs manually. Advanced callers
may supply or persist one when they need to transfer a handle between
processes, retry after losing local state, or reattach from another client.

Idempotency is scoped to the selected store's lifetime. If a microVM-local
store disappears with its compute, the protocol can no longer distinguish that
invocation from one that never started, and the same ID may execute again.
Callers that require the identity to survive compute replacement must place the
records on managed session storage, S3, or another store with that lifetime.

### Waiting and session lifecycle

RIP keeps two actions distinct:

- **Stop waiting**: a local timeout returns control to the caller. The remote
  execution continues by default.
- **Stop the Runtime session**: `StopRuntimeSession` terminates the execution
  environment and may affect every invocation or conversation in that
  session.

The initial protocol does not support invocation cancellation. A generic
handler may own asynchronous tasks, worker threads, subprocesses, or external
work that cannot be reliably stopped through one contract. Applications may
expose workload-specific cleanup outside RIP.

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

RIP v1 does not define an incremental streaming protocol. An adapter may
preserve an existing SSE or EventStream response while the connection remains
open, but recovery covers only lifecycle status and the terminal result. RIP
does not replay output after a disconnect.

## Persistence and retrieval

### Filesystem-backed invocation state

Process-local tasks, process IDs, and caches remain useful for tracking live
work, but invocation identity and terminal results are written to a store that
other requests or processes can read. The initial implementation can use one
filesystem store configured with a root path.

Local files do not by themselves provide session durability. Their value is
cross-request or cross-process visibility and an identical upgrade path to a
managed mount.

The store contains:

- a start record persisted before the user workload begins; and
- one terminal record containing the result or a structured error.

The start record identifies an accepted invocation. The terminal record stores
its result or structured error. Their exact schemas are implementation
details.

For an app handler, a replayable response can contain the status code, content
type, response body, or an artifact reference. It stores the normalized
response that the client observes, not the original Python object. A Rollout
result can remain a JSON response, while a custom application response may use
another content type. A command result can instead contain exit status, stdout,
stderr, and artifact references.

The complete input payload should not be persisted by default. Rollout payloads
may contain model credentials, and Sandbox commands may contain sensitive
arguments or environment values.

### Persistence scope and retrieval path

The same filesystem implementation can use either a microVM-local root or a
managed session-storage mount:

| Store location | Persistence scope | Retrieval path |
| --- | --- | --- |
| MicroVM-local filesystem | Current compute | Same-session `get` request |
| Managed session-storage mount | Runtime session, within the managed-storage lifecycle | Same-session `get` request |
| S3 | Configured bucket and retention policy | Direct S3 read or `get` request |

A microVM-local root is the baseline implementation: it requires no additional
storage configuration and can be exercised in local Docker tests. Its records
disappear when that compute is replaced.

AgentCore
[managed session storage](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-filesystem-configurations.html)
uses the same file layout at a configured mount path. It extends record
lifetime across compute stop/resume without requiring a customer-managed
bucket. It is currently a Preview feature on microVM runtimes: the documented
lifecycle expires after 14 idle days and resets on a Runtime version update.

Filesystem records are retrieved through a same-session data-plane adapter
regardless of whether the root is local or managed. Candidate implementations
include:

- an internal `InvokeAgentRuntime` operation intercepted before the user
  handler; or
- `InvokeAgentRuntimeCommand` running a deterministic helper that reads and
  emits the invocation record.

This proposal intentionally does not select between them without live
validation. Managed storage additionally requires validating `get` after the
original compute becomes idle and a new compute resumes the session.

S3 remains an alternative backend for workloads that need direct external
reads, larger artifacts, or independent retention.

RIP does not prescribe one universal default for every consumer. The public
handle need not expose a physical path, but the SDK must document whether its
records are compute-scoped, session-scoped, or externally retained.

### Recovery

`get` reads the selected store without entering the workload. A terminal
record returns `completed`; a surviving start-only record with no live
execution returns `interrupted`; no record returns `not_found`.

Managed storage preserves this distinction across compute replacement. A
local store does not, including when a hosted workload kills the app server
and Runtime replaces its compute.

Terminal state must be persisted to the selected store before the adapter
reports success. If persistence ultimately fails, the adapter reports failure
rather than success.

### Security boundary

Both local and managed filesystems avoid using one shared result bucket:

- invocation state is scoped to the current compute or Runtime session;
- applications do not need broad access to one shared bucket; and
- users do not need to construct per-session S3 prefix policies.

Managed storage adds recovery across compute replacement; it does not add
protection from code inside the session. A user handler or shell command with
arbitrary filesystem access in the same environment may read or alter either
filesystem store.

An S3 backend instead relies on its configured IAM policy, bucket, and key
layout. Selecting S3 does not give those objects AgentCore session isolation.

## Execution adapters

### App-handler adapter

The app-handler adapter carries RIP through `InvokeAgentRuntime`. A wrapper
around the registered `BedrockAgentCoreApp` entrypoint intercepts protocol
operations before they reach the user handler:

```text
InvokeAgentRuntime
  -> RIP envelope
  -> start / get dispatch
  -> registered application handler for start only
```

Both modes retain the same handler task, persist its terminal state, and use
AgentCore async-task tracking to keep the session busy until terminal
publication completes. Foreground execution waits for that task on the initial
connection; background execution returns `in_progress` after the task is
registered. Both modes continue to use the upstream sync/async handler
dispatch. Appendix A describes the initial in-process registry, persistence
ordering, and recovery behavior.

### Sandbox process adapter

The Sandbox process adapter needs an in-container execution manager that owns
the child process independently from the initial client connection:

```text
InvokeAgentRuntime or InvokeAgentRuntimeCommand
  -> RIP-aware Sandbox process manager
       -> child process
       -> persisted status and output
       -> foreground stream and wait
          or background handle
```

Foreground and background are client delivery choices over the same managed
process. Foreground waits on the initial connection where practical;
background returns a handle. In both cases the manager claims the invocation
ID, tracks the process, and persists terminal output.

An initial ART prototype could extend `agentcore-sandboxd` with the process
manager and call it through `InvokeAgentRuntime`. Another option is for
`InvokeAgentRuntimeCommand` to call a local helper. Live validation must decide
between them. Appendix A describes the illustrative prototype shape.

## Consumer 1: Rollout SDK

### Definition and scope change

After RIP is separated, the **Rollout SDK** is the ART training- and
evaluation-facing adapter that configures, submits, groups, and collects agent
rollouts over RIP.

Its scope changes as follows:

| Concern | Current owner | Proposed owner |
| --- | --- | --- |
| Runtime app wrapper and entrypoint dispatch | Rollout SDK (`AgentCoreRLApp` + `@app.rollout_entrypoint`) | RIP app adapter (`AgentCoreRuntimeApp` + `@app.entrypoint`) |
| Foreground/background execution | Background-only Rollout SDK behavior | RIP |
| Invocation identity, status, result, and retry | Implicit across `runtimeSessionId`, S3 key, and `RolloutFuture` | RIP |
| Invocation persistence and retrieval | Rollout SDK S3 implementation | RIP storage and retrieval adapters |
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

### Concrete rollout flow

A background training rollout can use RIP as follows:

1. The training backend creates any Rollout Gateway capture state it needs.
2. The Rollout SDK builds the agent payload and starts a RIP invocation with
   `background=true`.
3. The agent runs while the training integration retains the invocation
   handle.
4. The integration awaits the terminal result and independently finishes any
   trajectory-capture state.
5. The training backend combines the result, reward, and captured records
   according to its own sample contract.

RIP does not require its invocation ID to equal a Runtime session ID, gateway
session ID, rollout ID, training input ID, or conversation ID. An integration
may correlate them explicitly, but the protocol does not collapse their
meanings.

### Benefits to rollout users

- Training keeps the current fire-and-retrieve background model.
- Evaluation and interactive use can call the same handler in foreground mode.
- A filesystem-backed store can remove mandatory S3 bucket and policy setup;
  managed storage additionally supports recovery across compute replacement.
- A single sticky Runtime session can support follow-up invocations without
  treating result retrieval as session completion.
- Retries can address one execution without guessing from payload or S3 key.
- `RolloutClient` can focus on rollout configuration, batching, grouping, and
  training-facing policy instead of implementing a private task manager.

## Consumer 2: Sandbox SDK

### Definition and boundary

The **Sandbox SDK** is the workload-facing client for starting or attaching to
an isolated task environment and executing commands, shells, files, and
processes within it.

RIP applies specifically to command execution lifecycle. It does not absorb:

- Sandbox provisioning and session APIs;
- shell and file UX; or
- task, verifier, and image-adaptation contracts.

The native AgentCore interactive shell remains a separate Sandbox surface for
persistent terminal sessions. RIP applies when one command needs its own
identity, foreground or detached delivery, structured status, persisted result,
and retry semantics.

### Concrete Sandbox flows

The following API shapes are illustrative. They demonstrate user-visible
semantics without fixing method names, transport selection, or whether the
initial implementation extends `agentcore-sandboxd`.

Foreground execution can remain simple:

```python
result = sandbox.exec("pytest -q", timeout=900, background=False)
```

The process adapter starts one managed execution and the client waits for it on
the initial connection. RIP still claims a distinct invocation identity,
tracks the command as live, and persists its terminal status and output. If the
connection drops, the client can recover through the same invocation handle
rather than rerun the command.

Background execution uses the same method:

```python
handle = sandbox.exec("pytest -q", timeout=900, background=True)

status = handle.status()
result = handle.result(timeout=1200)
```

The returned handle identifies the command independently from its Runtime
session. It can expose status, result, and reattachment independently from
Sandbox termination.

### Benefits to Sandbox users

- Long builds, tests, and coding-agent harnesses do not depend on one command
  stream remaining connected.
- A detached command has a transferable handle for the lifetime of its
  selected store.
- Polling cannot accidentally rerun the command.
- Timeout can stop local waiting without destroying the environment.
- Rollout orchestration can use the same execution state model whether the
  workload is an agent handler or an in-sandbox harness process.

Sandbox does not become an RL SDK by using RIP. It remains a general task-
environment abstraction that can be consumed by evaluation, RL, or standalone
workloads.

## Ownership boundaries

| Layer | Owns | Does not own |
| --- | --- | --- |
| RIP | Invocation identity, foreground/background delivery, persisted lifecycle state and terminal result, retry semantics, waiting, storage and retrieval seams | Conversation history, rollout semantics, shell UX, trainer samples |
| Rollout SDK | Rollout payload and model configuration, batch submission, grouping, result policy, and session policy | Generic app task management, trajectory capture, conversation storage |
| Sandbox SDK | Sandbox sessions, command/shell/file UX, structured process results, Sandbox-specific handles | Agent payloads, rewards, trajectory capture |
| Rollout Gateway | Token-level trajectory capture and trace construction | Runtime invocation lifecycle and result delivery |
| Training backends | Capture-session orchestration, reward and trace joining, trainer-native sample construction | Generic Runtime execution protocol |
| AgentCore Runtime | Session routing, isolation, compute lifecycle, Runtime invocation APIs, and local or managed filesystem substrate | ART workload semantics |

This boundary intentionally leaves conversation implementation to the
application or agent framework. RIP accepts an optional conversation ID so it
does not prevent multi-turn or multi-conversation designs, but it does not
manage messages or memory.

## Public surface and migration

The app-side generic class can initially be `AgentCoreRuntimeApp`, retaining
the upstream `@app.entrypoint` name. `AgentCoreRLApp` and
`rollout_entrypoint` can remain temporary migration aliases while examples
move to the generic app surface.

The generic client and handle names remain open. `RolloutFuture` and a future
Sandbox execution handle can wrap the same internal RIP handle without
exposing a new public `Retriever` abstraction.

## Implementation phases

### Phase 1: specify and validate the protocol

- Define versioned `start` and `get` envelopes.
- Define invocation identity, repeated-start behavior, and the minimum state
  model.
- Implement the filesystem store against a configurable local root.
- Define compute- and session-scoped persistence guarantees.
- Add local Docker coverage for start, get, retry, completion, and
  interruption.
- Validate both candidate same-session retrieval transports.
- Test two identical requests with distinct invocation IDs and repeated
  submission of one invocation ID without re-entering the workload.

### Phase 2: app adapter and Rollout migration

- Introduce the generic app adapter and `AgentCoreRuntimeApp`.
- Add foreground/background modes to the client machinery.
- Allow the filesystem store root to use a managed session-storage mount.
- Validate stop/resume recovery with managed storage.
- Preserve the current S3 backend and direct retrieval path for Rollout.
- Separate wait timeout from session cleanup.
- Refactor `RolloutClient` and `RolloutFuture` into rollout-facing wrappers
  over RIP.
- Keep rollout payload fields and trainer-facing batching outside the generic
  layer.

### Phase 3: Sandbox process adapter and detached execution

- Define one Sandbox process manager and process-group lifecycle for
  foreground and detached commands.
- Preserve foreground streaming while persisting command status, exit code,
  and output for recovery.
- Add Sandbox-specific detached handles, polling, and reattachment.
- Prototype invocation-path dispatch through `agentcore-sandboxd`, while
  treating the exact transport and daemon shape as replaceable.
- Keep native Interactive Shell as a separate Sandbox surface.

### Phase 4: hardening and upstreaming

- Separate data-plane and new-session rate limiting.
- Add live integration coverage for local-store loss, managed-store
  reactivation, persistence failure, and storage retention.
- Define artifact policies for results that exceed the manifest format.
- Evaluate moving the generic app, client, and protocol behavior into
  `bedrock-agentcore-sdk-python`.
- Keep Rollout, Sandbox, Rollout Gateway, and trainer-specific integrations in
  ART.

## Open questions

- Should the Sandbox client use `InvokeAgentRuntime` or
  `InvokeAgentRuntimeCommand` to reach the process manager?
- Which existing Runtime API should serve `get` for filesystem records?
- How should each consumer select and expose its persistence scope?
- Which result and artifact size limits belong in the protocol versus each
  workload adapter?
- Which generic classes should be public in ART before the capability has an
  upstream home?

## Appendix A: Initial ART implementation sketch

This appendix is illustrative and is not part of the protocol contract. It
describes one way ART can validate RIP with the current AgentCore Runtime and
SDK surfaces. The protocol does not require these private envelope names, file
paths, or helper processes, and they should be replaceable by upstream
support.

### Current quota considerations

Service quotas are implementation inputs, not RIP semantics, and must be
rechecked as AgentCore evolves. The current
[Runtime quota table](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/bedrock-agentcore-limits.html#runtime-service-limits)
lists 1,000 TPS shared across data-plane APIs and 25 TPS for new Runtime
session creation. Client limiters should model them separately and still use
retry and backoff because the quotas are shared across callers.

### Client mapping

The logical operations do not need to become new public methods:

| Workload-facing operation | RIP operation |
| --- | --- |
| `client.invoke(...)` | `start` |
| `handle.status()`, `done()`, or `result()` | `get` |
| `StopRuntimeSession` or `sandbox.terminate()` | Session lifecycle outside RIP |

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

`get` uses the same envelope without application input. The wrapper removes
the reserved field before calling the registered handler. The exact namespace
is private and must not reuse `_rollout`.

### Shared implementation components

Both execution adapters can share three internal concepts:

- a filesystem invocation store configured with a root path;
- a process-local registry that retains live tasks or process handles; and
- an adapter that starts the workload through the appropriate Runtime
  mechanism.

The initial single-owner implementation can serialize `start` handling for
each invocation ID:

```python
# Illustrative ordering only.
with start_lock(invocation_id):
    if state := store.get(invocation_id):
        return state
    store.write_started(invocation_id)
    registry[invocation_id] = create_execution()
```

The lock prevents an overlapping request or retry from observing the ID as
absent and starting the same workload again. The layout can remain an
implementation detail:

```text
<invocation-store-root>/.agentcore-runtime/invocations/inv-123/
  started.json
  result.json
```

The store is authoritative for invocation identity and terminal result for as
long as its files remain available. The live registry controls work in the
current process and establishes whether a start record still has a live
execution owner.

### App-handler `start`

The app wrapper can implement `start` in this order:

1. Read the request context and validate the private envelope.
2. Acquire the process-local start lock for the invocation ID.
3. Read the selected store. If the ID already exists, return its observable
   state without entering the user handler.
4. Persist the start record.
5. Create and retain the handler task, install it in the live registry, and
   register AgentCore async-task tracking before releasing the lock.
6. Await the task for foreground delivery or return `in_progress` for
   background delivery.
7. Publish the terminal result before removing live tracking state.

Persisting terminal state for foreground and background invocations gives both
modes the same recovery semantics. The execution task must be owned
independently from the foreground request wait so losing that connection does
not discard invocation tracking or terminal publication.

### App-handler `get`

For the initial single-owner app adapter, the wrapper intercepts `get` before
the user handler and resolves status in this order:

```text
terminal record exists       -> completed; return its result or error
live registry contains ID    -> in_progress
start record exists only     -> interrupted
no record exists             -> not_found
```

Writing the start record and installing the live registry entry must be
serialized against `get`, so a concurrent read cannot mistake the brief
registration window for interruption. If a server process restarts while the
filesystem remains, the registry is empty and a remaining start-only record
can be reported as `interrupted`. If a local filesystem disappears with its
compute, the record is absent and the invocation becomes `not_found`; a managed
mount preserves the record across compute replacement.

The client initially performs this operation through another
`InvokeAgentRuntime` call using the same Runtime session ID. The same request
works for local and managed filesystem roots. Managed storage additionally
requires validating the path after the original compute becomes idle and the
session resumes.

### Sandbox command operations

Foreground and detached execution both require an in-container process owner
for persisted lifecycle state. One illustrative ART implementation extends
`agentcore-sandboxd` so its invocation path dispatches `start` and `get` to a
process manager:

```text
InvokeAgentRuntime
  -> agentcore-sandboxd
       -> start / get
       -> invocation store
       -> live process registry
       -> child process group
```

This diagram is not a fixed wire contract.
`InvokeAgentRuntimeCommand` calling a local helper is the other implementation
candidate. Regardless of which API reaches the manager, the owner needs to:

- serialize starts for the same invocation ID and persist its start record;
- start and track the workload process;
- persist stdout, stderr, exit status, and timeout state; and
- return status without rerunning the command.

Simply launching `nohup <command> &` through `InvokeAgentRuntimeCommand` is not
a sufficient design until live testing proves that descendants survive the
command stream and that the Runtime session remains active. The current
`agentcore-sandboxd` process can temporarily own detached children, but that is
an implementation experiment rather than a commitment to make the health shim
the permanent protocol boundary.

The native AgentCore Interactive Shell remains a direct Sandbox feature for
PTY-oriented workflows. It does not provide the per-command invocation records
defined by RIP.

## Appendix B: Related process and connection models

This appendix records implementation precedents that inform the Sandbox
adapter. They support the separation between delivery mode and execution
lifecycle, but none is a protocol dependency.

### Comparison

| Model | Execution identity | Disconnect behavior | Later retrieval | Retry behavior |
| --- | --- | --- | --- | --- |
| `InvokeAgentRuntimeCommand` | Initial command request and stream | Provides structured one-shot output while the stream is available | No separate command handle is exposed to the current Sandbox SDK | Retrying is a new command |
| AgentCore Interactive Shell | Runtime session ID plus shell ID | The named PTY can continue and reconnect | Replays terminal output, but does not define a persisted result for each command entered in the shell | Reusing a shell ID reconnects the terminal; it does not identify a command retry |
| E2B `envd` | Process ID or tag | The process is owned independently from the request stream | A client can reconnect to a live process; the inspected implementation retains terminal status briefly but does not replay missed output | Starting again creates another process |
| RIP Sandbox process adapter | Invocation ID | The managed process continues independently from the initial wait | Persisted status and terminal result are retrieved by invocation ID while the selected store survives | Reusing an invocation ID addresses the existing execution while its record survives |

The AgentCore
[interactive-shell documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-get-started-command-shell.html)
shows that persistent terminal identity and reconnect are native Runtime
capabilities. The shell is therefore the right substrate for interactive
terminal UX, while RIP supplies the missing per-command invocation lifecycle.

The
[E2B SDK](https://github.com/e2b-dev/E2B)
follows the same delivery-mode pattern proposed here:
[`run`](https://github.com/e2b-dev/E2B/blob/main/packages/js-sdk/src/sandbox/commands/index.ts)
always starts a process handle, then either returns that handle for background
execution or waits on it for foreground execution. Its
[`CommandHandle`](https://github.com/e2b-dev/E2B/blob/main/packages/js-sdk/src/sandbox/commands/commandHandle.ts)
can disconnect without killing the process and reconnect by process ID.

The corresponding `envd` implementation lives in the
[E2B infrastructure repository](https://github.com/e2b-dev/infra). Its
[`Start`](https://github.com/e2b-dev/infra/blob/main/packages/envd/internal/services/process/start.go)
implementation deliberately owns the process independently from the request
lifetime. Its current
[process service](https://github.com/e2b-dev/infra/blob/main/packages/envd/internal/services/process/service.go)
keeps live process state in memory and briefly retains terminal status, while
missed output is not replayed. This makes it a useful precedent for process
continuity, but not a substitute for RIP's persisted invocation record,
idempotent claim, and terminal-result retrieval within the selected store's
lifetime.

The resulting distinction is:

```text
E2B envd and AgentCore Interactive Shell
  -> execution can outlive a connection

RIP with a local filesystem
  -> invocation identity and terminal result outlive the initial connection
     and process-local memory while the compute survives

RIP with managed session storage
  -> the same records additionally survive compute replacement
```
