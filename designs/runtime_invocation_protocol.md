# Runtime Invocation Protocol: an execution lifecycle for stateful, long-running workloads

| Field | Value |
| --- | --- |
| Status | Proposed |
| Implementation | In progress |
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

The protocol is not an RL-specific abstraction. Its execution contract is
shared by two adapters:

- the HTTP app-handler adapter, used by HTTP agent rollouts and other
  applications built on `BedrockAgentCoreApp`; and
- the Sandbox process adapter, which executes commands in isolated task
  environments with the same foreground, detached, polling, and recovery
  semantics.

The Rollout and Sandbox SDKs retain their workload-specific APIs.

## Scope and companion designs

This document defines the shared invocation contract without requiring one
transport, implementation language, or public SDK surface.

- [AgentCoreRuntimeApp](./agentcore_runtime_app.md) owns the HTTP app adapter:
  entrypoint compatibility, request envelopes, task ownership, and filesystem
  storage.
- [Sandbox SDK](./sandbox_sdk.md) owns the process adapter: session APIs,
  AgentCore transport, process management, and output policy.

### Relationship to A2A

`AgentCoreRuntimeApp` adds `start/get` operations to HTTP agents, allowing
clients to submit work and retrieve results later. Sandbox commands use the
same operations. A2A agents use their existing task APIs and do not need to
implement RIP.

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

## Workload requirements

| Consumer | Execution target | Required lifecycle |
| --- | --- | --- |
| Rollout SDK's HTTP path | An application handler in a Runtime session | Submit agent work, retrieve its result after disconnecting, and retain the session for follow-up invocations |
| Sandbox SDK | A process in an isolated Runtime session | Run foreground or detached commands, reconnect by execution identity, and recover status and output |

These surfaces may use different AgentCore APIs, but need the same answers to:

- What identifies one execution independently of its Runtime session?
- Does the initial connection wait for the result?
- How does a client retrieve a result after disconnecting?
- What happens when a submission is retried?
- How are status, failure, and interruption represented?
- When may the Runtime session stop?
- What persistence scope does the workload require?

RIP defines those semantics once. An app-handler adapter may carry RIP through
`InvokeAgentRuntime`. The Sandbox process adapter selects and validates its own
transport as described in the [Sandbox SDK design](./sandbox_sdk.md).

## Problems addressed

| Coupling or missing abstraction | Consequence | RIP direction |
| --- | --- | --- |
| Rollout execution is always detached. | Short or interactive calls pay persistence and polling costs unnecessarily. | Support foreground and background delivery as modes of the same execution. |
| A command is identified only by its initial stream. | Long commands cannot be safely detached and reattached. | Give detached commands their own invocation handle. |
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
- Replacing A2A's Task lifecycle or requiring native A2A servers to implement
  RIP.
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
configuration. The [HTTP app design](./agentcore_runtime_app.md#http-mapping)
defines that adapter's envelope.

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

A completed invocation contains either a result or an error.

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
- one terminal record containing the result or an error.

The start record identifies an accepted invocation. The terminal record stores
its result or error. Their exact schemas are implementation details.

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
storage configuration. Its records disappear when that compute is replaced.

AgentCore
[managed session storage](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-filesystem-configurations.html)
uses the same file layout at a configured mount path. It extends record
lifetime across compute stop/resume without requiring a customer-managed
bucket. It is currently a Preview feature on microVM runtimes: the documented
lifecycle expires after 14 idle days and resets on a Runtime version update.

Filesystem records are retrieved through a same-session data-plane adapter
regardless of whether the root is local or managed. Possible mappings include:

- an internal `InvokeAgentRuntime` operation intercepted before the user
  handler; or
- `InvokeAgentRuntimeCommand` running a deterministic helper that reads and
  emits the invocation record.

The execution adapter selects its transport. That transport must support
retrieval without entering the workload, including after compute resumes when
the selected store supports that lifetime.

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

The same applies to an S3 Files mount: it is shared storage unless access
controls provide isolation. Session-specific paths can avoid collisions but do
not create an access-control boundary.

Managed session storage replicates writes asynchronously and documents a flush
during graceful shutdown. Atomic local publication does not by itself promise
that the latest write survives an abrupt compute failure. In-session retrieval
also depends on the app being reachable, and may resume compute to read a result.

## Execution adapters

### App-handler adapter

The app-handler adapter carries RIP through `InvokeAgentRuntime`.
It dispatches `start/get` before the user handler, owns execution independently
of the request, and persists the terminal result before reporting completion.
Ordinary HTTP requests can continue to use the same entrypoint.

The [AgentCoreRuntimeApp design](./agentcore_runtime_app.md) defines the
envelope, handler dispatch, concurrency, storage layout, and recovery choices.
These choices do not constrain the process adapter or an upstream service
implementation.

### Sandbox process adapter

A process adapter needs an in-container owner that retains the command
independently of the initial client connection. Foreground and background are
delivery choices over the same managed execution. Both claim the invocation ID,
track liveness, and persist the terminal result before reporting completion.

The [Sandbox SDK design](./sandbox_sdk.md) specifies the concrete daemon, API
mapping, process lifecycle, and output policy. Those choices implement RIP
without becoming requirements for an app-handler adapter or an upstream service
implementation.

## Consumer 1: Rollout SDK

### Session adapters and responsibilities

The **Rollout SDK** configures, submits, groups, and collects agent rollouts for
training and evaluation. Its `RolloutSession` interface exposes `setup`, `run`,
and `shutdown` across server protocols. The AgentCore adapters are:

| Session adapter | Agent server | Execution lifecycle |
| --- | --- | --- |
| `agentcore_http` | `AgentCoreRuntimeApp` adapting `BedrockAgentCoreApp` | RIP `start/get`, invocation records, and recovery |
| `agentcore_a2a` | A native A2A server adapted as described in [PR #156](https://github.com/awslabs/agentcore-rl-toolkit/pull/156) | A2A Task lifecycle and the server's task storage |

Each session adapter translates its server's operations and results into the
rollout interface. The agent's existing server protocol determines which
adapter to use.

| Concern | Owner |
| --- | --- |
| Handler execution, identity, status, results, and retries | RIP app/client adapter for HTTP; A2A server/client integration for A2A |
| Result persistence and retrieval | The selected server and its session adapter |
| Runtime quota handling | Runtime client machinery and shared rollout session limits |
| Session setup, result adaptation, and cleanup policy | The workload-specific `RolloutSession` adapter |
| Model endpoint, model ID, sampling configuration, batching, and grouping | Rollout SDK |
| Trainer-facing timeout and failure adaptation | Rollout SDK and the relevant training backend |
| Trajectory capture and correlation | Rollout Gateway and training backends |
| Reward computation and conversation history | Application or training integration |

The layering is:

```text
Training or evaluation integration
  -> Rollout SDK
       -> RolloutSession
            -> agentcore_http -> RIP -> AgentCoreRuntimeApp
            -> agentcore_a2a  -> A2A -> native A2A server
               (both server paths run on AgentCore Runtime)

Training backend
  <-> Rollout Gateway for trajectory capture
```

### HTTP rollout flow

A background training rollout through `agentcore_http` uses RIP as follows:

1. The training backend creates any Rollout Gateway capture state it needs.
2. The Rollout SDK builds the agent payload, and the HTTP session adapter
   starts a RIP invocation with `background=true`.
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

### Benefits to HTTP rollout users

- Training keeps the current fire-and-retrieve background model.
- Evaluation and interactive use can call the same handler in foreground mode.
- A filesystem-backed store can remove mandatory S3 bucket and policy setup;
  managed storage additionally supports recovery across compute replacement.
- A single sticky Runtime session can support follow-up invocations without
  treating result retrieval as session completion.
- Retries can address one execution without guessing from payload or S3 key.
- `RolloutClient` can focus on rollout configuration, batching, grouping, and
  training-facing policy instead of implementing a private task manager.

### Migration

The HTTP migration has two corresponding replacements:

- `AgentCoreRuntimeApp` replaces `AgentCoreRLApp`, retaining the standard
  `@app.entrypoint` application interface.
- A RIP-backed `agentcore_http` rollout session replaces `agentcore_s3`,
  retrieving invocation state and results through Runtime HTTP calls.

The app and client contracts must migrate together. The S3-based `RolloutClient`
requires the existing `AgentCoreRLApp` result contract. Callers retain that pair
during the transition. The `agentcore_http` adapter speaks RIP; it is not
wire-compatible with the older rollout-specific setup/status/start/dump HTTP
adapter.

Session adapter names describe the interaction path. Storage is a separate
choice: using an S3 mount for `AgentCoreRuntimeApp.state_dir` still retrieves
results through `agentcore_http`. Direct external S3 retrieval can remain a
separate capability without making S3 mandatory for HTTP invocations.

## Consumer 2: Sandbox SDK

The Sandbox SDK is the workload-facing client for isolated task environments.
It uses RIP for individual command executions: a command has an invocation ID,
foreground or background delivery, and a retrievable terminal result. A client
can stop waiting and later address the same execution without rerunning it.

Sandbox session management, terminal interaction, files, and image adaptation
remain outside RIP. In particular, a persistent interactive terminal can contain
many commands; its shell ID does not replace their invocation identities.
Sandbox remains a general environment SDK usable by evaluation, RL, and
standalone workloads.

See the [Sandbox SDK design](./sandbox_sdk.md) for the public API, execution
flows, AgentCore API choices, and implementation scope.

## Ownership boundaries

| Layer | Owns | Does not own |
| --- | --- | --- |
| RIP | Invocation identity, foreground/background delivery, persisted lifecycle state and terminal result, retry semantics, waiting, storage and retrieval seams | Conversation history, rollout semantics, shell UX, trainer samples |
| Rollout SDK | Rollout payload and model configuration, batch submission, grouping, result policy, and session policy | Generic app task management, trajectory capture, conversation storage |
| `RolloutSession` adapters | Workload setup, run, shutdown, and result adaptation over HTTP/RIP or A2A | A universal server wire protocol, conversation storage |
| Sandbox SDK | Sandbox sessions, command/shell/file UX, structured process results, Sandbox-specific handles | Agent payloads, rewards, trajectory capture |
| Rollout Gateway | Token-level trajectory capture and trace construction | Runtime invocation lifecycle and result delivery |
| Training backends | Capture-session orchestration, reward and trace joining, trainer-native sample construction | Generic Runtime execution protocol |
| AgentCore Runtime | Session routing, isolation, compute lifecycle, Runtime invocation APIs, and local or managed filesystem substrate | ART workload semantics |

This boundary intentionally leaves conversation implementation to the
application or agent framework. RIP accepts an optional conversation ID so it
does not prevent multi-turn or multi-conversation designs, but it does not
manage messages or memory.

## Public client surface

The HTTP adapter exposes `AgentCoreHttpClient` and `InvocationHandle`; their
API and plain HTTP compatibility are defined in the
[app design](./agentcore_runtime_app.md#http-client). Sandbox retains
`ExecHandle`. Consumers need not expose a shared `Retriever` abstraction.

## Open questions

- How should each consumer select and expose its persistence scope?
- Which result and artifact size limits belong in the protocol versus each
  workload adapter?
- Which generic classes should be public in ART before the capability has an
  upstream home?

## Appendix A: Client integration

HTTP rollout and Sandbox clients map their public APIs onto RIP without
exposing the adapter's storage layout or requiring a new public method for
every protocol operation.

### Quota considerations

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

## Appendix B: Related process and connection models

This appendix records implementation precedents that inform the Sandbox
adapter. They support the separation between delivery mode and execution
lifecycle, but none is a protocol dependency.

### Comparison

| Model | Execution identity | Disconnect behavior | Later retrieval | Retry behavior |
| --- | --- | --- | --- | --- |
| `InvokeAgentRuntimeCommand` | Initial command request and stream | Provides structured one-shot output while the stream is available | No separate per-command result-retrieval operation | Retrying is a new command |
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
