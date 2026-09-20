# Sandbox SDK: sessions and recoverable command execution

| Field | Value |
| --- | --- |
| Status | Accepted |
| Implementation | In progress |
| Date | 2026-09-19 |

## Summary

The Sandbox SDK runs commands in arbitrary images deployed on AgentCore Runtime.
The Python client owns the user-facing session and command APIs;
`agentcore-sandboxd` owns command execution inside the container.

The current change adds recoverable command execution: foreground and background
commands share one process manager, persist their results, and can be addressed
after the initial client connection is lost. It does not add interactive shells,
live output subscriptions, or a complete environment-management API.

This document records Sandbox-specific API and implementation decisions.
[Runtime Invocation Protocol (RIP)](./runtime_invocation_protocol.md) defines the
shared execution contract: identities, `start/get`, states, retry semantics, and
persistence guarantees. Those semantics remain authoritative in RIP.

## Scope and boundaries

| Component | Responsibility |
| --- | --- |
| Sandbox SDK | Session lifecycle, command arguments and results, execution handles, waiting and reattachment |
| sandboxd | AgentCore HTTP contract, session hold, process ownership, execution deadlines, output capture and result publication |
| RIP | The shared invocation lifecycle and recovery contract used by command and app-handler adapters |
| AgentCore Runtime | Session routing and isolation, compute lifecycle, native invocation and terminal APIs |

RIP applies to individual command executions. Sandbox session management, terminal
interaction, files, and image adaptation have their own responsibilities.
[Dynamic Sandbox Environments](./sandbox_dynamic_environments.md) remains a
separate proposal for loading task filesystems; it is not part of this change.

## Architecture

```text
Python Sandbox SDK
  -> InvokeAgentRuntime
       -> sandboxd /invocations
            -> session actions: start / stop / status
            -> RIP operations: start / get
                 -> process manager
                      -> shell process group
                      -> local invocation records and output

AgentCore /ping
  -> busy while a session hold or active execution exists
```

Session actions and invocation operations use separate payload namespaces. A
session `start` holds the environment available between commands; a RIP `start`
creates or addresses one command execution within it.

The SDK generates an invocation ID before submission and sends the same ID on
network retries. The process manager records the invocation before starting the
command, owns it independently of the HTTP request, and publishes the terminal
result before reporting completion.

## Public API

| Operation | Behavior |
| --- | --- |
| `SandboxClient.start()` | Start/hold a sandbox session and return a `Sandbox` |
| `SandboxClient.attach(session_id)` | Construct a handle for an existing session without a network call |
| `Sandbox.exec(command, ...)` | Start a managed command and wait for its terminal result |
| `Sandbox.exec(command, background=True, ...)` | Start a managed command and return an `ExecHandle` |
| `Sandbox.get_exec(invocation_id)` | Reconstruct an execution handle without submitting work |
| `ExecHandle.status()` | Query the RIP invocation state |
| `ExecHandle.result(timeout=...)` | Poll for the terminal result |
| `Sandbox.terminate()` | Release the session hold and stop the Runtime session, best-effort |

```python
from agentcore_rl_toolkit.sandbox import SandboxClient

client = SandboxClient(runtime_arn="arn:aws:bedrock-agentcore:...:runtime/...")
with client.start() as sandbox:
    result = sandbox.exec("pytest -q", cwd="/app", timeout=900)

    handle = sandbox.exec("pytest -q", cwd="/app", timeout=900, background=True)
    # These two IDs can also be saved and passed to another client process.
    recovered = client.attach(handle.session_id).get_exec(handle.invocation_id)
    result = recovered.result(timeout=1200)
```

Exiting the context terminates the session, including unfinished commands.
Callers transferring a sandbox beyond that context use explicit `start()` and
`terminate()` to control its lifetime.

`exec()` accepts an optional `invocation_id` for callers that need to retain the
identity before making a request. Reusing it addresses the original execution;
different IDs permit intentionally identical commands.

### Results and errors

`ExecResult` contains `exit_code`, `stdout`, `stderr`, `timed_out`, and per-stream
truncation flags. Nonzero exit codes and execution timeouts are normal result
data, preserving the existing evaluation/training interface.

An ambiguous submission or foreground connection failure raises `ExecError`.
Its `.handle` retains the execution identity so the caller can query the original
execution; the original transport exception is chained as `__cause__`.
Retrieving an interrupted/missing execution or a persisted execution error also
raises `ExecError`. Transport failures during a handle's `get` request propagate;
the caller already holds the handle and can retry that read.

### Execution, waiting, and session lifetime

- `exec(timeout=...)` limits remote command execution: default 300 seconds, range
  1–3600. The daemon kills the command's process group on expiry.
- `handle.result(timeout=...)` limits local polling. A wait timeout leaves the
  command and session running. An in-flight AWS request remains subject to the
  client's socket timeout and retry configuration.
- Session termination affects every command in that environment and is separate
  from either completing a command or stopping a wait.

## Execution and transport decisions

### One manager for foreground and background

Both modes use the same command arguments, process ownership, invocation records,
and result format. Foreground waits on the initial request; background returns
after registration. Disconnecting a foreground request only stops that wait.

E2B's envd is a useful precedent for independent process ownership: both modes
start the same managed process, and its SDK decides whether to wait on the handle.
Its background handle may retain the initial output stream. Our background start
instead finishes the initial response and supports later independent `get`
requests. RIP also adds persisted identity, deduplication, and terminal results;
PID reconnection alone does not supply that contract. See
[RIP's process-model comparison](./runtime_invocation_protocol.md#appendix-b-related-process-and-connection-models).

### AgentCore API selection

The initial implementation selects `InvokeAgentRuntime` for both command `start`
and `get`. sandboxd already serves `/invocations`, so this path can carry
structured requests directly to its manager without an additional helper process.
Live ACR validation remains necessary before treating the transport as settled.

Previously, `exec()` used `InvokeAgentRuntimeCommand` directly. That API provides
native streaming stdout/stderr and an exit event, but does not expose the
persisted per-command retrieval and client-generated identity required by RIP.
Rebuild deployed images when upgrading from the old health-only sandboxd. The SDK
does not silently fall back to direct command execution.

Transport is an internal choice per capability. A future interactive-shell
surface should wrap AgentCore's native `InvokeAgentRuntimeCommandShell`, which
already provides persistent terminal state and reconnection. A shell ID identifies
the terminal, not every command typed into it. This boundary does not commit this
change to implementing a `shell()` API or routing terminal traffic through RIP.

## Process ownership, storage, and output

Commands run in a fresh shell, defaulting to `/bin/sh`, with the container's
environment and working directory. The SDK composes per-call `cwd` and `env`
settings into the shell command. The daemon owns the command's process group,
enforces its deadline, and cleans up remaining group members on exit. Processes
that escape the group are outside this cleanup mechanism; Runtime remains the
isolation boundary.

One daemon owns one local store root. It writes a start record before execution
and atomically publishes a terminal result after collecting output. The start
record excludes command text and environment values. Output still contains
whatever the workload prints.

The root is configurable with `--state-dir`. The initial local store survives
requests and daemon restarts, not compute replacement. After daemon loss,
start-only records with no registered owner are `interrupted`; the new daemon
does not adopt or rerun old processes. Store loss loses result and deduplication
history. Managed-storage stop/resume is not validated, and automatic retention
is outside this change.

### Capture output now; expose live output separately

Output collection continues whether or not a client is waiting. Each stdout/stderr
file retains its first **256 KiB**. Later bytes are drained and discarded, and the
result explicitly marks the affected stream as truncated. This bounds per-command
output storage and final response size.

The current SDK returns final output only. It does not expose callbacks,
incremental output retrieval, or stream replay. This is a delivery choice, separate
from process ownership and result persistence. A later output subscription can
attach to the same execution without making that connection own its lifecycle;
its buffering and replay contract will need a separate design decision.

## Source layout

Keep the daemon as the independent Go module at `sandboxd/`, alongside the Python
SDK at `src/agentcore_rl_toolkit/sandbox/`. The static binary can be added to an
arbitrary image without installing Python or this SDK inside it.

Within the SDK, `client.py` keeps `SandboxClient`, `Sandbox`, `ExecHandle`, and
`ExecError` together because session and command handles share request handling.
`types.py` contains the independent `ExecResult` and `SandboxProtocolError`
definitions and does not import the client.

Within the daemon, `main.go` handles HTTP/session dispatch, `process.go` owns
execution, and `store.go` owns invocation records. No repository-wide package
reorganization or shared cross-language framework is needed for this scope.
The [sandboxd README](../sandboxd/README.md) documents the wire envelope, file
layout, build commands, and local smoke tests.

## Implementation and validation

The working implementation covers session holds, managed foreground/background
commands, local records, process-group deadlines, bounded output, `ExecHandle`,
and recovery by session/invocation ID. This delivers the Sandbox portion of RIP;
it does not require the app-handler or Rollout migration to land first.

Go process/HTTP tests cover concurrent duplicate starts, distinct IDs, disconnect
survival, timeout cleanup, output limits, persistence failures, and recovery from
records. Python tests cover the SDK API and real botocore HTTP requests to a local
daemon, including foreground connection loss followed by result retrieval.

An environment-gated test in `tests/sandbox/test_live.py` uses
`SANDBOX_RUNTIME_ARN` against a rebuilt image. Local tests have passed; live ACR
validation has not been run for this change. Service routing and lifecycle
behavior therefore remain to be verified.

Real-time output, interactive shells, async APIs, public command cancellation,
file transfer, session TTL policy, managed-storage recovery, and S3 adapters
remain outside this implementation.
