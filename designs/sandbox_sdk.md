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

The SDK supports recoverable command execution: foreground and background
commands share one process manager, persist their results, and can be addressed
after the initial client connection is lost.

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
separate proposal for loading task filesystems.

## Architecture

sandboxd builds independently as a static Go binary that can be added to any
image with a shell; the Python SDK runs on the client and is not required in
the sandbox image.

```text
Python Sandbox SDK
  -> InvokeAgentRuntime
       -> sandboxd /invocations
            -> session actions: start / stop / status
            -> RIP operations: start / get
                 -> process manager
                      -> shell process
                      -> local invocation records and output

AgentCore /ping
  -> busy while a session hold or active execution exists
```

Session actions and invocation operations use separate payload namespaces. A
session `start` holds the environment available between commands; a RIP `start`
creates or addresses one command execution within it.

The SDK generates an invocation ID before submission and sends the same ID on
network retries.

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
    existing_handle = client.attach(handle.session_id).get_exec(handle.invocation_id)
    result = existing_handle.result(timeout=1200)
```

`existing_handle` is a new local `ExecHandle` referring to the same execution.
Constructing it makes no network request and does not verify that the execution
exists. Calling `.result()` queries the daemon and waits for an `ExecResult`.

Exiting the context terminates the session, including unfinished commands.
Callers transferring a sandbox beyond that context use explicit `start()` and
`terminate()` to control its lifetime.

`exec()` accepts an optional `invocation_id` for callers that need to retain the
identity before making a request. Reusing it addresses the original execution;
different IDs permit intentionally identical commands.

### Results and errors

`ExecResult` contains `exit_code`, `stdout`, `stderr`, `timed_out`, and per-stream
truncation flags. Nonzero exit codes are returned as normal result data.

An execution deadline raises `ExecTimeoutError`, a subclass of `ExecError`.
Its `.result` contains the persisted `ExecResult`, including partial output and
the shell's exit code; `.handle` identifies the completed invocation. Foreground
`exec()` and `ExecHandle.result()` raise the same exception when they read a
timed-out result, including after reattachment or a duplicate-ID start.
The daemon still stores and returns the terminal result with `timed_out=true`.

An ambiguous submission or foreground connection failure raises `ExecError`.
Its `.handle` retains the execution identity so the caller can query the original
execution; the original transport exception is chained as `__cause__`.
Retrieving an interrupted/missing execution or a persisted execution error also
raises `ExecError`. Transport failures during a handle's `get` request propagate;
the caller already holds the handle and can retry that read.

### Execution, waiting, and session lifetime

- `exec(timeout=...)` limits remote command execution: default 300 seconds, range
  1–3600, including output waiting after shell exit.
- `handle.result(timeout=...)` limits local polling. A wait timeout leaves the
  command and session running and raises Python's built-in `TimeoutError`;
  the handle remains usable for a later wait. `ExecTimeoutError` represents a
  terminal execution timeout and is separate from this local wait timeout.
  An in-flight AWS request remains subject to the client's socket timeout and
  retry configuration.
- Session termination affects every command in that environment and is separate
  from either completing a command or stopping a wait.

## Execution and transport decisions

### One manager for foreground and background

Both modes use the same command arguments, process ownership, invocation records,
and result format. Foreground waits on the initial request; background returns
after registration and closes the response. Disconnecting a foreground request
only stops that wait.

E2B's envd is a useful precedent for independent process ownership: both modes
start the same managed process, and its SDK decides whether to wait on the handle.
Its background handle may retain the initial output stream. Our background start
instead finishes the initial response and supports later independent `get`
requests. RIP also adds persisted identity, deduplication, and terminal results;
PID reconnection alone does not supply that contract. See
[RIP's process-model comparison](./runtime_invocation_protocol.md#appendix-b-related-process-and-connection-models).

### AgentCore API selection

The SDK uses `InvokeAgentRuntime` for both command `start`
and `get`. sandboxd already serves `/invocations`, so this path can carry
structured requests directly to its manager without an additional helper process.

`InvokeAgentRuntimeCommand` provides native streaming stdout/stderr and an exit
event, but does not expose the persisted per-command retrieval and client-generated
identity required by RIP.
Falling back to direct command execution would bypass RIP.

Transport is an internal choice per capability. A future interactive-shell
surface should wrap AgentCore's native `InvokeAgentRuntimeCommandShell`, which
already provides persistent terminal state and reconnection. A shell ID identifies
the terminal, not every command typed into it.

## Process ownership, storage, and output

Commands run in a fresh shell, defaulting to `/bin/sh`, with the container's
environment and working directory. The SDK composes per-call `cwd` and `env`
settings into the shell command; if either setup step fails, the shell exits
before running the command.

Like envd's ordinary command path, completion waits for both shell exit and EOF
on stdout/stderr. Descendants can keep those pipes open after the shell exits;
the execution deadline still bounds that wait. Expiry kills only the direct
process and closes the output readers. The result retains captured output and
sets `timed_out=true`, even if the shell already exited with code 0.

The daemon does not kill descendants on completion or timeout. A background
service with redirected output can continue running for later commands to use.
Remaining processes are cleaned up when the Runtime session terminates; Runtime
remains the isolation boundary.

One daemon owns one local store root. It writes a start record before execution
and atomically publishes a terminal result after collecting output. The start
record excludes command text and environment values. Output still contains
whatever the workload prints.

The root is configurable with `--state-dir`. The local store survives
requests and daemon restarts, not compute replacement. After daemon loss,
start-only records with no registered owner are `interrupted`; the new daemon
does not adopt or rerun old processes. Store loss loses result and deduplication
history. Managed-storage stop/resume has not been validated. An automatic record
retention policy has not been implemented.

### Output capture and live delivery

Output collection continues whether or not a client is waiting. Each stdout/stderr
file retains its first **256 KiB**. Later bytes are drained and discarded, and the
result explicitly marks the affected stream as truncated. This bounds per-command
output storage and final response size.

The current SDK returns final output only. It does not expose callbacks,
incremental output retrieval, or stream replay. This is a delivery choice, separate
from process ownership and result persistence. A later output subscription can
attach to the same execution without making that connection own its lifecycle;
its buffering and replay contract will need a separate design decision.

## Implementation status

Real-time output, interactive shells, async APIs, public command cancellation,
file transfer, session TTL policy, managed-storage recovery, and S3 adapters
have not been implemented in the SDK.
