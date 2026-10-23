# Sandbox Quickstart

Run programs in an arbitrary Docker image deployed as a Bedrock AgentCore
Runtime sandbox. This example wraps a plain `debian:bookworm-slim` image with the
`agentcore-sandboxd` daemon and drives it with the sync sandbox client
(`agentcore_rl_toolkit.sandbox.SandboxClient`).

How it works: `agentcore-sandboxd` (a tiny Go binary, source in
[`sandboxd/`](../../sandboxd/)) satisfies AgentCore Runtime's container contract
(`/ping`, `/invocations` on port 8080) and manages the Healthy/HealthyBusy session
state. Commands use RIP `start/get` over `InvokeAgentRuntime`; the daemon owns
execution and saves results independently of the client connection. Rebuild the
image when upgrading the daemon alongside the SDK.

Commands are argv lists: `sb.exec(["python", "script.py", "--name", "hello world"])`.
For shell syntax, use `sb.exec(["/bin/sh", "-c", "ls | head"])`.
The image must contain the requested executables; a shell is needed only when
explicitly invoked.

## 1. Build the binary

Run from `examples/sandbox_quickstart`:

```bash
../../sandboxd/build.sh --stage .
```

This cross-compiles a static ARM64 Linux binary. If Go is not installed, the
script uses a `golang` container. The sandbox image contains the daemon and
the workload's programs; `agentcore-rl-toolkit` runs client-side.

## 2. Deploy

```bash
uv tool install bedrock-agentcore-starter-toolkit

agentcore configure --entrypoint agentcore-sandboxd-linux-arm64 \
  --name sandbox_quickstart --requirements-file pyproject.toml \
  --deployment-type container --protocol HTTP \
  --disable-memory --disable-otel --non-interactive
agentcore deploy --agent sandbox_quickstart
```

The [Dockerfile](Dockerfile) starts the Go binary. The CLI requires
`pyproject.toml` for configuration; it does not install those Python
dependencies in the image.

Set `SANDBOX_RUNTIME_ARN` to the ARN returned by deployment. After changing
`sandboxd`, rebuild the binary and rerun `agentcore deploy --agent sandbox_quickstart`.

The caller also needs IAM permissions for `bedrock-agentcore:InvokeAgentRuntime`,
`bedrock-agentcore:StopRuntimeSession` on the runtime.

## 3. Run the demo

```bash
uv sync
SANDBOX_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/... uv run python run_sandbox.py
```

Expected output:

```text
Sandbox session: 1f0e7a2c-...
architecture: aarch64
exit_code=0 timed_out=False
stdout: hello from aarch64
/app
stderr:
stdout: hi from /tmp
Sandbox terminated.
```

Nonzero exits return results: `sb.exec(["/bin/sh", "-c", "exit 3"])` returns
`ExecResult(exit_code=3, ...)`. Execution timeouts raise `ExecTimeoutError`;
its `.result` retains partial output and the exit code, and `.handle` identifies
the execution.

```python
from agentcore_rl_toolkit.sandbox import ExecTimeoutError

with client.start() as sb:
    try:
        result = sb.exec(["/bin/sh", "-c", "printf before; sleep 5"], timeout=1)
    except ExecTimeoutError as error:
        print(error.result.stdout)  # before
```

## Background commands and recovery

```python
with client.start() as sb:
    handle = sb.exec(["/bin/sh", "-c", "sleep 2; printf done"], background=True)
    # Save sb.session_id and handle.invocation_id to transfer to another client.
    existing_handle = client.attach(sb.session_id).get_exec(handle.invocation_id)
    result = existing_handle.result(timeout=30)
    print(result.stdout)  # done
```

Exiting the context terminates the session, including unfinished commands. Use
explicit `start()`/`terminate()` when transferring ownership beyond this scope.
A local result-wait timeout raises `TimeoutError` and leaves the command running.
See the [SDK design](../../designs/sandbox_sdk.md) for recovery, error handling,
and storage/output limits.

## Local smoke test (no AWS needed)

Follow the [daemon's local smoke test](../../sandboxd/README.md#local-smoke-test)
to exercise session actions and command execution over plain HTTP.
