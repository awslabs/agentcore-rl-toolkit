# Sandbox Quickstart

Run shell commands in an arbitrary Docker image deployed as a Bedrock AgentCore
Runtime sandbox. This example wraps a plain `debian:bookworm-slim` image with the
`agentcore-sandboxd` daemon and drives it with the sync sandbox client
(`agentcore_rl_toolkit.sandbox.SandboxClient`).

How it works: `agentcore-sandboxd` (a tiny Go binary, source in
[`sandboxd/`](../../sandboxd/)) satisfies AgentCore Runtime's container contract
(`/ping`, `/invocations` on port 8080) and manages the Healthy/HealthyBusy session
state. Commands use RIP `start/get` over `InvokeAgentRuntime`; the daemon owns
execution and saves results independently of the client connection. Rebuild the
image when upgrading from the older health-only daemon.

> **Note:** The base image must contain a shell (`/bin/sh`): commands are executed
> as shell commands inside the container. `scratch`/distroless images will not work.

## 1. Build the binary and push the image to ECR

With `.env` configured at the repo root (`ECR_REPO_NAME`, `AWS_REGION`,
`AWS_ACCOUNT` — see `.env.example`), one command does both steps:

```bash
./build_and_push.sh              # optional: pass a tag (default: sandbox-quickstart)
```

Note the sandbox image does **not** contain `agentcore-rl-toolkit` (unlike the agent
examples) — the SDK runs client-side; the image only needs the sandboxd binary and a shell.

<details>
<summary>What the wrapper runs (manual steps)</summary>

Build the sandboxd binary and stage it into this folder for the Docker build:

```bash
../../sandboxd/build.sh --stage .
```

This cross-compiles a static arm64 Linux binary (AgentCore Runtime is arm64-only
today). Works on x86 hosts — Go cross-compiles natively; if you have no Go
toolchain the script builds inside a `golang` container instead (no qemu needed).

Then build and push the image from the repo root:

```bash
./scripts/build_docker_image_and_push_to_ecr.sh \
  --dockerfile=examples/sandbox_quickstart/Dockerfile \
  --tag=sandbox-quickstart \
  --context=examples/sandbox_quickstart
```

The script builds with `--platform linux/arm64`. Since this Dockerfile only COPYs
the prebuilt binary (no RUN of arm64 tools), the build needs no qemu emulation.

</details>

## 2. Create the AgentCore runtime

```bash
uv sync                                   # installs example deps into ./.venv
cp config.example.toml config.toml        # fill in image_uri and execution_role_arn
uv run python deploy.py
```

`deploy.py` creates (or updates) the runtime from the pushed image and prints the
runtime ARN when the endpoint is ready.

The caller also needs IAM permissions for `bedrock-agentcore:InvokeAgentRuntime`,
`bedrock-agentcore:StopRuntimeSession` on the runtime.

## 3. Run the demo

```bash
SANDBOX_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/... uv run python run_sandbox.py
```

Expected output:

```text
Sandbox session: 1f0e7a2c-...
exit_code=0 timed_out=False
stdout: hello from aarch64
/app
stderr:
stdout: hi from /tmp
Sandbox terminated.
```

Nonzero exits return results: `sb.exec("exit 3")` returns
`ExecResult(exit_code=3, ...)`. Execution timeouts raise `ExecTimeoutError`;
its `.result` retains partial output and the exit code, and `.handle` identifies
the execution.

```python
from agentcore_rl_toolkit.sandbox import ExecTimeoutError

with client.start() as sb:
    try:
        result = sb.exec("printf before; sleep 5", timeout=1)
    except ExecTimeoutError as error:
        print(error.result.stdout)  # before
```

## Background commands and recovery

```python
with client.start() as sb:
    handle = sb.exec("sleep 2; printf done", background=True)
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
