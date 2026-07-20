# Sandbox Quickstart

Run shell commands in an arbitrary Docker image deployed as a Bedrock AgentCore
Runtime sandbox. This example wraps a plain `debian:bookworm-slim` image with the
`agentcore-sandboxd` health shim and drives it with the sync sandbox client
(`agentcore_rl_toolkit.sandbox.SandboxClient`).

How it works: `agentcore-sandboxd` (a tiny Go binary, source in
[`sandboxd/`](../../sandboxd/)) satisfies AgentCore Runtime's container contract
(`/ping`, `/invocations` on port 8080) and manages the Healthy/HealthyBusy session
state. Command execution uses AgentCore Runtime's native
`InvokeAgentRuntimeCommand` API — no exec daemon runs inside the image.

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
runtime ARN when the endpoint is ready. Like `build_and_push.sh`, it is temporary
scaffolding — a future phase moves provisioning into the SDK (`SandboxClient.create()`).

The caller also needs IAM permissions for `bedrock-agentcore:InvokeAgentRuntime`,
`bedrock-agentcore:InvokeAgentRuntimeCommand`, and
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

Failing commands are results, not exceptions — `sb.exec("exit 3")` returns
`ExecResult(exit_code=3, ...)`. Timeouts likewise: `result.timed_out` is `True`
and any partial output is retained.

## Local smoke test (no AWS needed)

The server is plain HTTP, so you can exercise the contract locally:

```bash
../../sandboxd/build.sh --arch amd64        # match your host arch
../../sandboxd/dist/agentcore-sandboxd-linux-amd64 &
curl -s localhost:8080/ping                                          # {"status":"Healthy"}
curl -s -XPOST localhost:8080/invocations -d '{"action":"start"}'    # {"status":"ok","state":"busy"}
curl -s localhost:8080/ping                                          # {"status":"HealthyBusy"}
curl -s -XPOST localhost:8080/invocations -d '{"action":"stop"}'
kill %1
```
