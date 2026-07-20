# agentcore-sandboxd

A minimal (stdlib-only) Go server that makes arbitrary Docker images satisfy the
[Bedrock AgentCore Runtime container contract](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html)
so they can be used as command-execution sandboxes.

It does exactly one thing: hold a busy/healthy flag.

- `GET /ping` → `{"status": "Healthy" | "HealthyBusy"}`. While `HealthyBusy`, AgentCore
  keeps the runtime session alive past its idle timeout.
- `POST /invocations` with `{"action": "start"}` → flips to busy; `{"action": "stop"}` →
  flips back to healthy; `{"action": "status"}` → reports without changing state.
  Unknown JSON fields are ignored (forward compatibility).

Command execution is **not** handled here — the sandbox SDK
(`agentcore_rl_toolkit.sandbox`) uses AgentCore Runtime's native
`InvokeAgentRuntimeCommand` API, which runs shell commands in the container
independently of this server.

## Build

```bash
./build.sh                       # arm64 (AgentCore's platform), -> dist/agentcore-sandboxd-linux-arm64
./build.sh --arch amd64          # for local testing on x86 hosts
./build.sh --stage ../examples/sandbox_quickstart   # also copy into a Docker build context
```

Requires either a local Go toolchain (≥1.21) or Docker (the script falls back to
building inside a `golang` container — no qemu needed, Go cross-compiles natively).

## Test

```bash
go test -race ./...
```

## Local smoke test

```bash
go run . &
curl -s localhost:8080/ping                                            # {"status":"Healthy"}
curl -s -XPOST localhost:8080/invocations -d '{"action":"start"}'      # {"status":"ok","state":"busy"}
curl -s localhost:8080/ping                                            # {"status":"HealthyBusy"}
curl -s -XPOST localhost:8080/invocations -d '{"action":"stop"}'       # {"status":"ok","state":"healthy"}
curl -s localhost:8080/ping                                            # {"status":"Healthy"}
```
