# agentcore-sandboxd

A small, stdlib-only Go daemon that makes arbitrary images satisfy the
[AgentCore Runtime container contract](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html).
It owns shell commands and their results independently of client connections.
The Python Sandbox SDK calls it through `InvokeAgentRuntime`.

For the SDK architecture and design decisions, see
[Sandbox SDK design](../designs/sandbox_sdk.md).

## Session and execution lifecycle

- `GET /ping` reports `HealthyBusy` while the session is explicitly held or a
  command is executing/publishing its result; otherwise it reports `Healthy`.
- `POST /invocations` with `{"action":"start"}` holds the session;
  `{"action":"stop"}` releases the hold; `{"action":"status"}` reads it.
  Releasing the hold does not cancel running commands. The SDK separately calls
  `StopRuntimeSession` to terminate the environment.
- Versioned RIP `start/get` requests manage individual executions. Foreground
  requests wait for completion; background requests return immediately after
  registration. Both modes use the same process manager and persisted result.

```json
{
  "_agentcore_runtime": {
    "version": 1,
    "operation": "start",
    "invocation_id": "command-123",
    "background": true
  },
  "command": "pytest -q",
  "shell": "/bin/sh",
  "timeout": 900
}
```

`version` identifies the RIP request/response format shared by the SDK and daemon.
Both currently support only version `1`; the daemon rejects other versions.
SDK or daemon releases that keep the same protocol format keep this value.

Use the same envelope with `operation: "get"`, without command input, to retrieve
status and the final result. IDs are 1–128 characters, start with an ASCII letter
or digit, and contain only letters, digits, `.`, `_`, and `-`.

A response contains `version`, `invocation_id`, and `status`:

- `in_progress`: a registered execution still owns the command.
- `completed`: `result` contains `exit_code`, `stdout`, `stderr`, `timed_out`,
  `stdout_truncated`, and `stderr_truncated`; infrastructure execution failures
  instead contain an `error` with `code` and `message`.
- `interrupted`: a start record survives but there is no tracked execution or
  terminal result. The daemon does not adopt or rerun orphaned processes.
- `not_found`: no invocation record exists in this store.

Reusing an invocation ID addresses its existing execution and ignores the newly
supplied command. Identical commands with different IDs execute independently.
A start record is written before launching work. Terminal state is atomically
published before reporting completion. Persistence failures never report success.

## Process and storage scope

Commands run as `<shell> -c <command>`, defaulting to `/bin/sh`, with inherited
container environment and working directory. Each command owns a process group.
The execution deadline defaults to 300 seconds (range 1–3600). Deadline expiry
kills the group and returns `timed_out=true` with partial output. The manager
also cleans up remaining group members when the command exits. Commands that
escape their process group are outside this mechanism; this is not another
isolation boundary inside the Runtime session.

Output is drained independently of HTTP clients. The first **256 KiB per stream**
is saved to disk; subsequent bytes are discarded and explicitly marked truncated.
There is no real-time output subscription or replay API in this version.

`--state-dir` selects the store root (default: `agentcore-sandboxd` beneath the
OS temporary directory, which honors `TMPDIR`). One daemon owns one root:

```text
<state-dir>/<invocation-id>/
  started.json
  stdout
  stderr
  result.json
```

The start record excludes command text and environment values. Output itself may
contain whatever the command prints. Records are retained until the store is
removed; there is no automatic retention policy yet. The default local store
survives requests and daemon restarts, **not compute replacement**. Store loss
also loses deduplication history. Managed-storage stop/resume is not validated.
Code inside the sandbox can access these files; they are not a security boundary.

## Build and test

```bash
./build.sh                       # arm64 -> dist/agentcore-sandboxd-linux-arm64
./build.sh --arch amd64          # local x86 testing
./build.sh --stage ../examples/sandbox_quickstart

go test -race ./...
```

Requires Go ≥1.21 or Docker (the build script can use a Go container).
The binary is static and the sandbox image needs only the binary and a shell.
Keep this independent Go module at `sandboxd/`; `process.go` owns execution,
`store.go` owns records, and `main.go` handles HTTP/session dispatch.

## Local smoke test

```bash
go run . --state-dir "$TMPDIR/sandboxd-records" &
curl -s localhost:8080/ping
curl -s -X POST localhost:8080/invocations -d '{"action":"start"}'
curl -s -X POST localhost:8080/invocations -d '{"_agentcore_runtime":{"version":1,"operation":"start","invocation_id":"demo","background":true},"command":"sleep 1; echo done"}'
curl -s -X POST localhost:8080/invocations -d '{"_agentcore_runtime":{"version":1,"operation":"get","invocation_id":"demo"}}'
```

The HTTP address defaults to `0.0.0.0:8080`; `--listen` permits an alternate local
test address. AgentCore deployments must keep port 8080.
