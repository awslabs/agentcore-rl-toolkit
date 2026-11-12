# Scale AI AgentEnv on AgentCore

Run a Strands agent against AgentEnv's simulated Slack environment through
[AgentCore session endpoints](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-session-endpoints.html).
[server.py](server.py) copies the official
[SlackEnv example](https://github.com/scaleapi/agentenv-framework/blob/3cf9dd363ce9e63197b79d15834179b450bd295f/packages/agentenv-protocol/README.md#L8)
under Apache-2.0, preserving its environment logic and adding AgentCore server
settings. It simulates Slack by storing messages in memory.

The environment runs in one Runtime session. Strands and the demo run on the
caller machine; Strands uses a Bedrock model to decide the tool call.
One Python server listens on port 8000:

| Step | Caller | Endpoint | Purpose |
| --- | --- | --- | --- |
| Discover | Demo | `GET /.well-known/agent-env.json` | Read the environment card |
| Initialize task state | Demo | `POST /agentenv`: `data/reset`, `data/add` | Clear state and seed a welcome message |
| Perform the task | Strands | `/mcp` | Discover and call `slack_send_message` |
| Grade | Demo | `POST /agentenv`: `data/get` | Read state and compute reward |
| Clean up | Demo | `StopRuntimeSession` | Stop the cloud session |

The task asks the agent to post exactly `Hello from AgentCore` to `general`.
Reward is `1.0` if the welcome message and that one new message are the final
state, otherwise `0.0`. Grading happens in the demo using the exported state.

## Why session endpoints?

The default MCP invocation entrypoint forwards requests to `/mcp`.
AgentEnv also needs the discovery and `/agentenv` routes. Session endpoints
preserve HTTP methods and paths, so all three routes on port 8000 are reachable.
`InvokeAgentRuntime` sends an MCP `ping` to establish the session and return its URL;
subsequent requests use that URL with SigV4.
The demo passes one UUID as both `mcpSessionId` and `runtimeSessionId`, so the SDK
sends matching session headers. Cleanup uses the same UUID with `StopRuntimeSession`. See
[MCP session management](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-mcp-protocol-contract.html#mcp-session-management).

The bootstrap `ping` is a JSON-RPC method sent to `/mcp`. Strands then sends
MCP `initialize` to negotiate the protocol. Task state is initialized
separately through `data/reset` and `data/add`. MCP transport is stateless;
the environment state lives in the server process.

The example uses AgentEnv's protocol SDK with MCP SDK 1.x.

## Run locally

From this directory, install the example and this checkout's toolkit client.
The session endpoint helper is not yet in the released toolkit:

```bash
uv sync --extra client
uv pip install -e '../..[rollout]'
uv run --no-sync python server.py
```

The `--no-sync` flag preserves the editable toolkit installation.
In another terminal, from this directory, select a Bedrock model with tool
support that your AWS identity can invoke:

```bash
export AWS_DEFAULT_REGION=REGION
export MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0
uv run --no-sync python demo.py \
  --local-url http://127.0.0.1:8000 --model-id "$MODEL_ID"
```

Both local and cloud modes require AWS credentials and `bedrock:InvokeModel`
access to the selected model or inference profile and its underlying models.
The model runs through the caller's credentials.

A successful run returns `"reward": 1.0`.

## Deploy

Build and push an ARM64 image to ECR:

```bash
export IMAGE_URI=ACCOUNT.dkr.ecr.REGION.amazonaws.com/REPOSITORY:TAG
docker buildx build --platform linux/arm64 --provenance=false -t "$IMAGE_URI" --push .
uv run --no-sync python deploy.py \
  --name agentenv_slack --image "$IMAGE_URI" \
  --role arn:aws:iam::ACCOUNT:role/YOUR_RUNTIME_EXECUTION_ROLE \
  --region REGION
```

The execution role needs a trust policy for `bedrock-agentcore.amazonaws.com`,
ECR image pull access, and CloudWatch Logs write access. The deploying identity needs
`bedrock-agentcore:CreateAgentRuntime`, `bedrock-agentcore:GetAgentRuntime`,
and `iam:PassRole` for the execution role.

## Run against AgentCore

The calling identity needs `bedrock-agentcore:InvokeAgentRuntime`,
`bedrock-agentcore:InvokeAgentRuntimeProxy`, and
`bedrock-agentcore:StopRuntimeSession`, scoped to the Runtime and its `DEFAULT`
endpoint as required by those actions.

```bash
uv run --no-sync python demo.py \
  --runtime-arn YOUR_RUNTIME_ARN --model-id "$MODEL_ID"
```

Delete the Runtime when finished:

```bash
aws bedrock-agentcore-control delete-agent-runtime \
  --agent-runtime-id YOUR_RUNTIME_ID --region REGION
```

## Adapt the pattern

[server.py](server.py) keeps the upstream `SlackEnv` tools and data handlers.
Its `data/add` handler accepts inline `DataPart` seeds of the form
`{"messages": [{"channel": "general", "text": "Welcome"}]}`.
It also supports the upstream `FilePart` seed formats.
[demo.py](demo.py) owns the ACR session and uses ordinary MCP and AgentEnv
requests to drive it. Change the seed, agent task, and grading condition
together for a different task.

The toolkit's `session_endpoint_client` reuses `SigV4HttpxAuth`, adding the
payload hash required by the session proxy. The demo passes that auth to
Strands' `MCPClient` to sign its requests.
