# Codex main agent

The main agent uses the pinned Python Codex SDK and Bedrock Runtime provider.
It registers four tools:

| Tool | Purpose |
| --- | --- |
| `query_open_targets_graphql` | Direct Open Targets retrieval |
| `query_gwas_catalog_sql` | Direct GWAS Catalog retrieval |
| `ask_opentargets_agent` | Delegate an Open Targets evidence task |
| `ask_gwas_catalog_agent` | Delegate a GWAS candidate-gene task |

Database schemas are prepared from the configured releases at startup, then read
on demand through Codex's shell.

## Run in Docker

Build from the example directory:

```bash
docker build --target codex -f deployment/Dockerfile -t biomedical-codex .
```

Create an external environment file with the [database/model
settings](../../.env.example), AWS authentication, and:

```dotenv
MAIN_AGENT_MODEL=your-bedrock-model-id
MAIN_AGENT_EFFORT=high
MAIN_AGENT_TIMEOUT_SECONDS=1800
AWS_REGION=us-west-2
```

Use a model available through your account's Bedrock Codex provider. The
[official Codex SDK documentation](https://learn.chatgpt.com/docs/codex-sdk)
describes the Python app-server integration. This example pins `openai-codex`
and its bundled CLI together; validate upgrades against this configuration.

```bash
docker run --rm --name biomedical-conversation \
  --env-file /path/to/biomedical-codex.env \
  -p 127.0.0.1:8080:8080 biomedical-codex

curl http://localhost:8080/invocations -H 'Content-Type: application/json' \
  -d '{"query":"Use Open Targets to retrieve disease associations for ENSG00000139618."}'
```

Use addresses reachable from inside the container. `127.0.0.1` refers to the
container itself; use a private database host and an S3 GWAS prefix for this
configuration.

Follow-up requests reuse the same container and Codex thread. Start a new
container for a new conversation. The response contains `response` and
`thread_id`. A failed/interrupted turn ends that conversation.

The dedicated container is the shell isolation boundary. Do not mount host
workspaces or a Docker socket. Its private `CODEX_HOME` and `/workspace` are
ephemeral.

## Connect subagents deployed on AgentCore

By default both subagent tools run in a local stdio child process. To use the
[MCP Runtime deployment](../../deployment/README.md), add:

```dotenv
BIOMEDICAL_RUNTIME_ARN=arn:aws:bedrock-agentcore:REGION:ACCOUNT:runtime/NAME-ID
```

The `subagents` MCP connection uses
[`mcp-proxy-for-aws-cli`](https://github.com/aws/mcp-proxy-for-aws) to sign and
forward requests using the AWS credential chain. The image installs version
1.7.0 in a separate environment because the proxy uses MCP 1.x and this example
uses MCP 2.x. The caller needs `bedrock-agentcore:InvokeAgentRuntime` on the
subagent Runtime; sessions expire according to its configured idle timeout.
Local subagent model settings are unnecessary in this mode; raw tools still
need both database configurations and network access.

To deploy the main agent itself on AgentCore, build the `codex` Docker target for
ARM64 and use the HTTP protocol. One `runtimeSessionId` represents one user
conversation. Reuse it for follow-ups; the session expires after its configured
idle timeout, discarding the conversation. See the shared
[deployment instructions](../../deployment/README.md).
