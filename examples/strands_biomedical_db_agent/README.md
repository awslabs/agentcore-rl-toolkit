# Strands biomedical database agents

Two Strands agents retrieve evidence from Open Targets and GWAS Catalog. Each
accepts a self-contained task and returns a structured answer. Serve them as MCP
tools on AgentCore Runtime or connect them to a Codex main agent.

| Agent | Scope | Database tool |
| --- | --- | --- |
| `opentargets` | Open Targets causal-gene and variant evidence | Read-only GraphQL |
| `gwas_catalog` | GWAS Catalog candidate-gene selection | Read-only DuckDB SQL |

The package contains agent code, prompts, schema metadata, and deployment recipes.
Database files are downloaded from their public upstream sources. Training/evaluation
datasets and model checkpoints are supplied separately.

## Install

Use Python 3.11 or newer. From this directory:

```bash
uv sync --extra mcp --extra runtime
cp .env.example .env
# Fill in model endpoints and database locations.
```

Load this file with `uv run --env-file .env` as shown below, or Docker's `--env-file`.

Use `mcp` for MCP serving, `runtime` for training entrypoints, and `codex` for the
main-agent integration.

## Prepare the databases

**Open Targets:** connect `OPEN_TARGETS_GRAPHQL_URL` to a Platform GraphQL endpoint
matching `OPEN_TARGETS_EXPECTED_RELEASE` (default `26.06`). The agent verifies the
release and loads its complete schema. Follow the [Open Targets deployment
guide](deployment/opentargets/README.md) to run the pinned API, ClickHouse, and
OpenSearch stack.

**GWAS Catalog:** download the three official Parquet files pinned to the
2026-03-17 release:

```bash
uv run biomedical-gwas-data --directory /path/to/persistent/gwas/2026-03-17
export GWAS_CATALOG_SQL_DATA_ROOT=/path/to/persistent/gwas/2026-03-17
```

The command verifies source sizes and SHA-256 hashes. The backend exposes
`gwas.associations`, `gwas.studies`, and `gwas.samples` as views with documented
column aliases. For S3 and AgentCore setup, see [deployment](deployment/README.md).

## Agent responses

Both agents return:

```json
{
  "status": "completed",
  "answer": "...",
  "decision_summary": "Supporting evidence, record identifiers, and uncertainty."
}
```

`status` is `completed`, `unresolved`, or `unsupported`. Completed responses require
an answer; the other statuses require `answer: null`. `decision_summary` explains
the supporting evidence, limitation, or missing context.

Each call starts a new conversation, with defaults of 20 turns, a 600-second
deadline, and 8,192 output tokens per model call. Supply all relevant context in
every task.

Model-specific sampling settings belong in `*_MODEL_SAMPLING_PARAMS`. The example
environment shows Qwen settings; adjust them for your model. When using a trained
checkpoint, use its matching chat template and validate it against this example's
prompts, tool schemas, and three-field response contract.

## MCP and Codex

Set each agent's OpenAI-compatible model endpoint and model ID in `.env`.

```bash
# Local stdio MCP server; stdout is reserved for MCP messages.
uv run --env-file .env biomedical-mcp

# AgentCore MCP transport on port 8000, path /mcp.
uv run --env-file .env biomedical-mcp --transport streamable-http
```

The server exposes `ask_opentargets_agent(query)` and
`ask_gwas_catalog_agent(query)`. Both database configurations and both model
configurations are required at startup.

The [Codex integration](integrations/codex/README.md) registers these two subagent
tools alongside `query_open_targets_graphql` and `query_gwas_catalog_sql`. It keeps
conversation history in one Codex thread while each subagent call is stateless.

## Training entrypoints

Each database agent has its own `AgentCoreRuntimeApp` entrypoint. Deploy them as
separate HTTP Runtimes; each needs only its own database configuration.
Model configuration comes from `_config` on each invocation.

| Agent | Local command | Docker target |
| --- | --- | --- |
| Open Targets | `uv run --env-file .env python -m apps.opentargets` | `opentargets` |
| GWAS Catalog | `uv run --env-file .env python -m apps.gwas_catalog` | `gwas-catalog` |

Both listen on port 8080, so run one per container. Send a self-contained task
to the matching endpoint, with caller-supplied `expected_answers` for the
exact-answer reward:

```json
{
  "query": "A self-contained candidate-gene selection task",
  "expected_answers": ["CALLER_SUPPLIED_ANSWER"],
  "_config": {
    "base_url": "http://model-host:8000/v1",
    "model_id": "your-model",
    "api_key": "gateway-session-key",
    "sampling_params": {"temperature": 0.7}
  }
}
```

Returns `response` and a scalar `reward`: `1.0` for a completed answer in
`expected_answers`, otherwise `0.0`. See the
[Runtime app contract](../../designs/agentcore_runtime_app.md) for invocation options.

## Tests

```bash
uv sync
uv run --no-sync pytest
uv run --no-sync ruff check .
```

Tests cover SQL/GraphQL access limits, query errors, snapshot checksums, and
lazy initialization and retry, using synthetic Parquet files and stubbed
GraphQL responses.
