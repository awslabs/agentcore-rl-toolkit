# Strands biomedical database agents

Two Strands agents retrieve evidence from Open Targets and GWAS Catalog. Each
accepts a self-contained task and returns a structured answer. Run them directly,
serve them as MCP tools on AgentCore Runtime, or connect them to a Codex main agent.

| Agent | Scope | Private database tool |
| --- | --- | --- |
| `opentargets` | Open Targets causal-gene and variant evidence | Read-only GraphQL |
| `gwas_catalog` | GWAS Catalog candidate-gene selection | Read-only DuckDB SQL |

The package contains agent code, prompts, schema metadata, and deployment recipes.
Database files are downloaded from their public upstream sources. Training/evaluation
datasets and model checkpoints are supplied separately.

## Install

Use Python 3.11 or newer. From this directory:

```bash
export UV_PROJECT_ENVIRONMENT="$TMPDIR/biomedical-agent-venv"
uv sync --extra mcp --extra aws --extra runtime
cp .env.example .env
# Fill in model endpoints and database locations.
```

Load this file with `uv run --env-file .env` as shown below, or Docker's `--env-file`.
The JSON values are dotenv values, not shell assignments to source.

Extras are independent: `mcp` serves the specialist tools, `aws` adds S3 access,
`runtime` adds the two training entrypoints, and `codex` installs the optional main-agent SDK.
The core package can also be installed with `pip install .`.

This example pins Strands 1.61.0 and MCP 2.1.1, within Strands' `mcp<2.2`
dependency range. The MCP servers use the SDK's `MCPServer` API.

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

## Run an agent

Set each agent's OpenAI-compatible model endpoint and model ID in `.env`.
For example:

```bash
uv run --env-file .env biomedical-agent opentargets \
  'Use Open Targets to retrieve disease associations for ENSG00000139618. Cite the records supporting your answer.'

uv run --env-file .env biomedical-agent gwas_catalog \
  'Using GWAS Catalog evidence for Alzheimer disease, choose one of APOE, BRCA1, or CFTR. Explain the supporting study and variant evidence.'
```

These are illustrative requests, without gold labels or expected scientific answers.

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
the supporting evidence, limitation, or missing context. Runtime/model failures
raise errors instead of becoming scientific decisions.

Each call creates a fresh Strands agent with `context_manager=False`,
up to 20 turns, a 600-second deadline, and 8,192 output tokens per model call by
default. Supply all relevant context in every task. Query tools bound responses
and report errors separately from empty results.

Model-specific sampling settings belong in `*_MODEL_SAMPLING_PARAMS`. The example
environment shows Qwen settings; adjust them for your model. When using a trained
checkpoint, use its matching chat template and validate it against this example's
prompts, tool schemas, and three-field response contract.

## MCP and Codex

```bash
# Local stdio MCP server; stdout is reserved for MCP messages.
uv run --env-file .env biomedical-mcp

# AgentCore MCP transport on port 8000, path /mcp.
uv run --env-file .env biomedical-mcp --transport streamable-http
```

The server exposes `ask_opentargets_agent(query)` and
`ask_gwas_catalog_agent(query)`. Both database configurations and both model
configurations are required at startup.

The [Codex integration](integrations/codex/README.md) registers these two specialist
tools alongside `query_open_targets_graphql` and `query_gwas_catalog_sql`. It keeps
conversation history in one Codex thread while each specialist call is stateless.

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

Only `query` goes to the model. Reward is exact membership, not an independent
scientific correctness assessment. Successful results contain `response` and
`reward`. Ordinary HTTP requests return the result directly.
`AgentCoreHttpClient` can start background invocations and retrieve results through
the Runtime `start/get` protocol; result delivery uses HTTP and filesystem records.
Model calls routed through the rollout gateway are captured for training.
See the [Runtime app contract](../../designs/agentcore_runtime_app.md).
Training scripts and datasets are outside this example.

## Tests

```bash
uv sync --extra aws
uv run --no-sync pytest
uv run --no-sync ruff check .
```

Tests cover SQL/GraphQL access limits, query errors, snapshot checksums, and
immutable S3 publication. They use synthetic database rows and stubbed HTTP/S3
responses, with no AWS credentials, database downloads, or model calls.
