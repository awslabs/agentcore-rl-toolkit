# Strands OfficeBench Agent

This agent evaluates LLMs on the [OfficeBench](https://github.com/zlwang-cs/OfficeBench) benchmark — 300 office automation tasks spanning calendar, email, Excel, Word, PDF, and OCR operations. It deploys to Bedrock AgentCore Runtime (ACR) for parallel evaluation at scale.

> **Note:** Results from this integration are **not directly comparable** with the original OfficeBench leaderboard. Key differences:
> - **Agent framework**: We use a Strands agent with parallel tool calling, whereas the original uses a sequential single-action-per-turn LLM policy with structured JSON output.
> - **System prompt**: The original provides detailed per-app action instructions with explicit DEMO strings (e.g., `"write text to a cell in the excel file: {'app': 'excel', 'action': 'set_cell', 'file_path': ..., 'row_idx': ..., 'column_idx': ..., 'text': ...}"`) and enforces the LLM to output a strict JSON action dict each turn. Our agent instead relies on Strands native tool definitions, where each tool's function signature and docstring are automatically converted to the model's tool-use schema — the LLM never sees raw JSON action formats.
> - **Tool implementation**: We created 20 Strands `@tool` functions (in `tools.py`) that wrap the original OfficeBench app scripts via subprocess. The tool signatures and docstrings are derived from the original scripts but adapted for Strands native tool-use format, replacing the original JSON action dispatch.
> - **Action space**: The original requires explicit app switching (`system.switch_app`) and task completion (`system.finish_task`). Our agent has all 20 tools available simultaneously and terminates naturally.
> - **Iteration control**: The original caps at 20 iterations with stuck detection (5 repeated actions). Our agent has no iteration limit.


## Benchmark Results With the Deployed Officebench in Agentcore Runtime

*Averaged over 5 runs (mean +/- std).*

| Model | Single App (93) | Two Apps (95) | Three Apps (112) | Overall (300) |
|-------|-----------------|---------------|------------------|---------------|
| **Claude Sonnet 4.5 - non-thinking (default config)** | 51.61 +/- 1.08 | 64.63 +/- 1.41 | 50.18 +/- 0.74 | **55.20 +/- 0.38** |
| **Claude Sonnet 4.5 - thinking (budget=10000)** | 48.82 +/- 2.10 | 65.90 +/- 1.91 | 47.86 +/- 1.85 | **53.87 +/- 1.07** |

## Prerequisites

- **AWS credentials**: IAM role with access to S3 and Bedrock AgentCore
- **AgentCore CLI**: `bedrock-agentcore-starter-toolkit` (pulled in via the example's `pyproject.toml`)

## Installation

```bash
cd examples/strands_officebench_agent

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e . -e '../../[core]'  # example and local SDK
```

## Quick Start

### 1. Preprocess tasks (upload to S3)

Upload all 300 OfficeBench tasks to S3 (run once). This requires a local clone of the OfficeBench repo:

```bash
git clone https://github.com/zlwang-cs/OfficeBench /path/to/OfficeBench

python preprocess.py \
    --officebench_dir /path/to/OfficeBench \
    --s3_bucket your-bucket \
    --s3_prefix officebench
```

This creates the following structure in S3:
```
s3://your-bucket/officebench/
├── 1-1/
│   ├── 0/config.json        # subtask 0
│   ├── 1/config.json        # subtask 1
│   └── testbed.tar.gz       # shared test data
├── 1-2/
│   └── ...
└── manifest.json
```

### 2. Configure the agent

Use the AgentCore CLI to generate a config (`.bedrock_agentcore.yaml`) and a default Dockerfile (under `.bedrock_agentcore/strands_officebench_agent/`):

```bash
agentcore configure \
    --entrypoint rl_app.py \
    --name strands_officebench_agent \
    --requirements-file pyproject.toml \
    --deployment-type container \
    --disable-memory \
    --non-interactive
```

If your inference server lives inside a VPC (e.g. a vLLM server on an AWS instance), add `--vpc --subnets $SUBNET_ID --security-groups $SECURITY_GROUP_ID` so the agent can reach it.

### 3. Override the generated Dockerfile

OfficeBench needs system-level dependencies (LibreOffice, Tesseract, ImageMagick) and a clone of the OfficeBench app scripts — none of which the auto-generated Dockerfile provides. This example ships a custom `Dockerfile` at the folder root that adds them; copy it over the generated one:

```bash
cp Dockerfile .bedrock_agentcore/strands_officebench_agent/Dockerfile
```

### 4. Deploy

```bash
agentcore deploy --agent strands_officebench_agent
```

This builds the ARM64 image (via CodeBuild), pushes it to ECR, and creates the AgentCore Runtime. The resulting `agent_arn` is recorded in `.bedrock_agentcore.yaml`.

Copy the config template and fill in the `agent_arn` (and any eval settings the benchmark scripts read):

```bash
cp config.example.toml config.toml
# Edit config.toml:
#   agentcore.agent_arn = "<the arn agentcore deploy printed>"
#   eval.s3_input_bucket / model_id / base_url
```

### 5. Run benchmark

```bash
# Full 300-task benchmark
python benchmark.py --exp_id my_run

# Quick test (1 task)
python benchmark.py --exp_id test --limit 1

# Only 1-app tasks
python benchmark.py --exp_id my_run_1app --category 1

# Resume if interrupted
python benchmark.py --exp_id my_run --resume
```

The benchmark uses background HTTP sessions and cleans up after each task.
Set `--output_dir ./results` to save:
- `results/{exp_id}/rollouts.jsonl` — per-task results with full conversation history
- `results/{exp_id}/summary.json` — scores by category

> **Security note — where `task_uri`/`testbed_uri` come from, and why they are a trust
> boundary.** `benchmark.py` lists the S3 prefix you populated in step 1
> (`list_all_subtasks`) and, for each subtask, derives a `task_uri`
> (`.../{task_id}/{subtask_id}/config.json`) and a `testbed_uri`
> (`.../{task_id}/testbed.tar.gz`). It puts both into each invocation payload
> (`{"task_uri": ..., "testbed_uri": ...}`) sent to the ACR agent. At rollout time the
> agent **downloads the task config and testbed from those URIs** and then runs with
> `shell` + all office tools against the extracted testbed and the task instruction —
> so whoever controls those S3 objects controls what a powerful agent executes.
>
> This example hardens the mechanics: the task config is validated against a pydantic
> `TaskConfig` (so `task` must be a string, never a `toolUse` content block that could
> bypass model invocation), and the testbed tarball is extracted with `filter="data"`
> (so a crafted archive cannot escape `/testbed` via `..`/absolute-path traversal). But
> hardening the mechanics does not make the **content** of a task instruction safe — it
> is free-form text driving `shell`. Only point the benchmark at a prefix in a **bucket
> you control**; do not accept `task_uri`/`testbed_uri` values that originate from
> untrusted principals.

The default output directory is under the operating system's temporary directory.

## Configuration

Deployment settings (image URI, execution role, network mode, ECR repo, etc.) are managed by the AgentCore CLI in `.bedrock_agentcore.yaml` — you don't need to duplicate them in `config.toml`. What `config.toml` supplies to the benchmark scripts is the deployed `agent_arn` plus the evaluation-time settings:

```toml
[agentcore]
agent_arn = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/agent-id"

[eval]
s3_input_bucket = "s3://your-bucket/officebench/"
model_id = "us.anthropic.claude-sonnet-5-5"
# base_url = "http://your-vllm-server:8000/v1"  # uncomment for vLLM
```

To re-deploy with different network settings (e.g. attaching a VPC), re-run `agentcore configure` with the new flags and `agentcore deploy --agent strands_officebench_agent`.

### Switching models

The model is configured at evaluation time (no redeployment needed):

```bash
# Bedrock Claude Sonnet 5.5
python benchmark.py --exp_id sonnet55 \
    --model_id us.anthropic.claude-sonnet-5-5

# Bedrock Claude Haiku 4.5
python benchmark.py --exp_id haiku45 \
    --model_id us.anthropic.claude-haiku-4-5-20251001-v1:0

# vLLM (local model)
python benchmark.py --exp_id qwen14b \
    --model_id qwen3_14b \
    --base_url http://10.4.209.154:8000/v1
```

### Sampling parameters

```bash
# Greedy decoding (matches original OfficeBench setting)
python benchmark.py --exp_id greedy --temperature 0

# With thinking mode (requires temperature=1, the default)
python benchmark.py --exp_id thinking --thinking_budget 10000

# Custom settings
python benchmark.py --exp_id custom --temperature 0.5 --max_tokens 4096
```

Note: thinking mode and temperature=0 are incompatible (Anthropic API requirement). Claude Sonnet 5.5 rejects `--thinking_budget` and non-default `--temperature` or `--top_p` values (it thinks adaptively by default), so run these examples with `--model_id` set to a model that accepts them, such as Claude Haiku 4.5.

## Local Testing (no ACR)

For development and debugging, you can run tasks locally without deploying to ACR.

Install dependencies and set up the environment:
```bash
cd examples/strands_officebench_agent
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e . -e '../../[core]'

# The OfficeBench app scripts hardcode /testbed/ as the working directory.
# Create a symlink so local runs write to /tmp/testbed/ instead:
sudo ln -sf /tmp/testbed /testbed
```

Run tasks:
```bash
# Single task
OFFICEBENCH_DIR=/path/to/OfficeBench python test_local.py --task_id 1-1 --subtask_id 0

# Batch local evaluation
OFFICEBENCH_DIR=/path/to/OfficeBench python run_local_eval.py --exp_id local_test --limit 10
```

## File Structure

```
strands_officebench_agent/
├── rl_app.py           # AgentCoreRuntimeApp with @app.entrypoint (runs in ACR)
├── tools.py            # 20 Strands @tool wrappers for OfficeBench apps
├── reward.py           # OfficeBenchReward with 9 evaluation functions
├── models.py           # Pydantic request models
├── utils.py            # S3 helpers (load task, setup testbed)
├── benchmark.py        # Full benchmark runner (ACR)
├── evaluate.py         # Batch evaluation via AgentCoreHttpSession (ACR, lower-level)
├── preprocess.py       # Package OfficeBench tasks for S3
├── Dockerfile          # Custom ACR container (LibreOffice, Tesseract, OfficeBench apps) — copy over the auto-generated one
├── test_local.py       # Single-task local testing (no ACR)
├── run_local_eval.py   # Batch local evaluation (no ACR)
├── config.toml         # Your configuration (gitignored)
├── config.example.toml # Configuration template
└── pyproject.toml      # Python dependencies
```

## Known Issues

### OfficeBench data issues (2 tasks)

These tasks have bugs in the ground truth and will always fail regardless of agent capability:

| Task | Issue |
|------|-------|
| `1-10/3` | `evaluate_excel_cell_value` expects `201000` but correct answer is `2001000` (2000000 + 1000) |
| `1-14/2` | Filename typo: eval references `salery.xlsx` but actual file is `salary.xlsx` |

### Local testing limitations

- PDF/Word conversion tools require LibreOffice (installed in Docker, not locally)
