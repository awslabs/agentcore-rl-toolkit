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
- **Docker**: With buildx support (for building arm64 images)

## Installation

```bash
cd examples/strands_officebench_agent

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
uv pip install -e ../../ --force-reinstall --no-deps  # install agentcore-rl-toolkit
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

### 2. Build Docker image & push to ECR

The Dockerfile automatically clones the OfficeBench repo and copies the app scripts during build — no local clone needed for this step.

```bash
cd /path/to/agentcore-rl-toolkit

./scripts/build_docker_image_and_push_to_ecr.sh \
    --dockerfile=examples/strands_officebench_agent/Dockerfile \
    --context=examples/strands_officebench_agent \
    --additional-context=toolkit=. \
    --tag=v1
```

Requires a `.env` file at the repo root:
```
AWS_REGION=xxxxx
AWS_ACCOUNT=xxxxxxxxxxxxx
ECR_REPO_NAME=xxxxxxxxxxxx
```

### 3. Deploy to AgentCore

Create your config file:
```bash
cp config.example.toml config.toml
# Edit config.toml with your values (image_uri, execution_role_arn, etc.)
```

Deploy:
```bash
python deploy.py
```

This prints the `agent_arn` — add it to `config.toml`.

### 4. Run benchmark

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

Results are saved to:
- `results/{exp_id}/rollouts.jsonl` — per-task results with full conversation history
- `results/{exp_id}/summary.json` — scores by category
- `s3://{s3_output_bucket}/{exp_id}/...` — rollout data on S3

## Configuration

Example configuration is in `config.toml`:

```toml
[agentcore]
region = "us-west-2"
agent_name = "my_strands_officebench_agent"
image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/strands-officebench-agent:v1"
execution_role_arn = "arn:aws:iam::123456789012:role/AgentCoreRuntime"
agent_arn = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/agent-id"

# Network configuration (optional)
network_mode = "VPC"
subnets = ["subnet-xxx"]
security_groups = ["sg-xxx"]

[eval]
s3_input_bucket = "s3://your-bucket/officebench/"
s3_output_bucket = "your-output-bucket"
model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
# base_url = "http://your-vllm-server:8000/v1"  # uncomment for vLLM
```

### Switching models

The model is configured at evaluation time (no redeployment needed):

```bash
# Bedrock Claude Sonnet 4.5
python benchmark.py --exp_id sonnet45 \
    --model_id us.anthropic.claude-sonnet-4-5-20250929-v1:0

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

Note: thinking mode and temperature=0 are incompatible (Anthropic API requirement).

## Local Testing (no ACR)

For development and debugging, you can run tasks locally without deploying to ACR.

Install dependencies and set up the environment:
```bash
cd examples/strands_officebench_agent
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
uv pip install -e ../../ --force-reinstall --no-deps

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
├── rl_app.py           # AgentCoreRLApp with @rollout_entrypoint (runs in ACR)
├── tools.py            # 20 Strands @tool wrappers for OfficeBench apps
├── reward.py           # OfficeBenchReward with 9 evaluation functions
├── models.py           # Pydantic request models
├── utils.py            # S3 helpers (load task, setup testbed)
├── benchmark.py        # Full benchmark runner (ACR)
├── evaluate.py         # Batch evaluation via RolloutClient (ACR, lower-level)
├── preprocess.py       # Package OfficeBench tasks for S3
├── deploy.py           # Deploy container to AgentCore
├── Dockerfile          # ACR container (LibreOffice, Tesseract, apps)
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
