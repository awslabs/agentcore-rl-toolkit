# AGENTS.md

This document provides context, patterns, and guidelines for AI coding assistants working in this repository. For human contributors, see [CONTRIBUTING.md](./CONTRIBUTING.md).

## Table of Contents

- [Quick Reference](#quick-reference)
- [Project Structure](#project-structure)
- [Product Overview](#product-overview)
  - [What is ACR](#what-is-acr)
  - [Why This SDK](#why-this-sdk)
  - [Background: BedrockAgentCoreApp](#background-bedrockagentcoreapp)
  - [What agentcore-rl-toolkit Provides](#what-agentcore-rl-toolkit-provides)
  - [Migration Guide (basic_app → rl_app)](#migration-guide-basic_app--rl_app)
  - [Deployment to ACR](#deployment-to-acr)
  - [Evaluation](#evaluation)
- [Environment Variables](#environment-variables)
- [Common Tasks](#common-tasks)
- [Development Tips](#development-tips)
- [Known Limitations & TODOs](#known-limitations--todos)
- [External References](#external-references)

---

## Quick Reference

### Key Commands

```bash
# Install dependencies (root package)
uv sync

# Run tests
uv run pytest tests/

# Build and push Docker image to ECR (current approach, may change)
./scripts/build_docker_image_and_push_to_ecr.sh \
  --dockerfile=examples/strands_math_agent/.bedrock_agentcore/strands_math_agent_rl/Dockerfile \
  --tag=latest \
  --context=examples/strands_math_agent

# Run example locally
cd examples/strands_math_agent && uv sync && uv run python rl_app.py
```

### Key Files

| File | Purpose |
|------|---------|
| `src/agentcore_rl_toolkit/app.py` | `AgentCoreRLApp` base class, `@rollout_entrypoint` decorator |
| `src/agentcore_rl_toolkit/frameworks/strands/vllm_model.py` | `vLLMModel` with token ID collection for RL training |
| `src/agentcore_rl_toolkit/client.py` | `RolloutClient` for batch evaluation |
| `src/agentcore_rl_toolkit/reward_function.py` | `RewardFunction` base class |
| `examples/strands_math_agent/` | GSM8K math agent example |
| `examples/strands_migration_agent/` | Java migration agent example |

---

## Project Structure

```
agentcore-rl-toolkit/
├── src/agentcore_rl_toolkit/
│   ├── __init__.py                 # Public exports
│   ├── app.py                      # AgentCoreRLApp base class
│   ├── client.py                   # RolloutClient for batch evaluation
│   ├── reward_function.py          # RewardFunction base class
│   └── frameworks/
│       └── strands/
│           ├── __init__.py
│           ├── app.py              # StrandsAgentCoreRLApp (legacy)
│           ├── vllm_model.py       # vLLMModel with token ID collection
│           └── rollout_collector.py # StrandsRolloutCollector (legacy)
├── examples/
│   ├── strands_math_agent/         # GSM8K example
│   │   ├── .bedrock_agentcore/     # Dockerfiles for deployment
│   │   ├── basic_app.py            # Production agent
│   │   ├── rl_app.py               # RL-adapted agent
│   │   ├── reward.py               # GSM8KReward implementation
│   │   └── pyproject.toml          # Example-specific dependencies
│   ├── strands_migration_agent/    # Java migration example
│   │   ├── dev_app.py              # RL-adapted migration agent
│   │   ├── evaluate.py             # Batch evaluation script
│   │   ├── reward.py               # MigrationReward implementation
│   │   └── pyproject.toml          # Example-specific dependencies
├── tests/
│   └── test_rollout_entrypoint.py
├── scripts/
│   └── build_docker_image_and_push_to_ecr.sh
├── pyproject.toml
└── uv.lock
```

---

## Product Overview

### What is ACR

This repo provides an SDK that helps developers train their agents with **Bedrock AgentCore Runtime (ACR)**.

ACR can be viewed as Lambda functions with session continuity:
- **Session routing**: Requests with the same session ID route to the same container for multi-turn interactions
- **Session isolation**: Different session IDs use separate runtime sessions (microVMs) for strong isolation
- **Auto-scaling**: New runtime sessions spin up instantly when needed
- **Sandboxed execution**: Each session runs in a secure microVM environment

These properties make ACR ideal for deploying LLM agents, and especially suited for online RL training which requires running many parallel agent rollouts securely and efficiently.

### Why This SDK

For online RL training techniques like GRPO, developers need to:
1. Gather rollouts and corresponding rewards
2. Invoke the model being trained (hosted on a training cluster) instead of using a model API

**Goal**: Help developers adapt their production agent with minimal friction for RL training with ACR, so most of the production codebase can be directly reused while enjoying ACR's security and efficiency benefits.

### Background: BedrockAgentCoreApp

When deploying an agent on ACR, developers follow the [HTTP protocol contract](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html#container-requirements-http):
- `/invocations` endpoint: Receives requests and processes them through agent logic
- `/ping` endpoint: Health checks for AWS monitoring

AWS provides `BedrockAgentCoreApp` from the [bedrock-agentcore-sdk-python](https://github.com/aws/bedrock-agentcore-sdk-python):

**BedrockAgentCoreApp features:**
- HTTP service wrapper with `/invocations`, `/ping`, `/ws` endpoints
- Built-in logging, error handling, and session management

**Key Decorators:**
- `@app.entrypoint` - Define your agent's main logic
- `@app.websocket` - WebSocket handler for bi-directional streaming
- `@app.ping` - Custom health checks
- `@app.async_task` - Background processing

**Example:**

```python
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from dotenv import load_dotenv
from strands import Agent
from strands.models import BedrockModel
from strands_tools import calculator

app = BedrockAgentCoreApp()

load_dotenv()

model = BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")

agent = Agent(
    model=model,
    tools=[calculator],
    system_prompt=(
        "Your task is to solve the math problem. "
        + "Use calculator when applicable. "
        + 'Let\'s think step by step and output the final answer after "####".'
    ),
)


@app.entrypoint
async def invoke_agent(payload):
    """
    Invoke the agent with a payload
    """
    user_input = payload.get("prompt")

    print("User input:", user_input)

    response = await agent.invoke_async(user_input)

    return response.message["content"][0]["text"]


if __name__ == "__main__":
    app.run()
```

More details can be found at https://aws.github.io/bedrock-agentcore-starter-toolkit/user-guide/runtime/overview.html.

### What agentcore-rl-toolkit Provides

When performing rollout in ACR during RL, we need to collect the rollout and reward and return them to the training engine. A naive approach of waiting synchronously requires maintaining many TCP connections, which is brittle and hard to manage.

#### Design Pattern 1: Fire-and-forget with background async processing

With `@app.rollout_entrypoint` decorator replacing `@app.entrypoint`:
- Agent processing moves to the background immediately
- Server returns an in-progress message right away
- Health status from `/ping` is automatically managed (busy while working, idle when done)
- ACR can manage session lifecycle to avoid early termination or wasteful idle sessions

#### Design Pattern 2: S3-based result delivery with HEAD polling

Since the client won't get results directly from HTTP:
- `@app.rollout_entrypoint` requires returning rollout and reward from the entrypoint
- Rollout data is saved to S3 with a predictable key returned in the immediate HTTP response
- Client polls S3 using efficient HEAD requests to detect when each result is available
- No additional messaging infrastructure required — S3 is the single source of truth

#### Core Classes

**AgentCoreRLApp** (`src/agentcore_rl_toolkit/app.py`)
- Inherits `BedrockAgentCoreApp` - drop-in replacement
- Provides `@app.rollout_entrypoint` decorator
- Expects `_rollout` dict in payload following `RolloutConfig` model (experiment id, session id, input id, base_url, model_id)
- Framework-agnostic: works with any agent framework, not just Strands

#### Utilities

**vLLMModel** (`src/agentcore_rl_toolkit/frameworks/strands/vllm_model.py`)
- Extends Strands `OpenAIModel` for use with vLLM/SGLang inference servers
- Collects token IDs (prompt and response) and logprobs directly from the inference server
- Avoids retokenization issues that cause training instability
- Use `model.get_token_data()` to retrieve collected token data after agent execution

**RewardFunction** (`src/agentcore_rl_toolkit/reward_function.py`)
- Base class for reward implementations
- Can be any function that outputs a scalar

### Migration Guide (basic_app → rl_app)

See `examples/strands_math_agent` for a complete example adapting from `basic_app.py` to `rl_app.py`.

#### Step 1: Switch to AgentCoreRLApp & Add Reward Function

- `AgentCoreRLApp` is a thin wrapper around `BedrockAgentCoreApp` — framework-agnostic
- Users implement the reward function for their use case

```diff
- from bedrock_agentcore.runtime import BedrockAgentCoreApp
+ from agentcore_rl_toolkit import AgentCoreRLApp
+ from agentcore_rl_toolkit.frameworks.strands.vllm_model import vLLMModel
+ from reward import GSM8KReward

- app = BedrockAgentCoreApp()
+ app = AgentCoreRLApp()
+ reward_fn = GSM8KReward()
```

#### Step 2: Create Model & Agent Inside Entrypoint

- Model config (`base_url`, `model_id`) comes from the `_rollout` payload, not environment variables
- Optional `sampling_params` (e.g., `max_completion_tokens`, `temperature`) can also be passed via `_rollout` for training-engine-controlled generation settings
- `vLLMModel` collects token IDs directly from the inference server, avoiding retokenization
- `api_key` is set to `"EMPTY"` — the standard vLLM convention for servers that don't require authentication
- Model and agent are created per-invocation inside the entrypoint
- This gives flexibility for the training engine to pass runtime configuration (inference address, sampling parameters, system prompt, etc.) to accommodate different learning scenarios
- This is safe because RL rollouts are single-invocation — the agent doesn't need persistent conversation history across requests, so there's no need to keep model/agent as global state

```diff
- model = BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")
- agent = Agent(model=model, tools=[calculator], system_prompt="...")

@app.rollout_entrypoint
def invoke_agent(payload: dict):
-     response = await agent.invoke_async(user_input)
+     base_url = payload["_rollout"]["base_url"]
+     model_id = payload["_rollout"]["model_id"]
+     params = payload["_rollout"].get("sampling_params", {})
+     model = vLLMModel(client_args={"api_key": "EMPTY", "base_url": base_url}, model_id=model_id, params=params)
+     agent = Agent(model=model, tools=[calculator], system_prompt="...")
+     response = agent(user_input)
```

#### Step 3: Collect Token Data & Return Rollout

The `@rollout_entrypoint` decorator automatically:
- Executes the function in the background (works with both sync and async functions)
- Saves rollout data to S3 with a predictable key
- Handles errors and saves error rollouts for client awareness

```diff
-   return response.message["content"][0]["text"]
+   rollout_data = model.get_token_data()
+   rewards = reward_fn(response_text=response.message["content"][0]["text"], ground_truth=answer)
+   return {"rollout_data": rollout_data, "rewards": rewards}
```

Each example in `/examples` contains `basic_app.py` and `rl_app.py` (or `dev_app.py`) to demonstrate this adaptation.

### Deployment to ACR

This package relies on [bedrock-agentcore-starter-toolkit](https://github.com/aws/bedrock-agentcore-starter-toolkit) for deployment:
- CLI tool to generate Dockerfiles, build images, push to ECR, and launch on ACR
- We prioritize container (ECR image) deployment for operational simplicity

**Current workflow:**
1. Dockerfiles are generated in `examples/{agent_name}/.bedrock_agentcore/{app_name}/Dockerfile`
2. Use `scripts/build_docker_image_and_push_to_ecr.sh` to build and push:
   ```bash
   ./scripts/build_docker_image_and_push_to_ecr.sh \
     --dockerfile=examples/strands_math_agent/.bedrock_agentcore/strands_math_agent_rl/Dockerfile \
     --tag=latest \
     --context=examples/strands_math_agent
   ```
3. Training engine takes ECR URI as config for deployment
4. Model config (`base_url`, `model_id`, and optionally `sampling_params`) is passed via the `_rollout` payload at invocation time

### Evaluation

Users can evaluate agents before and after training using the same `rl_app.py`.

**RolloutClient** (`src/agentcore_rl_toolkit/client.py`) orchestrates parallel evaluation:
- **Rate limiting**: Handles ACR TPS limits (25)
- **Concurrency control**: Manages ACR session limits (1000/account) and model API rate limits
- **S3 HEAD polling**: Polls S3 for completed results using efficient HEAD requests

**Note:** For evaluation, pass the appropriate `base_url`, `model_id`, and optionally `sampling_params` in the `_rollout` payload to point to the desired inference server (training cluster or hosted cloud model).

---

## Environment Variables

| Variable | Description | When Required |
|----------|-------------|---------------|
| `AWS_REGION` | AWS region | Always |
| `AWS_ACCOUNT` | AWS account ID | Deployment |
| `ECR_REPO_NAME` | ECR repository name | Deployment |

**Note:** `BASE_URL` and `MODEL_ID` are no longer set via environment variables. They are passed in the `_rollout` payload field along with optional `sampling_params`, allowing the training engine to configure them per-invocation.

See `.env.example` for template. The build script sources `.env` for deployment values.

---

## Common Tasks

### Adding a New Example Agent

1. Create folder in `examples/{agent_name}/`
2. Add `basic_app.py` (production version using `BedrockAgentCoreApp`)
3. Add `rl_app.py` (RL-adapted version using `AgentCoreRLApp` + `vLLMModel`)
4. Add `reward.py` with `RewardFunction` implementation
5. Add `pyproject.toml` with example-specific dependencies
6. Run `uv sync` in the example folder

### Adding Support for a New Framework

1. Create `src/agentcore_rl_toolkit/frameworks/{framework}/`
2. Implement `{Framework}AgentCoreRLApp` extending `AgentCoreRLApp`
3. Implement rollout collector appropriate for the framework
4. Export in `src/agentcore_rl_toolkit/__init__.py`

### Running Tests

```bash
uv run pytest tests/
```

### Building and Pushing Docker Images

```bash
# Ensure .env is configured with AWS_REGION, AWS_ACCOUNT, ECR_REPO_NAME
./scripts/build_docker_image_and_push_to_ecr.sh \
  --dockerfile=examples/strands_math_agent/.bedrock_agentcore/strands_math_agent_rl/Dockerfile \
  --tag=my-tag \
  --context=examples/strands_math_agent
```

### Running an Example Locally

```bash
cd examples/strands_math_agent
uv sync
uv run python rl_app.py
```

---

## Development Tips

### Package Management

- This package uses **uv** for dependency management
- All dependencies are installed in `.venv` at each level
- You can inspect source code of dependencies in `.venv/lib/python*/site-packages/`

### Per-Example Environments

Each example has its own `pyproject.toml` and uv environment:
```bash
cd examples/strands_math_agent
uv sync  # Creates .venv in this folder
source .venv/bin/activate
```

To use the latest local source of `agentcore-rl-toolkit` (e.g., for testing unreleased changes like `vLLMModel`):
```bash
uv pip install -e ../../ --force-reinstall --no-deps
```

### Finding Source Code

When source locations are unclear:
```python
import module_name
print(module_name.__file__)  # Shows the file path
```

### Pre-commit Hooks

This repo uses pre-commit hooks that run automatically on `git commit`:
- **ruff**: Linting and auto-formatting (will fix issues automatically)
- **commitizen**: Enforces [Conventional Commits](https://www.conventionalcommits.org/) format (e.g., `feat:`, `fix:`, `docs:`)
- Standard checks: trailing whitespace, YAML/TOML validation

To install hooks locally:
```bash
uv run pre-commit install
```

### Code Conventions

- Return dict with `rollout_data` and `rewards` keys from `@rollout_entrypoint`
- Create model and agent inside the entrypoint function (not at module level) so config comes from the `_rollout` payload
- Use `vLLMModel.get_token_data()` to collect token IDs instead of hook-based rollout collection
- Implement reward functions as classes inheriting `RewardFunction`

### Symlink Note

`CLAUDE.md` is a symlink to `AGENTS.md` to support both instruction formats for AI coding assistants.

---

## Known Limitations & TODOs

### Testing & CI
- Limited test coverage (expansion planned)
- No CI/CD pipeline yet (planned)

### Design Improvements
- **Model creation**: `vLLMModel` is currently under `frameworks/strands/` but is largely framework-agnostic; may be moved to a shared location

### Dependency Updates
- **bedrock-agentcore-starter-toolkit**: Needs upgrade to latest version for better Docker utilities and local testing support

---

## External References

- **ACR Documentation**: https://docs.aws.amazon.com/bedrock-agentcore/
- **ACR Runtime Guide**: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html
- **bedrock-agentcore-sdk-python** (provides `BedrockAgentCoreApp`): https://github.com/aws/bedrock-agentcore-sdk-python
- **bedrock-agentcore-starter-toolkit** (CLI tools, Dockerfile generation): https://github.com/aws/bedrock-agentcore-starter-toolkit
- **Runtime SDK Overview**: https://aws.github.io/bedrock-agentcore-starter-toolkit/user-guide/runtime/overview.html
- **HTTP Protocol Contract**: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html#container-requirements-http
- **rLLM SDK (reference)**: https://rllm-project.readthedocs.io/en/latest/core-concepts/sdk/#1-define-your-agent-function
