
# AgentCore RL Toolkit (ART)

Toolkit to Seamlessly Enable RL Training on Any Agent with Bedrock AgentCore.

## Overview

**AgentCore RL Toolkit (ART)** is an SDK that helps developers adapt their production LLM agents for online RL training with **AWS Bedrock AgentCore Runtime (ACR)**. It extends [bedrock-agentcore-sdk-python](https://github.com/aws/bedrock-agentcore-sdk-python/tree/main) so most of your production agent code can be directly reused.

LLM agent rollouts are long-running — an agent may run for minutes or hours making tool calls. The toolkit manages this with an async-first design:

- **Agent Adaptation** (`AgentCoreRLApp`, `vLLMModel`): Adapt your production agent for RL training inside ACR — the `@rollout_entrypoint` decorator runs your agent in the background with fire-and-forget semantics, collecting token-level data, computing rewards, and saving results to S3.
- **Simplified Rollout Collection** (`RolloutClient`, `RolloutFuture`): Submit rollout requests and collect results asynchronously with a future-based API, built-in concurrency control, rate limiting, and S3 polling — for both training loops and batch evaluation.

Both components use S3 as their data layer, but the complexity is fully abstracted — your code never reads from or writes to S3 directly.

## Installation

```bash
pip install agentcore-rl-toolkit
```

Or with uv:
```bash
uv add agentcore-rl-toolkit
```

To try the example agent applications without installing the package, each example under `examples/` has standalone dependencies. See individual example READMEs (e.g., `examples/strands_math_agent/README.md`).

## What is Bedrock AgentCore Runtime (ACR)?

ACR is AWS's serverless runtime for deploying LLM agents, which can be viewed as **Long-running Lambda functions (up to 8 hours) with session continuity**:

- **Session routing**: Requests with the same session ID route to the same container, enabling multi-turn interactions with persistent state
- **Session isolation**: Different session IDs use separate runtime sessions (microVMs), providing strong security isolation between concurrent requests
- **Auto-scaling**: New runtime sessions spin up instantly on demand to handle traffic spikes
- **Sandboxed execution**: Each session runs in a secure microVM environment with resource controls

While session routing is valuable for multi-turn production agents, RL rollouts typically run as single invocations. However, the other properties—**session isolation**, **auto-scaling**, and **sandboxed execution**—still make ACR particularly well-suited for **online RL training**, which requires running many parallel agent rollouts with tool uses securely and efficiently. ACR handles the infrastructure complexity while you focus on agent logic and reward design. After training, you can deploy your fine-tuned model on the same ACR stack with minimal code changes—enabling a fast path from experimentation to production.

## Key Features

- **Minimal code changes**: Adapt your production agent with a decorator swap (`@app.entrypoint` → `@app.rollout_entrypoint`) and model replacement — most of your agent code stays the same
- **Token-accurate rollout data**: `vLLMModel` collects token IDs and logprobs directly from the inference server, avoiding retokenization issues that cause training instability
- **Async end-to-end**: Fire-and-forget server + future-based client — no long-lived TCP connections needed for long-running agent rollouts
- **Simplified rollout collection**: `RolloutClient` manages concurrency, rate limiting, and S3 result polling out of the box for both training loops and batch evaluation
- **Production parity**: Direct code reuse from production agents ensures training behavior matches real deployment

## Agent-Side: Adapting Your Agent for RL

### Starting Point: A Deployment-Ready Agent

This section shows what a deployment-ready ACR agent looks like—use it as a reference if you're new to ACR, or skip ahead if you already have a production agent. The agent must conform to [ACR's HTTP contract](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html) (endpoints like `/invocations`, `/ping`), but [`BedrockAgentCoreApp`](https://aws.github.io/bedrock-agentcore-starter-toolkit/user-guide/runtime/overview.html) handles this for you.

For Strands agents, see the [Deploy to Bedrock AgentCore guide](https://strandsagents.com/latest/documentation/docs/user-guide/deploy/deploy_to_bedrock_agentcore/python/). For other frameworks, see the [ACR Getting Started documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-getting-started.html).

A minimal Strands agent looks like (see [`basic_app.py`](examples/strands_math_agent/basic_app.py) for full example):

```python
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent
from strands.models import BedrockModel

app = BedrockAgentCoreApp()
model = BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")
agent = Agent(model=model, tools=[...], system_prompt="...")

@app.entrypoint
def invoke_agent(payload):
    response = agent(payload.get("prompt"))
    return response.message["content"][0]["text"]

if __name__ == "__main__":
    app.run()
```

### Adapting for RL Training

Starting from a deployment-ready agent like above, here are the changes needed for RL training (see [`rl_app.py`](examples/strands_math_agent/rl_app.py) for full example):
```python
from agentcore_rl_toolkit import AgentCoreRLApp
from agentcore_rl_toolkit.frameworks.strands.vllm_model import vLLMModel
from strands import Agent

app = AgentCoreRLApp()
reward_fn = GSM8KReward()  # user-defined reward function

@app.rollout_entrypoint
def invoke_agent(payload: dict):
    base_url = payload["_rollout"]["base_url"]
    model_id = payload["_rollout"]["model_id"]
    model = vLLMModel(client_args={"api_key": "EMPTY", "base_url": base_url}, model_id=model_id)
    agent = Agent(model=model, tools=[...], system_prompt="...")

    response = agent(payload.get("prompt"))
    rollout_data = model.get_token_data()
    rewards = reward_fn(response_text=response.message["content"][0]["text"], ground_truth=payload.get("answer"))
    return {"rollout_data": rollout_data, "rewards": rewards}
```

**Key changes:**
1. `BedrockAgentCoreApp` → `AgentCoreRLApp` (framework-agnostic)
2. `BedrockModel` → `vLLMModel` created inside the entrypoint with `base_url`/`model_id` from `_rollout` payload
3. `@app.entrypoint` → `@app.rollout_entrypoint`
4. Use `model.get_token_data()` to collect token IDs directly (avoids retokenization)
5. Return `{"rollout_data": ..., "rewards": ...}` instead of text

**Why create model and agent inside the entrypoint?** During RL training, the training engine can pass runtime configuration—such as inference server address, sampling parameters, and system prompt—via the `_rollout` payload, giving flexibility to accommodate different learning scenarios. This is safe because RL rollouts are single-invocation: the agent doesn't need persistent conversation history across requests, so there's no need to define model and agent as global variables.

## Client-Side: Invoking Agents and Collecting Results

LLM agent rollouts are long-running and asynchronous — an agent may run for minutes or even hours as it reasons and makes tool calls. Both sides of this toolkit are designed around that constraint:

- **Agent side** (`@rollout_entrypoint`): Fire-and-forget — the HTTP response returns immediately while the agent runs in the background, saving rollout results to S3 when done. Covered above.
- **Client-side** (`RolloutClient` / `RolloutFuture`): Submits requests and provides a future-based API to collect results asynchronously as they complete. Covered here.

### Training Integration (`invoke()` + `RolloutFuture`)

For training loops (e.g., GRPO), use `client.invoke()` to submit individual rollouts and collect results with fine-grained control:

```python
from agentcore_rl_toolkit import RolloutClient

client = RolloutClient(
    agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123456789:runtime/abc",
    s3_bucket="my-rollout-bucket",
    exp_id="grpo-run-1",
    base_url="http://vllm-server:8000/v1",
    model_id="my-model",
)

# Submit rollouts for each input (non-blocking)
futures = []
for input in inputs:
    for _ in range(num_rollouts_per_input):
        f = client.invoke(
            payload={"prompt": input["prompt"], "answer": input["answer"]},
            input_id=input["id"],
        )
        futures.append(f)

# Collect results grouped by input_id
from collections import defaultdict
results_by_input = defaultdict(list)
for f in futures:
    result = f.result(timeout=600)
    results_by_input[f.input_id].append(result)
```

**How the future lifecycle works:**
- **`client.invoke()`** sends the request to ACR and returns a `RolloutFuture` immediately — this means ACR has received the request and a background agent session is processing it.
- **`future.result(timeout=...)`** blocks until the result appears in S3, polling with exponential backoff internally. It returns the complete rollout data (token IDs, rewards, etc.) once the agent finishes and writes to S3.

### Batch Evaluation (`run_batch()`)

For evaluation, `run_batch()` provides a higher-level API that manages concurrency, timeouts, and polling automatically:

```python
for item in client.run_batch(payloads, max_concurrent_sessions=100):
    if item.success:
        print(f"Payload {item.index}: rewards={item.result['rewards']}")
    else:
        print(f"Payload {item.index} failed: {item.error}")
```

See [`examples/strands_migration_agent/evaluate.py`](examples/strands_migration_agent/evaluate.py) for a full evaluation script.

## Start Training on Example Agents

### High-Level Training Architecture

The training architecture follows a **decoupled design** where agent rollouts and the training engine run separately:

```
┌────────────────────────────────────────────────────────────────┐
│                        Training Cluster                        │
│       ┌─────────────────────┐    ┌─────────────────────┐       │
│       │   Training Engine   ├───►│  Inference Servers  │       │
│       │     (e.g., veRL)    │    │    (vLLM/SGLang)    │       │
│       └───────┬─────────────┘    └───────▲─────────────┘       │
│               │                          │                     │
└───────────────┼──────────────────────────┼─────────────────────┘
                │ 1. Submit N prompts      │                        ◄── ART Client-Side (1): RolloutClient
                │    to ACR                │ 2. Model inference     ◄── ART Agent-Side (2): vLLMModel
                ▼                          │    calls from agents
┌────────────────────────────────────────────────────────────────┐
│                  AWS Bedrock AgentCore Runtime                 │
│        ┌───────────┐  ┌───────────┐       ┌───────────┐        │
│        │   Agent   │  │   Agent   │  ...  │   Agent   │        │
│        │ Session 1 │  │ Session 2 │       │ Session N │        │
│        └─────┬─────┘  └─────┬─────┘       └─────┬─────┘        │
│              │              │                   │              │
│              └──────────────┴───────────────────┘              │
│                             │ 3. Save rollouts + rewards       │  ◄── ART Agent-Side (3): @rollout_entrypoint
│                             ▼                                  │
│                   ┌───────────────────┐                        │
│                   │  S3 (rollouts)    │                        │
│                   └─────────┬─────────┘                        │
└─────────────────────────────┼──────────────────────────────────┘
                              │ 4. Poll S3 HEAD for results         ◄── ART Client-Side (4): RolloutFuture
                              ▼
                  Training Engine receives
                  rollouts for policy update
```

**Workflow:**
1. **Prompt submission**: Training engine uses `RolloutClient` to submit N prompts to ACR — it injects rollout configs (`input_id`, `s3_bucket`, etc.) and handles rate limiting and concurrency, while ACR auto-scales N parallel agent sessions
2. **Agent execution**: Each agent session runs the `@rollout_entrypoint`, which launches the agent loop asynchronously (reporting `/ping` as busy) and calls the inference server via `vLLMModel` for model responses
3. **Rollout collection**: When the agent finishes, `@rollout_entrypoint` saves rollouts and rewards to S3 and reports `/ping` as idle so ACR can reclaim the session
4. **Policy update**: Training engine uses `RolloutFuture` to poll S3 for completed rollouts and updates the model

This architecture enables parallel and highly efficient rollouts with secure execution during RL training. The decoupled design means training libraries only need the agent's container image to start training—agent code and dependencies stay completely separate from the training library.

**Supported Training Libraries:**
- To be announced.

### Prepare Your Agent Container

ACR deploys agents as Docker containers. Most Dockerfiles can be auto-generated with the [AgentCore CLI](https://aws.github.io/bedrock-agentcore-starter-toolkit/api-reference/cli.html). Example command:

```bash
# Run from examples/strands_math_agent
agentcore configure --entrypoint rl_app.py \
  --requirements-file pyproject.toml \
  --deployment-type container --disable-memory --non-interactive
```

You can further customize these Dockerfiles if needed. Once you have a Dockerfile, follow these steps to build and push your agent image to ECR.

### Setup Credentials and Environment Variables

First, make sure `aws sts get-caller-identity` returns the right identity. If not, follow the [developer guide](https://docs.aws.amazon.com/en_us/serverless-application-model/latest/developerguide/serverless-getting-started-set-up-credentials.html) to set up AWS Credentials. After setup, run `aws sts get-caller-identity` again to verify.

Next, the build script requires info related to your AWS account. Create a `.env` file from the example:

```bash
cp .env.example .env
```

Then edit `.env` and fill in your values:
- `AWS_REGION`: Your AWS region (e.g., `us-west-2`)
- `AWS_ACCOUNT`: Your AWS account ID
- `ECR_REPO_NAME`: Your ECR repository name

### Build and Push Docker Image

```bash
# Use examples/strands_math_agent as an example
chmod +x scripts/build_docker_image_and_push_to_ecr.sh
./scripts/build_docker_image_and_push_to_ecr.sh \
  --dockerfile=examples/strands_math_agent/.bedrock_agentcore/strands_math_agent_rl/Dockerfile \
  --tag=dev \
  --context=examples/strands_math_agent
```

## Contributing

We welcome contributions!

- **Adding examples**: Create a new folder under `examples/` with its own `pyproject.toml`
- **Improving the core library**: See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup

For bug reports, feature requests, and pull request guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
