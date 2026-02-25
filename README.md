
# AgentCore RL Toolkit (ART)

Toolkit to Seamlessly Enable RL Training on Any Agent with Bedrock AgentCore.

## Overview

**AgentCore RL Toolkit (ART)** is an SDK that helps developers train their LLM agents using reinforcement learning (RL) with **AWS Bedrock AgentCore Runtime (ACR)**. It extends [bedrock-agentcore-sdk-python](https://github.com/aws/bedrock-agentcore-sdk-python/tree/main) to help adapt any existing production agents for RL training with minimal code changes.

The toolkit handles the complexity of:
- **Rollout collection**: Automatically captures agent interactions (conversations, tool calls, etc.) during RL training episodes
- **ACR integration**: Works seamlessly with ACR's auto-scaling and session management for efficient parallel rollout generation
- **Data pipeline**: S3 storage for asynchronous rollout delivery to training engines
- **Reward interface**: Provides a standard base class for implementing custom reward functions

## Installation

```bash
pip install agentcore-rl-toolkit
```

Or with uv:
```bash
uv add agentcore-rl-toolkit
```

To try the example agent applications without installing the package, each example under `examples/` has standalone dependencies. See individual example READMEs (e.g., `examples/strands_math_agent/README.md`).

### What is Bedrock AgentCore Runtime (ACR)?

ACR is AWS's serverless runtime for deploying LLM agents, which can be viewed as **Lambda functions with session continuity**:

- **Session routing**: Requests with the same session ID route to the same container, enabling multi-turn interactions with persistent state
- **Session isolation**: Different session IDs use separate runtime sessions (microVMs), providing strong security isolation between concurrent requests
- **Auto-scaling**: New runtime sessions spin up instantly on demand to handle traffic spikes
- **Sandboxed execution**: Each session runs in a secure microVM environment with resource controls

While session routing is valuable for multi-turn production agents, RL rollouts typically run as single invocations. However, the other properties—**session isolation**, **auto-scaling**, and **sandboxed execution**—still make ACR particularly well-suited for **online RL training**, which requires running many parallel agent rollouts with tool uses securely and efficiently. ACR handles the infrastructure complexity while you focus on agent logic and reward design. After training, you can deploy your fine-tuned model on the same ACR stack with minimal code changes—enabling a fast path from experimentation to production.

### Key Features

- **Minimal migration effort**: Convert your production agent to RL-ready with a simple decorator change (`@app.entrypoint` → `@app.rollout_entrypoint`)
- **Framework support**: Built-in support for Strands framework with extensible architecture for other frameworks
- **Fire-and-forget pattern**: Background async processing eliminates brittle long-lived TCP connections
- **S3-based result delivery**: Rollout results stored in S3, polled via efficient S3 HEAD requests
- **Production-ready**: Direct code reuse from production agents ensures training reflects real deployment behavior

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
async def invoke_agent(payload):
    response = await agent.invoke_async(payload.get("prompt"))
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
    model = vLLMModel(client_args={"api_key": "abc", "base_url": base_url}, model_id=model_id)
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
                │ 1. Submit N prompts      │ 2. Model inference
                │    to ACR                │    calls from agents
                ▼                          │
┌────────────────────────────────────────────────────────────────┐
│                  AWS Bedrock AgentCore Runtime                 │
│        ┌───────────┐  ┌───────────┐       ┌───────────┐        │
│        │   Agent   │  │   Agent   │  ...  │   Agent   │        │
│        │ Session 1 │  │ Session 2 │       │ Session N │        │
│        └─────┬─────┘  └─────┬─────┘       └─────┬─────┘        │
│              │              │                   │              │
│              └──────────────┴───────────────────┘              │
│                             │ 3. Save rollouts + rewards       │
│                             ▼                                  │
│                   ┌───────────────────┐                        │
│                   │  S3 (rollouts)    │                        │
│                   └─────────┬─────────┘                        │
└─────────────────────────────┼──────────────────────────────────┘
                              │ 4. Poll S3 HEAD for results
                              ▼
                  Training Engine receives
                  rollouts for policy update
```

**Workflow:**
1. **Prompt submission**: Training engine submits N prompts to ACR, which auto-scales to spin up N parallel agent sessions
2. **Agent execution**: Each agent session processes its prompt, calling the inference server for model responses
3. **Rollout collection**: Agents save rollouts and rewards to S3
4. **Policy update**: Training engine polls S3 for completed rollouts and updates the model

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
