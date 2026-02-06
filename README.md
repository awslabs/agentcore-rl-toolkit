
# AgentCore RL Toolkit (ART)

Toolkit to Seamlessly Enable RL Training on Any Agent with Bedrock AgentCore.

## Overview

**AgentCore RL Toolkit (ART)** is an SDK that helps developers train their LLM agents using reinforcement learning (RL) with **AWS Bedrock AgentCore Runtime (ACR)**. It extends [bedrock-agentcore-sdk-python](https://github.com/aws/bedrock-agentcore-sdk-python/tree/main) to help adapt any existing production agents for RL training with minimal code changes.

The toolkit handles the complexity of:
- **Rollout collection**: Automatically captures agent interactions (conversations, tool calls, etc.) during RL training episodes
- **ACR integration**: Works seamlessly with ACR's auto-scaling and session management for efficient parallel rollout generation
- **Data pipeline**: Manages S3 storage and SQS messaging for asynchronous rollout delivery to training engines
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
- **Event-driven architecture**: S3 + SQS integration for reliable, scalable rollout delivery
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
from agentcore_rl_toolkit import StrandsAgentCoreRLApp, StrandsRolloutCollector
from strands import Agent

app = StrandsAgentCoreRLApp()
model = app.create_openai_compatible_model()
rollout_collector = StrandsRolloutCollector()
agent = Agent(model=model, system_prompt="...", hooks=[rollout_collector])
reward_fn = GSM8KReward() # user-defined reward function

@app.rollout_entrypoint
async def invoke_agent(payload):
    response = await agent.invoke_async(payload.get("prompt"))
    rollout_data = rollout_collector.get_rollout_data()
    rewards = reward_fn(response_text=response.message["content"][0]["text"], ground_truth=payload.get("answer"))
    return {"rollout_data": rollout_data, "rewards": rewards}
```

**Key changes:**
1. `BedrockAgentCoreApp` → `StrandsAgentCoreRLApp`
2. `BedrockModel` → `app.create_openai_compatible_model()` (points to training cluster)
3. Add `StrandsRolloutCollector` hook to capture conversations
4. `@app.entrypoint` → `@app.rollout_entrypoint`
5. Return `{"rollout_data": ..., "rewards": ...}` instead of text

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
│           ┌─────────────────┐    ┌─────────────────┐           │
│           │  S3 (rollouts)  │───►│  SQS (notify)   │           │
│           └─────────────────┘    └────────┬────────┘           │
└───────────────────────────────────────────┼────────────────────┘
                                            │ 4. Poll for results
                                            ▼
                                    Training Engine receives
                                    rollouts for policy update
```

**Workflow:**
1. **Prompt submission**: Training engine submits N prompts to ACR, which auto-scales to spin up N parallel agent sessions
2. **Agent execution**: Each agent session processes its prompt, calling the inference server for model responses
3. **Rollout collection**: Agents save rollouts and rewards to S3, send notifications via SQS
4. **Policy update**: Training engine polls SQS, collects rollouts from S3, and updates the model

This architecture enables parallel and highly efficient rollouts with secure execution during RL training. The decoupled design means training libraries only need the agent's container image to start training—agent code and dependencies stay completely separate from the training library.

### Supported Training Libraries

AgentCore runtime is currently supported by:
- **[veRL](https://github.com/volcengine/verl)**: See the integration [PR](https://github.com/volcengine/verl/pull/4216).

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
bash ./scripts/build_docker_image_and_push_to_ecr.sh --dockerfile=examples/strands_math_agent/.bedrock_agentcore/strands_math_agent_rl/Dockerfile --tag=dev
```

### Start Training with veRL

Once your container is pushed to ECR, configure veRL with the AgentCore-specific parameters:

```bash
# AgentCore configuration for veRL
actor_rollout_ref.rollout.name=agentcore
actor_rollout_ref.rollout.agentcore.agent_name=<your-agent-name>
actor_rollout_ref.rollout.agentcore.container_uri=<account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>
actor_rollout_ref.rollout.agentcore.s3_bucket=<your-s3-bucket>
```

| Parameter | Description |
|-----------|-------------|
| `agent_name` | Name for your agent in ACR |
| `container_uri` | ECR URI of your RL-ready agent container |
| `s3_bucket` | S3 bucket for storing rollout data |

For a complete example, see the [veRL AgentCore integration PR](https://github.com/volcengine/verl/pull/4216).

## Contributing

We welcome contributions!

- **Adding examples**: Create a new folder under `examples/` with its own `pyproject.toml`
- **Improving the core library**: See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup

For bug reports, feature requests, and pull request guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
