# Strands Math Agent

Run `basic_app.py` or `rl_app.py` locally or on AgentCore Runtime (ACR).
Use `rl_app.py` for training and evaluation.

## Installation

```bash
cd examples/strands_math_agent

uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
```

Install the [AgentCore CLI](https://github.com/aws/agentcore-cli) on the deployment
machine (Node.js 20.12+ required):

```bash
npm install -g @aws/agentcore
```

## Setup AWS Credentials

Since the basic app uses Amazon Bedrock to access Claude models, you'll need to configure AWS credentials and ensure model access. Make sure `aws sts get-caller-identity` returns the right identity. If not, follow the [developer guide](https://docs.aws.amazon.com/en_us/serverless-application-model/latest/developerguide/serverless-getting-started-set-up-credentials.html) to set up AWS Credentials. After setup, run `aws sts get-caller-identity` again to verify.


## Run Basic App Locally with Bedrock API

### Start the application server
```bash
# Go to the example folder
cd examples/strands_math_agent

# Start the server in one terminal
python basic_app.py

# Submit the following request in another terminal
curl -X POST http://localhost:8080/invocations \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?"}'
```

> **Security note:** The `/invocations` payloads shown above are validated by the
> entrypoint before they reach the agent. Both basic app and RL app in subsequent section
> construct an `InvocationRequest` (see `models.py`) whose `prompt` field is typed `str`, so
> a non-string value — e.g. a `toolUse` content block — is rejected before the agent runs.
> When adapting this example, keep that typed field: passing a `toolUse` content block to
> a Strands agent can bypass model invocation and dispatch a tool directly.

## Run Basic App Hosted on ACR

### Deploy

Run from `examples/strands_math_agent`:

```bash
mkdir -p deploy
cd deploy
agentcore create --name mathbasic --skip-git
cd mathbasic
agentcore add runtime --name strands_math_agent_basic --language python \
  --network-mode PUBLIC
```

On `runtimes[0]` in `agentcore/agentcore.json`, set `codeLocation` to `"../.."`,
`build` to `"Container"`, and `dockerfile` to `"Dockerfile.basic"`. Then deploy:

```bash
agentcore deploy --yes
agentcore status --json
```

### Invoke

Invoke the deployed endpoint with a prompt:

```bash
agentcore invoke --runtime strands_math_agent_basic \
  --payload '{"prompt": "Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?"}'
cd ../..
```

## Run RL App Locally with a vLLM Server

### Start a local vLLM server

In a separate environment with vLLM installed and a GPU available:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 8192 --port 4000 --enable-auto-tool-choice --tool-call-parser hermes
```

## Run RL App

`rl_app.py` uses [`AgentCoreRuntimeApp`](../../designs/agentcore_runtime_app.md),
reads model config from `_config`, and returns `{"reward": float}`. It supports the
[verl](../../src/agentcore_rl_toolkit/backends/verl/README.md),
[Tinker API](../../src/agentcore_rl_toolkit/backends/tinker_api/README.md),
[slime](../../docs/site/src/content/docs/guides/slime-backend-setup.md), and
[SageMaker](../../src/agentcore_rl_toolkit/backends/experimental/sagemaker/SETUP.md) backends.

```bash
python rl_app.py
```

Send requests from another terminal:

```bash
PAYLOAD='"prompt": "Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?", "answer": "694", "_config": {"base_url": "http://localhost:4000/v1", "model_id": "Qwen/Qwen3-4B-Instruct-2507"}'

# Ordinary request: returns {"reward": ...}
curl -X POST http://localhost:8080/invocations -H "Content-Type: application/json" -d "{$PAYLOAD}"

# Background start, then get by invocation ID
curl -X POST http://localhost:8080/invocations -H "Content-Type: application/json" \
     -d "{$PAYLOAD, \"_agentcore_runtime\": {\"version\": 1, \"operation\": \"start\", \"invocation_id\": \"inv1\", \"background\": true}}"
curl -X POST http://localhost:8080/invocations -H "Content-Type: application/json" \
     -d '{"_agentcore_runtime": {"version": 1, "operation": "get", "invocation_id": "inv1"}}'
```

### Deploy with the AgentCore CLI

Run from `examples/strands_math_agent`. Set the VPC IDs for the host
running your inference server or rollout gateway:

```bash
mkdir -p deploy
cd deploy
agentcore create --name math --skip-git
cd math
agentcore add runtime --name strands_math_agent_rl --language python \
  --network-mode VPC \
  --network-config '{"vpcId":"vpc-xxxxxxxxxxxxxxxxx","subnets":["subnet-xxxxxxxxxxxxxxxxx"],"securityGroups":["sg-xxxxxxxxxxxxxxxxx"]}'
```

In `agentcore/agentcore.json`, set `runtimes[0].codeLocation` to `"../.."`. Then deploy:

```bash
agentcore deploy --yes
agentcore status --json
cd ../..
```

Set `AGENT_RUNTIME_ARN` to the ARN printed by the deployment. For a publicly
reachable model endpoint, replace the VPC flags with `--network-mode PUBLIC`.
Run `agentcore deploy --yes` from `deploy/math` for subsequent deployments.

### Evaluate

Evaluate the deployed agent on GSM8K:

```bash
uv pip install datasets
python evaluate.py --agent_arn "$AGENT_RUNTIME_ARN" \
  --base_url "http://INFERENCE_SERVER_IP:4000/v1" \
  --model_id Qwen/Qwen3-4B-Instruct-2507 --limit 10 \
  --output_dir ./results --exp_id gsm8k_eval
```

Results are appended to `<output_dir>/<exp_id>.jsonl`. Without `--output_dir`,
they go under the operating system's temporary directory.
