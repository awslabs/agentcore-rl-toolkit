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

Install the deployment CLI on the machine used to deploy the agent:

```bash
uv tool install bedrock-agentcore-starter-toolkit
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
Once the app works locally, we can deploy it to ACR. It's straightforward with the AgentCore CLI.

First, configure the agent. The configs will be saved to `.bedrock_agentcore.yaml`, and the Dockerfile will be saved under `.bedrock_agentcore`.

```bash
agentcore configure --entrypoint basic_app.py --name strands_math_agent_basic --requirements-file pyproject.toml --deployment-type container --disable-memory --non-interactive
```
Then, we can deploy the agent by specifying its name.

```bash
agentcore deploy --agent strands_math_agent_basic
```

### Invoke

To invoke the deployed endpoint, simply use `agentcore invoke` with the payload. Check `agentcore invoke --help` for other optional arguments.

```bash
agentcore invoke --agent strands_math_agent_basic '{"prompt": "Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?"}'
```

## Run RL App Locally with a vLLM Server

The most common setup during RL training is to use a local inference engine to serve the model being trained and generate responses. So, we will try to replicate this setting and let the RL app use a local model.

### Start a local vLLM server

This step assumes access to GPU locally. To start with, install vLLM in a separate repo and environment
if needed. Then, start the vLLM server.

```bash
# Assume you are inside examples/strands_math_agent, navigate outside the project repo entirely
cd ....

# Create a new project directory
mkdir vllm-server
cd vllm-server

# Initialize uv env and add vLLM as a dependency
uv init
uv add vllm

# Start the vLLM server using Qwen3-4B-Instruct-2507 as an example.
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

PAYLOAD='"prompt": "Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?", "answer": "694", "_config": {"base_url": "http://localhost:4000/v1", "model_id": "Qwen/Qwen3-4B-Instruct-2507"}'

# Ordinary request: returns {"reward": ...}
curl -X POST http://localhost:8080/invocations -H "Content-Type: application/json" -d "{$PAYLOAD}"

# Background start, then get by invocation ID
curl -X POST http://localhost:8080/invocations -H "Content-Type: application/json" \
     -d "{$PAYLOAD, \"_agentcore_runtime\": {\"version\": 1, \"operation\": \"start\", \"invocation_id\": \"inv1\", \"background\": true}}"
curl -X POST http://localhost:8080/invocations -H "Content-Type: application/json" \
     -d '{"_agentcore_runtime": {"version": 1, "operation": "get", "invocation_id": "inv1"}}'
```

### Deploy

If the gateway or inference server uses a private address, deploy the agent
into its VPC:

```bash
agentcore configure --entrypoint rl_app.py --name strands_math_agent_rl \
  --requirements-file pyproject.toml --deployment-type container --protocol HTTP \
  --vpc --subnets "$SUBNET_ID" --security-groups "$SECURITY_GROUP_ID" \
  --disable-memory --non-interactive
agentcore deploy --agent strands_math_agent_rl
```

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

## Appendix

The following covers instructions for a set of common dev workflows.

### Run Basic App Inside Docker Locally

```bash
# Make sure you're in examples/strands_math_agent/

# Build Docker
docker build -t math:dev --load . -f .bedrock_agentcore/strands_math_agent_basic/Dockerfile
```

Add AWS credentials to your `.env` file since Docker can't access your host's AWS credential chain:

```bash
cp .env.example .env
```

Then edit .env and fill in your own AWS credentials.

```bash
# Run Docker
# Note that we override the docker CMD to avoid cluttering error logs due to missing OTLP collector, which is not set up locally.
docker run -p 8080:8080 --env-file .env math:dev python -m basic_app

# Submit request
curl -X POST http://localhost:8080/invocations \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?"}'
```

### Run RL App Inside Docker Locally

Assume the vLLM server has been set up per the instructions above.

```bash
# Make sure you're in examples/strands_math_agent/

# Build Docker
docker build --build-context toolkit=../.. -t math_rl:dev --load . -f .bedrock_agentcore/strands_math_agent_dev/Dockerfile

# Run Docker
# In addition to overriding the docker CMD, we also directly use the host's network so that the agent
# can access the locally hosted model via http://localhost:4000/v1. Alternatively, replace `localhost`
# with IP of your machine in BASE_URL and keep the port mapping (-p 8080:8080)
docker run --network host --env-file .env math_rl:dev python -m rl_app

# Submit request
curl -X POST http://localhost:8080/invocations \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?",
       "answer": "694",
       "_config": {
         "base_url": "http://localhost:4000/v1",
         "model_id": "Qwen/Qwen3-4B-Instruct-2507"
       }
     }'
```

### Build Docker images for ACR
Currently, ACR requires ARM64 containers (AWS Graviton), so only containers built on ARM64 machines will work when deployed to agentcore runtime. By default, the `AgentCore CLI` handles this automatically by building ARM64 containers in the cloud with CodeBuild. But if you want to test or debug, you can also build to ARM64 locally using `docker buildx`.

```bash
# Run only once
docker buildx create --use

docker buildx build --build-context toolkit=../.. --platform linux/arm64 -t math_rl:dev --load . -f .bedrock_agentcore/strands_math_agent_dev/Dockerfile
```

### Manual deployment
Instead of letting `AgentCore CLI` handle everything, we can also build docker image (ARM64) and push to ECR on our own. See `scripts/build_docker_image_and_push_to_ecr.sh` for more detail.

We can also use an existing IAM role for the agent's execution role instead of using the auto-generated one. See [doc](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-permissions.html) for essential permissions for ACR.

```bash
agentcore configure -e my_agent.py --execution-role arn:aws:iam::111122223333:role/MyRole
```

And in general, for more manual control over deployment, you can use boto3.

```python
import boto3

# Create the client
client = boto3.client('bedrock-agentcore-control', region_name="us-east-1")

# Call the CreateAgentRuntime operation
response = client.create_agent_runtime(
    agentRuntimeName='hello-strands',
    agentRuntimeArtifact={
        'containerConfiguration': {
            # Your ECR image Uri
            'containerUri': '123456789012.dkr.ecr.us-east-1.amazonaws.com/my-agent:latest'
        }
    },
    networkConfiguration={"networkMode":"PUBLIC"},
    # Your AgentCore Runtime role arn
    roleArn='arn:aws:iam::123456789012:role/AgentRuntimeRole'
)
```
