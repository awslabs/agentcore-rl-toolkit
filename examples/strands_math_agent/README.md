# Strands Math Agent

We provide instructions to run `basic_app.py` and `rl_app.py` both as local servers and as
http endpoints deployed to AgentCore Runtime (ACR).

## Installation

```bash
cd examples/strands_math_agent

uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
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

### Setup S3

As explained in the main [README](../../README.md#key-features), RL apps follow a fire-and-forget pattern where rollout and reward are saved to S3. The client polls S3 using efficient HEAD requests to detect when each result is available. We recommend setting up this resource and testing locally first to catch any issues early.

```bash
# Create S3 bucket
aws s3 mb s3://agentcore-rl
```

### Start the application server

Now we can test the RL app similar to what we did with the basic app above.

```bash
# Start the server in one terminal
python rl_app.py

# Submit the following request in another terminal
# Note in the payload, we also provide the ground truth answer to calculate reward.
# Replace {REGION} and {ACCOUNT} with your AWS region and account ID.
curl -X POST http://localhost:8080/invocations \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?",
       "answer": "694",
       "_rollout": {
         "exp_id": "test",
         "s3_bucket": "agentcore-rl",
         "session_id": "session_123",
         "input_id": "prompt_123",
         "base_url": "http://localhost:4000/v1",
         "model_id": "Qwen/Qwen3-4B-Instruct-2507"
       }
     }'
```
You should see the rollout and reward saved to `s3://agentcore-rl/test/prompt_123_session_123.json`.

> **Note:** The `_rollout` config must include `base_url` and `model_id`, which tell the agent which inference server to use. The remaining fields (`exp_id`, `s3_bucket`, `session_id`, `input_id`) control S3 result storage and are optional — if omitted, S3 save will be skipped. During training, the full `_rollout` config is injected automatically by the training engine. We recommend testing the full flow locally.


## Run RL App Hosted on ACR
Similar to the basic app, we can also deploy our RL app to ACR with the CLI tool. One caveat, however, is to make sure ACR can access the locally hosted vLLM server. By default, ACR is deployed to public net. So if the vLLM server resides in a VPC, ACR will not be able to access it.

### Deploy

**Option 1**

Assuming the vLLM server is hosted on an AWS managed instance inside a VPC, we can deploy the ACR to the same VPC to ensure access. This corresponds to the scenario where users use the training infra from AWS.

```bash
# Get BASE_URL with private IP
BASE_URL="http://$(hostname -I | awk '{print $1}'):4000/v1"
MODEL_ID=Qwen/Qwen3-4B-Instruct-2507

# Specify subnet and security group info from your instance's network details
SUBNET_ID="subnet-0123456789abcdefg"
SECURITY_GROUP_ID="sg-0123456789abcdefg"

# Configure the agent with details above
agentcore configure --entrypoint rl_app.py --name strands_math_agent_rl --requirements-file pyproject.toml --deployment-type container --vpc --subnets $SUBNET_ID --security-groups $SECURITY_GROUP_ID --disable-memory --non-interactive
```

**Option 2**
It's also possible that the inference servers are presented as public URLs. In this case, we can directly deploy ACR to public net. The agent can invoke the inference server much like it uses an external API model.

```bash
agentcore configure --entrypoint rl_app.py --name strands_math_agent_rl --requirements-file pyproject.toml --deployment-type container --disable-memory --non-interactive
```

While this option is feasible thanks to the [decoupled design](../../README.md#high-level-training-architecture), we have not extensively tested it yet. We will explore this path in the near future, and potentially add API key auth option.

Deploy the agent after it has been configured through either option mentioned above.

```bash
agentcore deploy --agent strands_math_agent_rl
```

### Setup IAM Permissions for ACR

The S3 bucket was already created in the local setup section. Now we need to grant the ACR agent's execution role permission to access it.

```bash
# Find your rl agent's execution_role in .bedrock_agentcore.yaml and set the role name
# e.g., arn:aws:iam::123456789:role/AmazonBedrockAgentCoreSDKRuntime-us-west-2-abc123 -> AmazonBedrockAgentCoreSDKRuntime-us-west-2-abc123
ROLE_NAME="YOUR_ROLE_NAME_HERE"

# Add S3 permissions to the execution role
aws iam put-role-policy --role-name $ROLE_NAME \
  --policy-name RLToolkitAccess \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": ["s3:PutObject", "s3:GetObject"],
        "Resource": "arn:aws:s3:::agentcore-rl/*"
      }
    ]
  }'
```

### Invoke

```bash
agentcore invoke --agent strands_math_agent_rl '{
  "prompt": "Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?",
  "answer": "694",
  "_rollout": {
    "exp_id": "test",
    "s3_bucket": "agentcore-rl",
    "session_id": "session_213",
    "input_id": "prompt_213",
    "base_url": "http://localhost:4000/v1",
    "model_id": "Qwen/Qwen3-4B-Instruct-2507"
  }
}'
```

You should be able to see the rollout and reward saved to `s3://agentcore-rl/test/prompt_213_session_213.json`.

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
docker build -t math_rl:dev --load . -f .bedrock_agentcore/strands_math_agent_rl/Dockerfile

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
       "_rollout": {
         "exp_id": "test",
         "s3_bucket": "agentcore-rl",
         "session_id": "session_321",
         "input_id": "prompt_321",
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

docker buildx build --platform linux/arm64 -t math_rl:dev --load . -f .bedrock_agentcore/strands_math_agent_rl/Dockerfile
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
