# Strands Math Agent

## Installation

```bash
cd examples/strands_math_agent

# Option A: Use the main project's venv (if already activated)
uv pip install -e .

# Option B: Create a separate venv for this example
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
uv pip install -e ../../ --force-reinstall --no-deps . # install the parent repo

```

## Run Basic App With Bedrock API
```bash
cd examples/strands_math_agent

# start the server in one terminal
python basic_app.py

# submit the following request in another terminal
curl -X POST http://localhost:8080/invocations \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?"}'
```

## Run Basic App Inside Docker Locally

### Build Docker

```bash
# Make sure you are at the project root.
cd ../../

# Build Docker
docker build -t math:dev --load . -f .bedrock_agentcore/examples_strands_math_agent_basic_app/Dockerfile

# Run Docker
# Note that we override the docker CMD to avoid cluttering error logs due to missing OTLP collector, which is not set up locally.
docker run -p 8080:8080 --env-file examples/strands_math_agent/.env math:dev python -m basic_app

# Submit request
curl -X POST http://localhost:8080/invocations \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?"}'
```

## Run RL App With a Local vLLM Server
```bash

# Start vLLM server (assume access to GPU)
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 8192 --port 4000 --enable-auto-tool-choice --tool-call-parser hermes

# Create .env file from examples/strands_math_agent/.env.example
cp .env.example .env

# Update the following env vars in .env if needed
BASE_URL=http://localhost:4000/v1
MODEL_ID=Qwen/Qwen3-4B-Instruct-2507

# Submit request
# Note: the main difference between this request to RL app and that to basic app is the "_training"
# field. This field will be prepared automatically by the training framework (veRL) during RL training,
# but when we test it out locally, we will need to specify them, especially s3 bucket name and sqs url.
# You will need to create a sqs queue and s3 bucket if you don't have existing ones.

curl -X POST http://localhost:8080/invocations \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?",
       "answer": "694",
       "_training": {
         "exp_id": "test",
         "sqs_url": "https://sqs.{region}.amazonaws.com/{account_id}/{queue_name}",
         "s3_bucket": "{bucket_name}",
         "session_id": "session_123",
         "input_id": "prompt_123"
       }
     }'
```

## Run RL App Inside Docker Locally

### Build Docker
```bash
# Make sure you are at the project root.
cd ../../

# Build Docker
docker build -t math_rl:dev --load . -f .bedrock_agentcore/examples_strands_math_agent_rl_app/Dockerfile

# Run Docker
# In addition to overriding the docker CMD, we also directly use the host's network so that the agent
# can access the locally hosted model via http://localhost:4000/v1. Alternatively, replace `localhost`
# with IP of your machine in BASE_URL and keep the port mapping (-p 8080:8080)
docker run --network host --env-file examples/strands_math_agent/.env math_rl:dev python -m rl_app

# Submit request
curl -X POST http://localhost:8080/invocations \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?",
       "answer": "694",
       "_training": {
         "exp_id": "test",
         "sqs_url": "https://sqs.{region}.amazonaws.com/{account_id}/{queue_name}",
         "s3_bucket": "{bucket_name}",
         "session_id": "session_123",
         "input_id": "prompt_123"
       }
     }'
```
