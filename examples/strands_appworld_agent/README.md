# Strands AppWorld Agent

An RL-trainable [AppWorld](https://github.com/StonyBrookNLP/appworld) agent using Bedrock AgentCore RL Toolkit. The agent solves day-to-day tasks by interacting with simulated app APIs (Spotify, Venmo, Gmail, etc.) via a Python REPL.

The agent uses a **ReAct code agent** pattern: a single `execute` tool runs Python code in an AppWorld sandbox environment. The agent discovers and calls APIs on-demand through the pre-loaded `apis` object, keeping tool context minimal regardless of how many APIs exist.

## Installation

Set the repo root and appworld agent paths — these variables are used throughout this document:

```bash
export TOOLKIT_ROOT=/path/to/your/agentcore-rl-toolkit/repo
export APPWORLD_DIR=$TOOLKIT_ROOT/examples/strands_appworld_agent

cd $APPWORLD_DIR

# Ensure git-lfs is installed (appworld stores bundle files in Git LFS)
git lfs install

uv venv --python 3.12
source .venv/bin/activate
UV_GIT_LFS=1 uv pip install -e . -e '../../[core]'  # example and local SDK
```

After installing, set up AppWorld data:

```bash
appworld install
appworld download data
```

## Run RL App Locally with a vLLM Server

### Start a local vLLM server

```bash
# In a separate directory/environment
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 8192 --port 4000 --enable-auto-tool-choice --tool-call-parser hermes
```

### Prepare the environment file

Prepare the environment file so Strands runs in non-interactive server mode:

```bash
cp .env.example .env
```

### Start the application server

```bash
# Start the server in one terminal
cd $APPWORLD_DIR
python rl_app.py

# Submit a request in another terminal
# Task id list could be found at data split files in the downloaded AppWorld
# dataset (data/datasets, dataset is downloaded by running the command
# `appworld download data`)
curl -X POST http://localhost:8080/invocations \
     -H "Content-Type: application/json" \
     -d '{
       "task_id": "3d9a636_1",
       "_config": {
         "base_url": "http://localhost:4000/v1",
         "model_id": "Qwen/Qwen3-4B-Instruct-2507",
         "sampling_params": {"max_completion_tokens": 8192}
       }
     }'
```

The request returns `{"reward": ...}`. Training and evaluation clients use
background HTTP start/get with the same `_config` fields.

## Docker

### Build & run locally

Build the docker image:

```bash
docker buildx build \
  --build-context toolkit=$TOOLKIT_ROOT \
  -t appworld:dev --load \
  -f $APPWORLD_DIR/Dockerfile \
  $APPWORLD_DIR
```

Then start the server as follows, and send the request.
```bash
# Run with host network so the agent can access the locally hosted vLLM server
docker run --network host --env-file $APPWORLD_DIR/.env appworld:dev python -m rl_app

# Submit request (same curl as above)
```

### Build & push to ECR

You need to build docker and push it to AWS ECR for deploying agent, running evaluation or running RL training with AgentCore.

```bash
cd $TOOLKIT_ROOT
cp .env.example .env
# Edit .env and fill in your AWS region, account ID, and ECR repo name before proceeding
# AWS_REGION=us-west-2
# AWS_ACCOUNT=your-aws-account-number
# ECR_REPO_NAME=your-ecr-repo-name
# The script uses the AWS CLI, which reads credentials from ~/.aws/credentials.
# Make sure this is configured (e.g., run `aws configure`) before proceeding.

./scripts/build_docker_image_and_push_to_ecr.sh \
  --dockerfile=$APPWORLD_DIR/Dockerfile \
  --tag=dev \
  --context=$APPWORLD_DIR \
  --additional-context=toolkit=$TOOLKIT_ROOT
```

## Deploy

Create your `config.toml` file and fill in the `[agentcore]` section:

```bash
cd $APPWORLD_DIR
cp config.example.toml config.toml
```

Edit `config.toml` with your deployment values:

```toml
[agentcore]
region = "us-west-2"
agent_name = "my_strands_appworld_agent"
image_uri = "ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/example-agent:tag"  # ECR image URI from the push step
execution_role_arn = "arn:aws:iam::ACCOUNT_ID:role/EXAMPLE_ROLE"

# Network configuration (optional — omit for PUBLIC mode)
network_mode = "VPC"
subnets = ["subnet-xxxxxxxxxxxxxxxxx"]
security_groups = ["sg-xxxxxxxxxxxxxxxxx"]
```

Then run:

```bash
python deploy.py
```

The deploy script will print the Amazon Resource Name (ARN) of your deployed agent in its output log. Copy this value — you will need it for evaluation.

## Evaluate

After deploying the agent to ACR, you can run batch evaluation against the AppWorld benchmark. The evaluation script uses `AgentCoreHttpSession` for background HTTP invocations and session cleanup, and writes completed results to JSONL.

First, fill in the `agent_arn` and the `[eval]` section in your `config.toml`:

- **`agent_arn`**: The ARN printed in the deploy step output log.

```toml
[agentcore]
agent_arn = "arn:aws:bedrock-agentcore:REGION:ACCOUNT_ID:runtime/AGENT_ID"

[eval]
base_url = "http://INFERENCE_SERVER_IP:4000/v1"
model_id = "Qwen/Qwen3-4B-Instruct-2507"
sampling_params = {temperature = 0.0}
```

The evaluator bounds concurrency with `--max_concurrent` and session starts with
`--tps_limit`, and stops every session on completion or timeout.

Ensure you have run `appworld download data` to download appworld dataset folder into the same path where you will run evaluation, evaluation script will load AppWorlds's built-in dataset splits (`train`, `dev`, `test_normal`, `test_challenge`) via `--dataset` from the dataset folder.

```bash
cd $APPWORLD_DIR

# Run full evaluation on the normal test split
python evaluate.py --dataset test_normal --exp_id my_eval --max_concurrent 50 --timeout 1800

# Quick test with a few tasks
python evaluate.py --dataset test_normal --exp_id my_eval_test --limit 5

# Only difficulty-1 tasks from the train split
python evaluate.py --dataset train --difficulty 1 --exp_id my_eval_easy

```

Note that all arguments can also be passed via CLI to override `config.toml` values.

Results are saved as `<output_dir>/<exp_id>.jsonl`; pass `--output_dir ./results`
to choose a directory. The default is under the operating system's temporary directory.

## File Structure

```
strands_appworld_agent/
├── rl_app.py            # AgentCoreRuntimeApp with @app.entrypoint (ReAct code agent)
├── reward.py            # AppWorldReward (pass_count / num_tests)
├── evaluate.py          # Async batch evaluation via AgentCoreHttpSession (ACR)
├── deploy.py            # Deploy container to AgentCore
├── Dockerfile           # ACR container (AppWorld data baked in)
├── config.example.toml  # Configuration template
├── .env.example         # Environment variable template
└── pyproject.toml       # Python dependencies
```
