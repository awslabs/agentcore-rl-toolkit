# Strands AppWorld Agent

An RL-trainable AppWorld agent using Bedrock AgentCore RL Toolkit. The agent solves day-to-day tasks by interacting with simulated app APIs (Spotify, Venmo, Gmail, etc.) via a Python REPL.

## Installation

Set the repo root and appworld agent paths — these variables are used throughout this document:

```bash
export TOOLKIT_ROOT=/path/to/your/agentcore-rl-toolkit/repo
export APPWORLD_DIR=$TOOLKIT_ROOT/examples/strands_appworld

cd $APPWORLD_DIR

# Ensure git-lfs is installed (appworld stores bundle files in Git LFS)
git lfs install

uv venv --python 3.12
source .venv/bin/activate
UV_GIT_LFS=1 uv pip install -e .
uv pip install -e ../../ --force-reinstall --no-deps  # install the parent toolkit
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

### Setup S3

```bash
aws s3 mb s3://agentcore-rl
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
       "_rollout": {
         "exp_id": "appworld_test",
         "s3_bucket": "agentcore-rl",
         "session_id": "session_001",
         "input_id": "test_0",
         "base_url": "http://localhost:4000/v1",
         "model_id": "Qwen/Qwen3-4B-Instruct-2507",
         "sampling_params": {"max_completion_tokens": 8192}
       }
     }'
```

You should see the rollout and reward saved to `s3://agentcore-rl/appworld_test/test_0_session_001.json`.

> **Note:** The `_rollout` config must include `base_url` and `model_id`, which tell the agent which inference server to use. The remaining fields (`exp_id`, `s3_bucket`, `session_id`, `input_id`) control S3 result storage and are optional — if omitted, S3 save will be skipped.

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

The agent inside the container needs AWS credentials to access S3 (for saving rollout results and downloading datasets). Since Docker containers can't access your host's AWS credential, you need to pass them explicitly via an `.env` file:

```bash
cp $APPWORLD_DIR/.env.example $APPWORLD_DIR/.env
# Edit .env and add your AWS credentials.
# If you have configured your AWS credential, you should be able to find
# them at ~/.aws/credentials.
# AWS_ACCESS_KEY_ID=your_access_key_id
# AWS_SECRET_ACCESS_KEY=your_secret_access_key
# AWS_REGION=us-west-2
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

After deploying the agent to ACR, you can run batch evaluation against the AppWorld benchmark. The evaluation scripts use `RolloutClient` to submit requests to ACR and poll S3 for results.

First, fill in the `agent_arn` and the `[eval]` section in your `config.toml`:

- **`agent_arn`**: The ARN printed in the deploy step output log.
- **`s3_output_bucket`**: The S3 bucket where evaluation rollout results will be saved.

```toml
[agentcore]
agent_arn = "arn:aws:bedrock-agentcore:REGION:ACCOUNT_ID:runtime/AGENT_ID"

[eval]
s3_output_bucket = "agentcore-rl"
base_url = "http://INFERENCE_SERVER_IP:4000/v1"
model_id = "Qwen/Qwen3-4B-Instruct-2507"
sampling_params = {temperature = 0.0}
```

### Sync evaluation

Both `evaluate.py` and `evaluate_async.py` require an `--input` argument: a local path or S3 path (`s3://bucket/key`) to a text file containing task IDs, one per line. To get the full list of task IDs, follow the [official AppWorld instructions](https://github.com/StonyBrookNLP/appworld) to download the benchmark data as a `data/` folder. The task ID files are located under `data/datasets/` (e.g., `test_normal.txt`, `test_challenge.txt`, `train.txt`, `dev.txt`).

```bash
cd $APPWORLD_DIR

# Run full evaluation on the normal test split
python evaluate.py --input /path/to/data/datasets/test_normal.txt --exp_id my_eval --max_concurrent 50 --timeout 1800

# Quick test with a few tasks
python evaluate.py --input /path/to/data/datasets/test_normal.txt --exp_id my_eval_test --limit 5
```

### Async evaluation

The async script supports two modes:
- **batch** (default): Uses `run_batch_async()` with managed concurrency
- **individual**: Uses `invoke_async()` + `gather` for fine-grained control

```bash
cd $APPWORLD_DIR

# With custom concurrency and timeout
python evaluate_async.py --input /path/to/data/datasets/test_normal.txt --mode batch --exp_id my_eval_async --max_concurrent 50 --timeout 1800
```

Note that all arguments can also be passed via CLI to override `config.toml` values.

Both `evaluate.py` (sync) and `evaluate_async.py` (async) can run multiple agent instances concurrently via `--max_concurrent`. The difference is that the sync script submits requests sequentially — a slow submission (e.g., ACR cold start) blocks the next one — while the async script dispatches submissions as concurrent tasks so cold starts don't block each other.

#### Connection pool sizing

Both scripts accept `--max_pool_connections` (default: 10) to control the urllib3 connection pool size for boto3 clients. If this value is smaller than `--max_concurrent`, you may see urllib3 warnings like `"Connection pool is full, discarding connection"`. This is **not an error** — requests still succeed, but excess connections are created and discarded instead of being reused from the pool, adding minor TCP/TLS handshake overhead. To eliminate these warnings, set `--max_pool_connections` to match `--max_concurrent`:

```bash
python evaluate.py --input /path/to/data/datasets/test_normal.txt --exp_id my_eval --max_concurrent 50 --max_pool_connections 50
```

Results are saved as JSONL files under `results/` (e.g., `results/my_eval.jsonl`).
