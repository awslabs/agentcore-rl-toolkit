# Strands Migration Agent

This agent migrates repos written in Java 8 to use Java 17. This example is under active development alongside the `agentcore-rl-toolkit` library.

## Basic Setup

Before running the agent, verify that Java 17 and Maven 3.9.6 are installed:

### Check Installation

```bash
# Java
java --version
```

Reference output:
```
openjdk 17.0.17 2025-10-21
OpenJDK Runtime Environment (build 17.0.17+10-Ubuntu-122.04)
OpenJDK 64-Bit Server VM (build 17.0.17+10-Ubuntu-122.04, mixed mode, sharing)
```

```bash
# Maven
mvn --version
```

Reference output:
```
Apache Maven 3.9.6 (bc0240f3c744dd6b6ec2920b3cd08dcc295161ae)
Maven home: /opt/maven
Java version: 17.0.17, vendor: Ubuntu, runtime: /usr/lib/jvm/java-17-openjdk-amd64
Default locale: en, platform encoding: UTF-8
OS name: "linux", version: "6.8.0-1031-aws", arch: "amd64", family: "unix"
```

### Installation Instructions

If Java or Maven are not installed, follow these instructions:

#### Install Java 17 (OpenJDK)

```bash
# Install OpenJDK 17
sudo apt update
sudo apt install -y openjdk-17-jdk

# Verify installation
java --version
```

If multiple Java versions are installed and the system's update-alternatives is still not pointing to Java 17, run:

```bash
sudo update-alternatives --config java
```

This will list all installed Java versions and let you pick Java 17.

#### Install Maven 3.9.6

```bash
# Download and install Maven
curl -O https://archive.apache.org/dist/maven/maven-3/3.9.6/binaries/apache-maven-3.9.6-bin.zip
unzip apache-maven-3.9.6-bin.zip
sudo mv apache-maven-3.9.6 /opt/

# Create symlinks
sudo ln -s /opt/apache-maven-3.9.6 /opt/maven # for MAVEN_HOME
sudo ln -s /opt/apache-maven-3.9.6/bin/mvn /usr/local/bin/mvn # so mvn works without PATH setup

# Clean up
rm apache-maven-3.9.6-bin.zip

# Verify installation
mvn --version
```

## Installation

```bash
cd examples/strands_migration_agent

uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
uv pip install -e ../../ --force-reinstall --no-deps # install the parent repo

```

## Run locally

First, preprocess the MigrationBench dataset and upload to S3:

```bash
# Create S3 bucket if needed
aws s3 mb s3://my-migration-bench-data

# Full dataset (takes several hours)
python preprocess.py --s3-bucket-name my-migration-bench-data

# Or quick test with 2 repos, no S3 upload
python preprocess.py --s3-bucket-name my-migration-bench-data --max-repos-per-split 2 --skip-s3-sync
```

After data preprocessing is done, you can start testing the agent

```bash
# Start a local vLLM server
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
-tp 8 \
--port 4000 \
--enable-auto-tool-choice \
--tool-call-parser qwen3_coder \
--max-model-len 262144

# Start the app server with hot reloading
uvicorn dev_app:app --port 8080 --reload --reload-dir ../..

# Submit request
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Please help migrate this repo: {repo_path}. There are {num_tests} test cases in it.",
    "repo_uri": "s3://{BUCKET}/tars/test/15093015999__EJServer/15093015999__EJServer.tar.gz",
    "metadata_uri": "s3://{BUCKET}/tars/test/15093015999__EJServer/metadata.json",
    "require_maximal_migration": false,
    "_rollout": {
        "exp_id": "dev",
        "s3_bucket": "agentcore-rl",
        "session_id": "session_x",
        "input_id": "prompt_y",
        "base_url": "http://localhost:4000/v1",
        "model_id": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "sampling_params": {"max_completion_tokens": 8192}
    }
  }'

```

## Docker

### Build & run locally

Set the repo root path first:

```bash
export TOOLKIT_ROOT=/path/to/your/agentcore-rl-toolkit/repo
export MIGRATION_DIR=$TOOLKIT_ROOT/examples/strands_migration_agent
```

```bash
docker buildx build \
  --build-context toolkit=$TOOLKIT_ROOT \
  -t migration:dev --load \
  -f $MIGRATION_DIR/Dockerfile \
  $MIGRATION_DIR
```

Add AWS credentials to your `.env` file since Docker can't access your host's AWS credential chain:

```bash
cp $MIGRATION_DIR/.env.example $MIGRATION_DIR/.env
# Edit .env and fill in your AWS credentials
# AWS_ACCESS_KEY_ID=your_access_key_id
# AWS_SECRET_ACCESS_KEY=your_secret_access_key
# AWS_REGION=us-west-2
```

Then start the server as follows, and send the request.
```bash
# Run with host network so the agent can access the locally hosted vLLM server
docker run --network host --env-file $MIGRATION_DIR/.env migration:dev python -m dev_app

# Submit request (same curl as above)
```

### Build & push to ECR

```bash
cd /path/to/your/agentcore-rl-toolkit/repo
cp .env.example .env
# Edit .env and fill in your AWS credentials and ECR repo name before proceeding

./scripts/build_docker_image_and_push_to_ecr.sh \
  --dockerfile=$MIGRATION_DIR/Dockerfile \
  --tag=dev \
  --context=$MIGRATION_DIR \
  --additional-context=toolkit=$TOOLKIT_ROOT
```

## Deploy

Create your `config.toml` file and fill in the `[agentcore]` section:

```bash
cd /path/to/your/agentcore-rl-toolkit/repo/examples/strands_migration_agent
cp config.example.toml config.toml
```

Edit `config.toml` with your deployment values:

```toml
[agentcore]
region = "us-west-2"
agent_name = "my_strands_migration_agent"
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

After deploying the agent to ACR, you can run batch evaluation against the MigrationBench dataset. The evaluation scripts use `RolloutClient` to submit requests to ACR and poll S3 for results.

First, fill in the `agent_arn` and the `[eval]` section in your `config.toml`:

- **`agent_arn`**: The ARN printed in the deploy step output log.
- **`s3_input_bucket`**: The S3 path where `preprocess.py` uploaded the dataset. For example, if you ran `python preprocess.py --s3-bucket-name my-migration-bench-data`, the test split is at `my-migration-bench-data/tars/test/`.
- **`s3_output_bucket`**: The S3 bucket where evaluation rollout results will be saved.

```toml
[agentcore]
agent_arn = "arn:aws:bedrock-agentcore:REGION:ACCOUNT_ID:runtime/AGENT_ID"

[eval]
s3_input_bucket = "my-migration-bench-data/tars/test/"
s3_output_bucket = "agentcore-rl"
base_url = "http://INFERENCE_SERVER_IP:4000/v1"
model_id = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
sampling_params = {max_completion_tokens = 8192}
```

### Sync evaluation

```bash
cd /path/to/your/agentcore-rl-toolkit/repo/examples/strands_migration_agent

# Run full evaluation
python evaluate.py --exp_id my_eval --max_concurrent 50 --timeout 1800

# Quick test with a few repos
python evaluate.py --exp_id my_eval_test --limit 5
```

### Async evaluation

The async script supports two modes:
- **batch** (default): Uses `run_batch_async()` with managed concurrency
- **individual**: Uses `invoke_async()` + `gather` for fine-grained control

```bash
cd /path/to/your/agentcore-rl-toolkit/repo/examples/strands_migration_agent

# With custom concurrency and timeout
python evaluate_async.py --mode batch --exp_id my_eval_async --max_concurrent 50 --timeout 1800
```

Note that all arguments can also be passed via CLI to override `config.toml` values.

Both `evaluate.py` (sync) and `evaluate_async.py` (async) can run multiple agent instances concurrently via `--max_concurrent`. The difference is that the sync script submits requests sequentially — a slow submission (e.g., ACR cold start) blocks the next one — while the async script dispatches submissions as concurrent tasks so cold starts don't block each other.

Results are saved as JSONL files under `results/` (e.g., `results/my_eval.jsonl`).
