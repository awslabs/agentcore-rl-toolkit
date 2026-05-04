# Strands Migration Agent

This agent tackles the problem of code migration from Java 8 to Java 17 as introduced in [MigrationBench](https://github.com/amazon-science/MigrationBench).
It builds upon the official [JavaMigrationAgent](https://github.com/amazon-science/JavaMigration/tree/main/java_migration_agent) with open source LLMs.
This example is under active development alongside the `agentcore-rl-toolkit` library.

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

Set the repo root and migration agent paths — these variables are used throughout this document:

```bash
export TOOLKIT_ROOT=/path/to/your/agentcore-rl-toolkit/repo
export MIGRATION_DIR=$TOOLKIT_ROOT/examples/strands_migration_agent

cd $MIGRATION_DIR

uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
uv pip install -e ../../ --force-reinstall --no-deps # install the parent repo
```

## Run locally

First, preprocess the MigrationBench dataset and upload to S3:

```bash
cd $MIGRATION_DIR

# Create S3 bucket if needed
aws s3 mb s3://my-migration-bench-data

# Full dataset (takes several hours)
python preprocess.py --s3-bucket-name my-migration-bench-data

# Or quick test with 2 repos, no S3 upload
python preprocess.py --s3-bucket-name my-migration-bench-data --max-repos-per-split 2 --skip-s3-sync
```

After data preprocessing is done, you can start testing the agent. First, prepare the environment file so Strands runs in non-interactive server mode:

```bash
cp .env.example .env
```

Then start the vLLM server, the app, and submit a request. Each command runs in its own terminal:

```bash
# Terminal 1: Start a local vLLM server
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
-tp 8 \
--port 4000 \
--enable-auto-tool-choice \
--tool-call-parser qwen3_coder \
--max-model-len 262144
```

```bash
# Terminal 2: Start the app server with hot reloading (from $MIGRATION_DIR)
cd $MIGRATION_DIR
uvicorn rl_app:app --port 8080 --reload --reload-dir ../..
```

```bash
# Terminal 3: Submit request
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Please help migrate this repo: {repo_path}. There are {num_tests} test cases in it.",
    "repo_uri": "s3://my-migration-bench-data/tars/test/15093015999__EJServer/15093015999__EJServer.tar.gz",
    "metadata_uri": "s3://my-migration-bench-data/tars/test/15093015999__EJServer/metadata.json",
    "require_maximal_migration": false,
    "use_dependency_search_tool": true,
    "apply_static_update": true,
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

Build the docker image:

```bash
docker buildx build \
  --build-context toolkit=$TOOLKIT_ROOT \
  -t migration:dev --load \
  -f $MIGRATION_DIR/.bedrock_agentcore/public_maven/Dockerfile \
  $MIGRATION_DIR
```

The agent inside the container needs AWS credentials to access S3 (for saving rollout results and downloading datasets). Since Docker containers can't access your host's AWS credential, you need to pass them explicitly via an `.env` file:

```bash
cp $MIGRATION_DIR/.env.example $MIGRATION_DIR/.env
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
docker run --network host --env-file $MIGRATION_DIR/.env migration:dev python -m rl_app

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
  --dockerfile=$MIGRATION_DIR/.bedrock_agentcore/public_maven/Dockerfile \
  --tag=dev \
  --context=$MIGRATION_DIR \
  --additional-context=toolkit=$TOOLKIT_ROOT
```

### Maven Mirror

The docker file `$MIGRATION_DIR/.bedrock_agentcore/public_maven/Dockerfile` in above commands uses public maven [source](https://repo.maven.apache.org/) to download Java dependencies. While it works reliably at most scenarios, we found sometimes in RL training, public maven source may restrict Internet acess as too many AgentCore Runtime sessions are downloading from maven at the same time, causing these sessions fail due to timeout. If you meet the same issue, please use the docker file `$MIGRATION_DIR/.bedrock_agentcore/aws_maven_mirror/Dockerfile` to build Migration agent instead. It creates a maven download mirror source at AWS CodeArtifact, which caches all downloaded Java dependencies so AgentCore Runtime sessions can directly fetch them instead of always downloading them from public maven.

First run the following commands to setup your AWS CodeArtifact repo for maven mirror source:
```bash
aws codeartifact create-domain --domain migration-aws-maven-mirror --region us-west-2
aws codeartifact create-repository --domain migration-aws-maven-mirror --repository maven-central-cache --region us-west-2
aws codeartifact associate-external-connection \
    --domain migration-aws-maven-mirror --repository maven-central-cache \
    --external-connection public:maven-central --region us-west-2
# Grant the AgentCoreRuntime IAM role codeartifact:GetAuthorizationToken
# and codeartifact:ReadFromRepository:
aws iam put-role-policy \
    --role-name AgentCoreRuntime \
    --policy-name CodeArtifactReadAccess \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "CodeArtifactRead",
                "Effect": "Allow",
                "Action": [
                    "codeartifact:GetAuthorizationToken",
                    "codeartifact:ReadFromRepository",
                    "codeartifact:GetRepositoryEndpoint"
                ],
                "Resource": "*"
            },
            {
                "Sid": "STSServiceBearerToken",
                "Effect": "Allow",
                "Action": "sts:GetServiceBearerToken",
                "Resource": "*",
                "Condition": {
                    "StringEquals": {
                        "sts:AWSServiceName": "codeartifact.amazonaws.com"
                    }
                }
            }
        ]
    }'
```
Then follow the same commands above to build Migration agent, just changing the docker file path.

## Deploy

Create your `config.toml` file and fill in the `[agentcore]` section:

```bash
cd $MIGRATION_DIR
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
cd $MIGRATION_DIR

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
cd $MIGRATION_DIR

# With custom concurrency and timeout
python evaluate_async.py --mode batch --exp_id my_eval_async --max_concurrent 50 --timeout 1800
```

Note that all arguments can also be passed via CLI to override `config.toml` values.

Both `evaluate.py` (sync) and `evaluate_async.py` (async) can run multiple agent instances concurrently via `--max_concurrent`. The difference is that the sync script submits requests sequentially — a slow submission (e.g., ACR cold start) blocks the next one — while the async script dispatches submissions as concurrent tasks so cold starts don't block each other.

#### Connection pool sizing

Both scripts accept `--max_pool_connections` (default: 10) to control the urllib3 connection pool size for boto3 clients. If this value is smaller than `--max_concurrent`, you may see urllib3 warnings like `"Connection pool is full, discarding connection"`. This is **not an error** — requests still succeed, but excess connections are created and discarded instead of being reused from the pool, adding minor TCP/TLS handshake overhead. If you want to eliminate these warnings, you can set `--max_pool_connections` to match `--max_concurrent`:

```bash
python evaluate.py --exp_id my_eval --max_concurrent 50 --max_pool_connections 50
```

Results are saved as JSONL files under `results/` (e.g., `results/my_eval.jsonl`).

## 📚 Citation
If you use our work on code migration, please cite
```bibtex
@misc{liu2025migrationbenchrepositorylevelcodemigration,
      title={MigrationBench: Repository-Level Code Migration Benchmark from Java 8},
      author={Linbo Liu and Xinle Liu and Qiang Zhou and Lin Chen and Yihan Liu and Hoan Nguyen and Behrooz Omidvar-Tehrani and Xi Shen and Jun Huan and Omer Tripp and Anoop Deoras},
      year={2025},
      eprint={2505.09569},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2505.09569},
}
```
