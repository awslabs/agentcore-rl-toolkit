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

Run from the repository root:

```bash
cd examples/strands_migration_agent

uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
```

Run the remaining commands from `examples/strands_migration_agent`.

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

After data preprocessing is done, you can start testing the agent. Start the vLLM server, the app, and submit a request. Each command runs in its own terminal:

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
# Terminal 2: Start the app server with hot reloading from the example directory
uvicorn rl_app:app --port 8080 --reload
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
    "_config": {
        "api_key": "EMPTY",
        "base_url": "http://localhost:4000/v1",
        "model_id": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "sampling_params": {"max_completion_tokens": 8192}
    }
  }'
```

> **Security note — `repo_uri` is a trust boundary.** Each invocation carries a `repo_uri`
> S3 URI; the agent downloads that repo tarball, extracts it, and runs with `shell` + `editor`
> tools against the extracted tree. The extraction uses `filter="data"` so a crafted tar
> cannot escape the work directory via path traversal, and `prompt` is a typed `str` (see
> `models.py`) so a `toolUse` content block cannot bypass the model. Even so, only point this
> agent at `repo_uri`s from a **bucket you control** — the extracted code and the prompt both
> drive a powerful agent.


## CodeArtifact Mirror (Optional)

When running RL training or evaluation with many concurrent containers, Maven Central may return HTTP 429 (rate limit) errors. You can set up an AWS CodeArtifact repository as a caching proxy to avoid this.

### 1. Create CodeArtifact resources

```bash
# Create domain
aws codeartifact create-domain --domain migration-aws-maven-mirror

# Create repository and connect it to Maven Central
aws codeartifact create-repository \
  --domain migration-aws-maven-mirror \
  --repository maven-central-cache

aws codeartifact associate-external-connection \
  --domain migration-aws-maven-mirror \
  --repository maven-central-cache \
  --external-connection public:maven-central
```

### 2. Set environment variables

When running locally, copy the example env file and add these to your `.env` file so they are picked up by `load_dotenv()`:

```bash
cp .env.example .env
```

```
CODEARTIFACT_DOMAIN=migration-aws-maven-mirror
CODEARTIFACT_OWNER=123456789012
CODEARTIFACT_REPO=maven-central-cache
```

When these environment variables are not set, the agent uses Maven Central directly (default behavior). At startup, `configure_codeartifact_token()` fetches an auth token via boto3 and generates `~/.m2/settings.xml` automatically.


## Deploy to AgentCore

Run from `examples/strands_migration_agent`. Set the VPC IDs for the
host running your inference server:

```bash
npm install -g @aws/agentcore

agentcore create --name migration --no-agent --output-dir ./deploy --skip-git
cd deploy/migration
agentcore add agent --name strands_migration_agent --type byo \
  --code-location ../.. --entrypoint rl_app.py \
  --build Container --language Python \
  --framework Strands --model-provider Bedrock --protocol HTTP \
  --network-mode VPC --vpc-id vpc-xxxxxxxxxxxxxxxxx \
  --subnets subnet-xxxxxxxxxxxxxxxxx --security-groups sg-xxxxxxxxxxxxxxxxx
```

For a publicly reachable inference server, replace the VPC flags with
`--network-mode PUBLIC`.

If using the CodeArtifact mirror, set `envVars` on the runtime in
`agentcore/agentcore.json` before deploying:

```json
"envVars": [
  {"name": "CODEARTIFACT_DOMAIN", "value": "migration-aws-maven-mirror"},
  {"name": "CODEARTIFACT_OWNER", "value": "123456789012"},
  {"name": "CODEARTIFACT_REPO", "value": "maven-central-cache"}
]
```

Deploy:

```bash
agentcore deploy --yes
agentcore status --json
cd ../..
```

Set `AGENT_RUNTIME_ARN` to the ARN printed by the deployment. Run
`agentcore deploy --yes` from `deploy/migration` to redeploy after source changes.

### Runtime permissions

Get the execution role for the deployed runtime:

```bash
EXECUTION_ROLE_ARN=$(aws bedrock-agentcore-control get-agent-runtime \
  --agent-runtime-id "${AGENT_RUNTIME_ARN##*/}" \
  --query roleArn --output text)
YOUR_EXECUTION_ROLE="${EXECUTION_ROLE_ARN##*/}"
```

#### S3 access

```bash
aws iam put-role-policy --role-name $YOUR_EXECUTION_ROLE \
  --policy-name RLToolkitAccess \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": ["s3:GetObject"],
        "Resource": "arn:aws:s3:::my-migration-bench-data/*"
      }
    ]
  }'
```

#### CodeArtifact access

Skip this step if you are not using the [CodeArtifact mirror](#codeartifact-mirror-optional).

```bash
REGION=$(aws ec2 describe-availability-zones --query 'AvailabilityZones[0].RegionName' --output text)
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

aws iam put-role-policy \
  --role-name $YOUR_EXECUTION_ROLE \
  --policy-name CodeArtifactAccess \
  --policy-document "$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "codeartifact:GetAuthorizationToken",
        "codeartifact:GetRepositoryEndpoint"
      ],
      "Resource": [
        "arn:aws:codeartifact:${REGION}:${ACCOUNT}:domain/migration-aws-maven-mirror",
        "arn:aws:codeartifact:${REGION}:${ACCOUNT}:repository/migration-aws-maven-mirror/maven-central-cache"
      ]
    },
    {
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
}
EOF
)"
```

## Evaluate

To evaluate the deployed agent on MigrationBench, create `config.toml` and fill
in its Runtime ARN, dataset path, and model settings:

```bash
cp config.example.toml config.toml
```

- **`agent_arn`**: The Runtime ARN printed by the deploy step.
- **`s3_input_bucket`**: The S3 path where `preprocess.py` uploaded the dataset. For example, if you ran `python preprocess.py --s3-bucket-name my-migration-bench-data`, the test split is at `my-migration-bench-data/tars/test/`.

```toml
[agentcore]
agent_arn = "arn:aws:bedrock-agentcore:REGION:ACCOUNT_ID:runtime/AGENT_ID"

[eval]
s3_input_bucket = "my-migration-bench-data/tars/test/"
base_url = "http://INFERENCE_SERVER_IP:4000/v1"
model_id = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
sampling_params = {max_completion_tokens = 8192}
```

### Run evaluation

```bash
# Run full evaluation
python evaluate.py --exp_id my_eval --max_concurrent 50 --timeout 1800

# Quick test with a few repos
python evaluate.py --exp_id my_eval_test --limit 5
```

Use `--max_concurrent` to limit parallel tasks, `--tps_limit` to limit session
starts per second, and `--timeout` to set the per-task deadline.
Results are saved to `<output_dir>/<exp_id>.jsonl`; pass `--output_dir ./results`
to choose a directory. The default is under the operating system's temporary directory.

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
