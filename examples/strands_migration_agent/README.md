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
```bash
docker buildx build --build-context toolkit=../.. -t migration:dev --load -f Dockerfile .
```

Since Docker can't access your host's AWS credential chain, append these AWS credentials to your `.env` file.

```bash
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_REGION=us-west-2
```

Then start the server as follows, and send the request.

```bash
docker run --network host --env-file .env migration:dev python -m dev_app
```

### Build & push to ECR

```bash
./scripts/build_docker_image_and_push_to_ecr.sh \
--dockerfile=examples/strands_migration_agent/Dockerfile \
--tag=dev \
--context=examples/strands_migration_agent \
--additional-context=toolkit=.
```

## Deploy

Create your config.toml file and fill in the values.
```bash
cp config.example.toml config.toml
```

Then run
```bash
python deploy.py
```
