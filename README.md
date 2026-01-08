
# AgentCore RL Toolkit (ART)

Toolkit for Seamlessly Enabling RL Training on Any Agent with Bedrock AgentCore.

## Repo Structure

- **Main package**: `agentcore-rl-toolkit` is a thin wrapper around of [bedrock-agentcore-sdk-python](https://github.com/aws/bedrock-agentcore-sdk-python/tree/main) that allows developers to start RL training their production agent with only a few lines of code change.
- **Examples**: Located in `examples/` directory, each with their own `pyproject.toml` and dependencies. Their corresponding docker files are located in `.bedrock_agentcore`, most of which have been generated automatically (see [instructions below](#prepare-docker-file)).


## Start Training on Example Agents
AgentCore runtime is currently supported by the following training library.
- [verl](https://github.com/volcengine/verl): related [PR](https://github.com/volcengine/verl/pull/4216).

Before training, build the docker for the RL-ready application and upload to ECR. To do this, follow the steps below:

### Setup Credentials and Environment Variables

First, make sure `aws sts get-caller-identity` returns the right identity. If not, follow the [developer guide](https://docs.aws.amazon.com/en_us/serverless-application-model/latest/developerguide/serverless-getting-started-set-up-credentials.html) to set up AWS Credentials. After setup, run `aws sts get-caller-identity` again to verify.

Next, the build script requires info related to your AWS account. Create a `.env` file from the example:

```bash
cp .env.example .env
```

Then edit `.env` and fill in your values:
- `AWS_REGION`: Your AWS region (e.g., `us-west-2`)
- `AWS_ACCOUNT`: Your AWS account ID
- `ECR_REPO_NAME`: Your ECR repository name

### Build and Push Docker Image

```bash
# Use examples/strands_math_agent as an example
chmod +x scripts/build_docker_image_and_push_to_ecr.sh
bash ./scripts/build_docker_image_and_push_to_ecr.sh --dockerfile=.bedrock_agentcore/examples_strands_math_agent_rl_app/Dockerfile --tag=dev
```

Then, go to the training library of your choice and simply provide agentcore specific config args to start training.


## Development

### Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install uv if you haven't already, follow the installation [guide](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) here.

### For Package Development

If you're developing or contributing to the `agentcore-rl-toolkit` package itself:

```bash
# Enter the repository
cd agentcore-rl-toolkit

# Create and activate uv environment
uv venv --python 3.13
source .venv/bin/activate

# Install with development dependencies
uv sync --frozen --extra dev

# Install pre-commit hooks
pre-commit install
```

Additionally, when co-developing the toolkit together with examples, add the following to the example app's docker file so that changes to the toolkit is reflected in the container.

```bash
COPY . .
RUN uv pip install --force-reinstall --no-deps .
```

### For Running Examples

Each example has its own dependencies and can be installed independently. Follow the README for specific examples there (e.g., `examples/strands_math_agent/README.md`).

## Appendix

### Prepare Docker file

Docker file for most examples can be automatically generated with the `agentcore` CLI. Use `examples/strands_math_agent` as an example:

```bash
agentcore configure --entrypoint examples/strands_math_agent/rl_app.py --requirements-file examples/strands_math_agent/pyproject.toml --deployment-type container --disable-memory --non-interactive
```

Make sure to run the command in project root.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
