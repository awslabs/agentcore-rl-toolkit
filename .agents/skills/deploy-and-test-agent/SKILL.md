---
name: deploy-and-test-agent
description: Deploy repository example agents to Bedrock AgentCore Runtime or validate their changes locally and in the cloud. Use for example deployment and integration tests, including testing unpublished agentcore-rl-toolkit code.
---

# Deploy and test example agents

Read the selected example's README, Dockerfile, and dependencies. Use its
entrypoint, deployment commands, task payload, and evaluator. Follow the
machine's instructions for temporary directories, caches, and environments.

## Choose the validation

- Agent logic or request handling: run the app locally with a representative
  request from its README.
- Image or dependency changes: build and invoke the container.
- Deployment, IAM, networking, or CodeBuild changes: deploy and invoke a real
  Runtime when cloud testing is authorized.

Respect the requested test scope. A cloud test can cover image validation
without a separate local build. Documentation-only changes need no deployment.
Examples that deploy existing ECR images through an SDK script should keep that
workflow.

## Select the toolkit source

Example dependencies normally install a released toolkit. For local testing of
unpublished changes, install the example's dependencies in its environment,
then run `uv pip install -e '.[core]'` from the repository root in that same
environment. Check the imported `agentcore_rl_toolkit.__file__` to confirm the
checkout is being used.

For a container or cloud test of unpublished changes:

1. Stage the agent source and its Dockerfile in a temporary build directory.
   Build the checkout with `uv build --wheel`, setting `--out-dir` to that
   directory. Keep only the wheel for the revision being tested.
2. Install the wheel explicitly in the staged Dockerfile after the example's
   dependencies. For these examples' system-Python images, use
   `RUN uv pip install --system --reinstall ./agentcore_rl_toolkit-*.whl`
   after copying the wheel into the image, before switching to a non-root user.
   Ensure the wheel is included in the build context after applying `.dockerignore`.
3. Verify the installed package came from that wheel; the version alone cannot
   distinguish it from a PyPI release with the same version. Retain the wheel
   digest and build log with the test result.

Run the CLI from the staged directory. Keep this test override out of the
example's release dependencies. The
[Runtime test agents](../../../tests/runtime/agents/math/README.md) already
provide a Dockerfile that installs a toolkit wheel.

## Run locally

Use the Python version and environment required by the example. Keep a local
model server's dependencies in a separate environment, and use the README's
model settings and request payload.

For container debugging, use the example directory as the build context, or
the staged directory when testing a wheel. Use `--platform linux/arm64` when
checking the Runtime image; running that image locally needs an ARM64 host or
working emulation.

Select the Dockerfile for the entrypoint being tested. Math keeps separate
files under `.bedrock_agentcore/<agent_name>/Dockerfile`; pass that path with
`docker buildx build -f`. Other container examples use their root Dockerfile.

```bash
docker buildx build --platform linux/arm64 -f Dockerfile -t agent-test --load .
docker run --rm -p 8080:8080 agent-test
```

Pass required environment variables or AWS credentials
at runtime using an untracked env file or an appropriate credential mount.
Keep credential files out of the build context.

Container `localhost` refers to the container. For a model running on the host,
use a reachable host address, or Linux host networking. Check `/ping`, then send
the example's `/invocations` payload and inspect its result. Exercise `start/get`
when testing the background invocation lifecycle.

## Deploy to AWS

Use the Python `bedrock-agentcore-starter-toolkit` CLI, installed separately
from agent dependencies, and the example's deployment commands. Both Python
and npm CLIs use the command name `agentcore`; check which one is on PATH.
Confirm the AWS account with `aws sts get-caller-identity` and inspect
`.bedrock_agentcore.yaml` before updating an existing deployment.

Run `agentcore configure` and `agentcore deploy` from the example or staged
source directory. Keep `.bedrock_agentcore.yaml` for redeployment; it records
the source path, account, region, and created resources.

The CLI reuses an existing source-root `Dockerfile` first, otherwise
`.bedrock_agentcore/<agent_name>/Dockerfile`. It generates one only when
neither exists. The Dockerfile's CMD controls startup, even if `--entrypoint`
names another file. Preserve the selected Dockerfile when staging a test;
if changing Math's agent name, place its Dockerfile under that name too.

For Sandbox Quickstart, compile `sandboxd` first and pass the resulting binary
as `--entrypoint`. The CLI accepts it without a Python placeholder, but still
requires the example's `pyproject.toml` as `--requirements-file`.

Container deployment builds on ARM64 CodeBuild by default; `--local-build`
builds locally before pushing and deploying. The CLI creates ECR and execution
roles automatically. The caller needs deployment permissions, and the Runtime
role needs the example's data and model permissions. For a private model or
gateway, configure VPC subnets and security groups that allow the Runtime to
reach it. Build dependencies must also be accessible from CodeBuild.

## Verify and finish

Wait for the Runtime and endpoint to be ready, then run a small real task with
the example's evaluator or HTTP client, preserving the full task payload and
`_config`. Use `SandboxClient` for Sandbox commands.

Check the task result and expected reward, not just deployment status or HTTP
200. For SDK scripts that create and update a Runtime, exercise both paths and
confirm the DEFAULT endpoint serves the returned version after an update.
For failures, inspect build logs or the failing Runtime session before retrying.
Retain the tested source or wheel, Runtime ARN, invocation result, and cleanup
status locally.

Stop the test sessions and local processes. Keep deployments requested for
reuse. For a disposable CLI deployment, preview `agentcore destroy --agent <name>
--delete-ecr-repo --dry-run`, then use `--force` instead of `--dry-run` to
remove its resources and verify removal.
Clean up separately created test fixtures and SDK-managed test resources too.
Limit cleanup to resources created for the test; shared source buckets and
pre-existing deployments stay in place. Delete temporary project state only
after its cloud resources have been removed.
