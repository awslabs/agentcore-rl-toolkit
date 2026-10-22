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

Use the staged directory as the CLI's code location. Keep this test override
out of the example's release dependencies. The
[Runtime test agents](../../../tests/runtime/agents/math/README.md) already
provide a Dockerfile that installs a toolkit wheel.

## Run locally

Use the Python version and environment required by the example. Keep a local
model server's dependencies in a separate environment, and use the README's
model settings and request payload.

For container debugging, use the example directory as the build context, or
the staged directory when testing a wheel. Use `--platform linux/arm64` when
checking the Runtime image; running that image locally needs an ARM64 host or
working emulation. For example:

```bash
docker buildx build --platform linux/arm64 -t agent-test --load .
docker run --rm -p 8080:8080 agent-test
```

Select the Dockerfile for the entrypoint being tested, such as `Dockerfile.basic`
for Math's basic app. Pass required environment variables or AWS credentials
at runtime using an untracked env file or an appropriate credential mount.
Keep credential files out of the build context.

Container `localhost` refers to the container. For a model running on the host,
use a reachable host address, or Linux host networking. Check `/ping`, then send
the example's `/invocations` payload and inspect its result. Exercise `start/get`
when testing the background invocation lifecycle.

## Deploy to AWS

Use the npm `@aws/agentcore` CLI and the example's deployment commands. Confirm
the AWS account with `aws sts get-caller-identity`. The CLI uses configured
region defaults for a new deployment and the saved target for an existing
project; inspect `agentcore/aws-targets.json` before updating one.

Keep reusable CLI projects under the example's ignored `deploy/` directory.
For a temporary deployment test, keep the project beside the staged source and
give it a distinct project name. In `agentcore/agentcore.json`, set the runtime's
`codeLocation` relative to the CLI project root; `../..` in the example recipes
points back to the example. Retain both directories for redeployment.

The CLI detects `Dockerfile` and selects a container build automatically. For a
custom filename, set both `build: "Container"` and `dockerfile`. The image's CMD
controls startup; the CLI removes the CodeZip-only `runtimeVersion` itself.

Container deployment builds on ARM64 CodeBuild. In VPC mode, check both the
Runtime's route to the model or gateway and the build infrastructure's access
to AWS services and dependency sources. Use existing suitable subnets and
security groups. The CLI provisions build roles; the caller still needs
deployment permissions, and the Runtime role needs the example's data and
model permissions. Consult the official
[CLI documentation](https://github.com/aws/agentcore-cli#readme) and
[command reference](https://github.com/aws/agentcore-cli/blob/refactor/command.md)
for execution roles and networking options.

## Verify and finish

Wait for the Runtime and endpoint to be ready, then run a small real task with
the example's evaluator, HTTP client, or `agentcore invoke --runtime <name>
--payload '<JSON>'`. `--payload` forwards the full task JSON, including `_config`,
to the Runtime.

Check the task result and expected reward, not just deployment status or HTTP
200. For SDK scripts that create and update a Runtime, exercise both paths and
confirm the DEFAULT endpoint serves the returned version after an update.
For failures, inspect build logs or the failing Runtime session before retrying.
Retain the tested source or wheel, Runtime ARN, invocation result, and cleanup
status locally.

Stop the test sessions and local processes. Keep deployments requested for
reuse. For a disposable CLI project, run `agentcore remove all --yes` followed
by `agentcore deploy --yes` from that project's directory, then verify removal.
Clean up separately created test fixtures and SDK-managed test resources too.
Limit cleanup to resources created for the test; shared bootstrap resources and
pre-existing deployments stay in place. Delete temporary project state only
after its cloud resources have been removed.
