# TMax agent on AgentCore

`AgentCoreRuntimeApp` runs TMax's Vanillux2 loop against an OpenAI-compatible
model endpoint; its bash tool calls a sandboxd session in the task's image.
The eval driver invokes only the harness runtime.

Each task image has one reusable V2 runtime and its `DEFAULT` endpoint.
Every rollout gets a fresh task session. The harness terminates that
session, and the driver terminates the harness session. Deployments remain
available for the next eval. Subsequent deployments reuse the runtime ARNs
recorded in the manifest; unchanged configuration does not create a new
runtime version.

## Task and verifier separation

Use pinned official Harbor bundles, with each task's original environment,
instruction, and verifier kept together. `prepare.py` copies only
`environment/` into the Docker build context and adds the ARM64 sandboxd
binary. Tests are packaged separately under
`s3://YOUR_VERIFIER_BUCKET/tmax/verifiers/`.

Only after the model loop ends does the harness fetch tests from S3,
transfer them into the existing task session, and run the original
`/tests/test.sh`. Tests and verifier outputs never enter the model
conversation. This prevents early verifier disclosure; it is not
intended to resist an agent deliberately tampering with the grader.

Both harness and task runtimes use VPC networking with separate security
groups. On the cluster security group, allow all TCP ports (0–65535) from
the harness security group so training gateways can choose their listening
port. Do not grant that access to
the task security group or attach the cluster's internal-traffic security
group to task runtimes. Sandbox execution isolation does not prevent
network access to the cluster.

Task runtimes use the subnet's NAT route for package downloads.
Their security group needs no
inbound rules: invocations go through the AgentCore API, which forwards
them to sandboxd on container port 8080. Use separate execution roles:

| Role | Required access |
|---|---|
| Both | Pull from this ECR repository; write their AgentCore logs |
| Harness | Invoke/stop this example's task runtimes; `GetObject` on `tmax/verifiers/*` |
| Task | No S3 read or runtime invocation permissions |

Pass the current model or gateway address in `base_url` with each invocation;
it is not part of the runtime deployment. For nodes using this cluster
security group, changing the address or TCP port requires no security-group
update or runtime redeployment. The ingress rule refers to the harness
security group, not a node IP.
The subnet also needs an outbound route to AgentCore APIs. AgentCore manages
the VPC interfaces through its `AWSServiceRoleForBedrockAgentCoreNetwork`
service-linked role; the task execution role does not need permission to
manage EC2 network interfaces.

## Run

All host scratch files and environments go under `$TMPDIR`.
From the repository root:

```bash
export UV_CACHE_DIR="$TMPDIR/uv-cache"
export UV_PYTHON_INSTALL_DIR="$TMPDIR/uv-python"
export UV_PROJECT_ENVIRONMENT="$TMPDIR/tmax-eval-venv"
export RUFF_CACHE_DIR="$TMPDIR/ruff-cache"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
uv sync --project examples/tmax_agent --group dev

uv run --no-project --python 3.12 --with harbor==0.23.0 \
  python examples/tmax_agent/download.py \
  --subset /path/to/tasks.json --output "$TMPDIR/tmax-data"
```

Pass a task selection JSON via `--subset`. Its `tasks` entries specify
`package.org.name`, `package.name`, and `content_hash` to pin each Harbor
task version.

Provide a model endpoint that supports Chat Completions tool calls and is
reachable from the harness runtime.

Create one ECR repository, such as `tmax`, with immutable tags.
Fill in execution roles and VPC details in a scratch copy of
`config.example.toml`. Authenticate Docker to that repository using a
`DOCKER_CONFIG` directory under `$TMPDIR`. The host must support ARM64
Docker builds (a native builder or registered QEMU).
Run deployment with an AWS profile authorized to pass the two execution
roles (`iam:PassRole`); permission to invoke runtimes alone is insufficient.

```bash
mkdir -p "$TMPDIR/tmax-live"
cp examples/tmax_agent/config.example.toml "$TMPDIR/tmax-live/config.toml"
# Fill in the account, roles, subnet and security group before continuing.
export DOCKER_CONFIG="$TMPDIR/tmax-live/docker"
export GOCACHE="$TMPDIR/go-build"

"$UV_PROJECT_ENVIRONMENT/bin/python" examples/tmax_agent/prepare.py \
  --config "$TMPDIR/tmax-live/config.toml" \
  --output examples/tmax_agent/manifest.json \
  "$TMPDIR/tmax-data/TASK_NAME"

"$UV_PROJECT_ENVIRONMENT/bin/python" examples/tmax_agent/deploy.py \
  --config "$TMPDIR/tmax-live/config.toml" \
  --manifest examples/tmax_agent/manifest.json

"$UV_PROJECT_ENVIRONMENT/bin/python" examples/tmax_agent/evaluate.py \
  --manifest examples/tmax_agent/manifest.json \
  --base-url http://MODEL_HOST:8000/v1 \
  --model-id YOUR_MODEL_ID \
  --output "$TMPDIR/tmax-live/results.json" --concurrency 2
```

Pass the downloaded task directories to `prepare.py`. The manifest records
source hashes, image tags, verifier keys and runtime ARNs. Existing image
tags are reused.

The harness retains TMax's 64-step budget, 16,384 maximum generated tokens
per turn, temperature 0.7, top-p 0.95, and the original task timeouts.
See [NOTICE.md](NOTICE.md) for upstream attribution and the small protocol
adaptations.

Rewards, trajectories, timing, token usage and verifier stdout are returned
through `AgentCoreRuntimeApp` HTTP start/get and saved by the eval driver
to its local `--output` JSON file. S3 is used only to deliver verifier bundles.
The driver reports infrastructure errors separately from task reward 0.
After an agent timeout, the verifier grades the work completed so far;
the result retains the timeout and partial trajectory.

## Checks and cleanup

```bash
"$UV_PROJECT_ENVIRONMENT/bin/ruff" check examples/tmax_agent
```

Live eval is required to validate VPC routing, ARM64 task compatibility
and V2 readiness.

To remove cloud deployments, delete only the runtime IDs recorded in the manifest using
`aws bedrock-agentcore-control delete-agent-runtime --region us-east-1
--agent-runtime-id ID`. ECR images and S3 verifier bundles are retained for
reproducibility. Remove this example's roles and dedicated security groups
only after its runtimes have been deleted. The cluster security group is
shared with other workloads.
