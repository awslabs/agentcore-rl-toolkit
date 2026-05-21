---
title: rllm backend setup
description: Train an AgentCore Runtime-deployed agent with the rllm training backend.
---

This doc describes how to train an AgentCore Runtime-deployed agent with the
[rllm](https://github.com/rllm-org/rllm) training backend. rLLM natively supports two training backends: [verl](https://github.com/volcengine/verl) — self-hosted; you provide and manage the GPU cluster. [Tinker](https://tinker-docs.thinkingmachines.ai/tinker/) — managed training service; no cluster to operate.

## Prerequisites

- Python 3.10+ and [`uv`](https://docs.astral.sh/uv/).
- A clone of this repo (`agentcore-rl-toolkit`).
- AWS credentials with permission to invoke an AgentCore Runtime and
  read/write an S3 bucket.
- An AgentCore Runtime deployment of your agent — follow the
  [Prepare agent for RL](/agentcore-rl-toolkit/guides/agent-adaptation/)
  guide. Save the resulting **runtime ARN** — passed to rllm via
  `rllm.remote_runtime.agentcore.agent_runtime_arn` (see Step 3).
- An S3 bucket for rollout result delivery — passed to rllm via
  `rllm.remote_runtime.agentcore.s3_bucket` (see Step 3).

## Step 1: Install the toolkit and rllm

From a clone of this repo, install the toolkit plus the `rllm` extras
(this pulls in rllm and a training backend):

```bash
cd /path/to/agentcore-rl-toolkit
uv pip install -e ".[rllm]"
```

This installs `agentcore-rl-toolkit` (which provides ACR communication
and the rllm AgentCore integration) along with rllm itself. See
[rllm's installation guide](https://rllm-project.readthedocs.io/en/latest/getting-started/installation/#basic-installation)
for additional install instructions for different training backends —
[verl](https://github.com/volcengine/verl) or
[Tinker](https://tinker-docs.thinkingmachines.ai/tinker/) — and
follow its install instructions for any additional dependencies (GPU
drivers, Tinker credentials, etc.).

## Step 2: Prepare data

The training dataset is a JSONL file where each line is one rollout
request. Each row's fields are forwarded **directly** as the `payload`
dict your `@rollout_entrypoint` function receives — no wrapper or
envelope. Put every per-rollout config the agent needs at the top
level: prompt, ground-truth answer, task IDs, etc.

Example (GSM8K):

```json
{"idx": 0, "prompt": "How many ...?", "answer": "42", "data_source": "gsm8k"}
```

## Step 3: Run training

AgentCore-related parameters are passed to rllm as config overrides on
the training command line. The runtime ARN and S3 bucket from the
prereqs are **required** — they tell the backend which ACR deployment
to invoke and where to read rollout results from. The remaining
parameters enable the AgentCore backend and tune its rate / timeout
behavior:

```bash
rllm.remote_runtime.enabled=true \
rllm.remote_runtime.backend=agentcore \
rllm.remote_runtime.agentcore.agent_runtime_arn=arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/your-agent \
rllm.remote_runtime.agentcore.s3_bucket=your-s3-bucket \
rllm.remote_runtime.agentcore.tps_limit=25 \
rllm.remote_runtime.agentcore.session_timeout=300 \
```

- `rllm.remote_runtime.enabled=true` + `backend=agentcore` — enables AgentCore.
- `agent_runtime_arn` / `s3_bucket` — **required**, from the prereqs.
- `tps_limit=25` — default ACR rate limit; adjustable per AWS account.
- `session_timeout=300` — 5-minute session timeout; configure per use case.

With [Tinker](https://github.com/rllm-org/rllm/tree/main/rllm-tinker) backend
([`train_agentcore_math_tinker.sh`](https://github.com/rllm-org/rllm/blob/main/examples/agentcore_math/train_agentcore_math_tinker.sh)):

```bash
bash examples/agentcore_math/train_agentcore_math_tinker.sh
```

With [verl](https://github.com/volcengine/verl) backend
([`train_agentcore_math_verl.sh`](https://github.com/rllm-org/rllm/blob/main/examples/agentcore_math/train_agentcore_math_verl.sh)):

```bash
bash examples/agentcore_math/train_agentcore_math_verl.sh
```

## Training workflow

1. rLLM loads prompt batches and submits each as a separate agent session
2. AgentCore auto-scales containers running agent code
3. Model calls route through `model-gateway`, capturing token data
4. Agents compute rewards and return results to S3
5. rLLM retrieves rewards and combines with token data for policy updates

For known issues see
[rllm troubleshooting](/agentcore-rl-toolkit/troubleshooting/rllm/).
