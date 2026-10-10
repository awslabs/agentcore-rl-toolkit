# Strands Tau-Bench Agent (modularized user simulator)

This is the modularized version of [`../strands_taubench_agent`](../strands_taubench_agent):
the same RL-adapted [tau2-bench](https://github.com/sierra-research/tau2-bench) agent for
training with Bedrock AgentCore Runtime (ACR), with the **user turn decoupled** from the
rollout into a reusable `UserSimulator` abstraction. Rollout behavior is unchanged; see
[Verification](#verification).

Each rollout is a multi-turn conversation between two models:

- **Assistant** — the vLLM-served model being trained (a customer-service agent
  that follows the domain policy and calls tau-bench tools).
- **User simulator** — a Bedrock Claude model that plays the customer, driven by
  the task's scenario. Here it sits behind the `UserSimulator` interface in
  `user_simulator.py`, so a training system (for example a multi-turn RL framework that
  owns the conversation loop) can drive it directly.

Tool calls execute against a fresh tau-bench environment (DB + tools); telecom
additionally exposes user-side tools (device actions). The reward follows
tau-bench's `reward_basis` (DB / COMMUNICATE / ENV_ASSERTION / ACTION) and is
computed deterministically at the end of the rollout.

## What changed from `strands_taubench_agent`

| Concern | Original (`rl_app.py`) | Here |
|---|---|---|
| User model construction, user system prompt, user tools, termination keywords, `meta.user_model` | Inline in `_setup_rollout` / `_run_conversation` | `TauBenchUserSimulator` in `user_simulator.py` |
| User-facing projection of the conversation | Via `_run_agent_turn(role="user")` | Inside `UserSimulator.step()` |
| One ReAct turn of a Strands agent | `rl_app._run_agent_turn` (appends to the history) | `utils.run_agent_turn` (returns the new messages; the caller appends) |
| Conversation loop | `_run_conversation` | Unchanged, but it only depends on the `UserSimulator` interface |

The interface is small and benchmark-agnostic:

```python
class UserSimulator(Protocol):
    def reset(self, task: dict, env: Any) -> None: ...          # bind one episode's task and env
    def step(self, conversation: list[dict]) -> UserTurnResult: ...  # next user turn; does not mutate
    def describe(self) -> dict: ...                               # saved as meta.user_model

@dataclass
class UserTurnResult:
    messages: list[dict]            # this turn's messages, tagged from="user"
    text: str                       # text of the turn's last message
    terminated: bool                # user ended the conversation
    termination_reason: str | None  # ###STOP### / ###TRANSFER### / ###OUT-OF-SCOPE###
    metadata: dict | None           # reserved (None today)
```

The environment keeps a single owner: `reset(task, env)` wraps the user tools around the
episode's `env`, so user and assistant tools act on the same state (telecom's
`sync_tools` depends on this).

## Architecture

```
Training Cluster (e.g. HyperPod)
├── Training Engine (veRL / GRPO)
├── vLLM Inference Server (serves the model being trained)
│
└──► ACR (Bedrock AgentCore Runtime)
     ├── Agent Session 1  ──► vLLM       (assistant inference)
     ├── Agent Session 2       + Bedrock (user simulator, via UserSimulator)
     └── ...                   + tau-bench env (tools + DB)
         │
         └──► S3 (rollout data + rewards)
```

The assistant model is reached over HTTP via `base_url` (passed per-invocation
in the `_rollout` payload). The user simulator calls Bedrock using the ACR
container's execution-role credentials — see [IAM permissions](#5-set-up-iam-permissions).

## Files

| File | Purpose |
|------|---------|
| `rl_app.py` | Entrypoint — multi-turn orchestrator with `@rollout_entrypoint` |
| `user_simulator.py` | `UserSimulator` interface, `UserTurnResult`, and `TauBenchUserSimulator` (Bedrock user model, user prompt, user-facing history projection, user tools, termination) |
| `reward.py` | `TauBenchReward` — DB / COMMUNICATE / ENV_ASSERTION / ACTION axes, dispatched per task by `reward_basis` |
| `utils.py` | Strands tool wrapping, one-turn agent runner, message-role conversion, thinking-token handling |
| `constants.py` | System prompts, user-model config, orchestrator config, `DOMAINS_USER_TOOLS` |
| `pyproject.toml` | Example dependencies (tau2-bench is installed separately via the Dockerfile) |
| `test_local.py` | Smoke test — runs one task against a local server or a deployed ACR agent |
| `tasks/` | Sample tau2-bench tasks, one per domain (airline / retail / telecom) |
| `tests/`, `conftest.py` | Golden parity test: 9 scripted rollouts must reproduce goldens recorded from the original code |

## Prerequisites

- **AWS credentials** with access to Bedrock (user simulator), S3 (rollout
  results), and AgentCore. Verify with `aws sts get-caller-identity`.
- A **vLLM server** serving the model being trained.
- An **S3 bucket** for rollout results.

Two separate packages are involved — don't confuse them:

| Package | What it gives you | Source |
|---------|-------------------|--------|
| `agentcore-rl-toolkit` (this repo) | The RL SDK (`AgentCoreRLApp`, `@rollout_entrypoint`) **and this example** | `git clone` (below) |
| `bedrock-agentcore-starter-toolkit` | AWS's `agentcore` CLI used to deploy to ACR | `pip install` (below) |

## Installation

```bash
# 1. Clone this repo — it provides both the RL SDK source and this example folder
git clone https://github.com/awslabs/agentcore-rl-toolkit.git
cd agentcore-rl-toolkit/examples/strands_taubench_agent_modularized

# 2. Create and activate the example's environment
uv venv --python 3.13
source .venv/bin/activate

# 3. Install dependencies into the venv:
#    a) the example's own deps (pyproject.toml)
#    b) agentcore-rl-toolkit from local source (so unreleased SDK changes are picked up)
#    c) the agentcore CLI (separate AWS tool, used in the deploy steps below)
uv pip install -e .
uv pip install -e ../../ --force-reinstall --no-deps
uv pip install bedrock-agentcore-starter-toolkit
```

> tau2-bench is **not** a pip dependency — it is not on PyPI and its data files
> require an editable install from the repo root, so it is not in `pyproject.toml`.
> It is installed inside the container instead (see [step 2](#2-customize-the-dockerfile)).

## Verification

This version was checked against the original `../strands_taubench_agent` (commit `90f9d25`)
before submission. The golden parity test ships in `tests/`; the rest of the verification
tooling and its data are kept outside this repository.

Run the parity test (no GPU, network, or AWS access needed; tau2-bench must be installed
locally, as in [step 7](#7-test-the-agent)):

```bash
# from examples/strands_taubench_agent_modularized, with .venv active
uv pip install pytest
python -m pytest tests/
```

Each scenario runs `invoke_agent` against the real tau2 environment, tools, and reward, with
both models replaced by scripted replies, and compares every model call's input, the
conversation, the reward, and the returned result against a golden in `tests/golden/`.
Re-record the goldens only for an intended behavior change, and review their diff:
`TAUBENCH_RECORD_GOLDEN=1 python -m pytest tests/`.

- **Code-level parity.** Nine scripted scenarios across airline, retail, and telecom run the
  real tau2 environment, tools, and reward, with both models replaced by recorded replies.
  Given the same replies, both versions send byte-identical inputs to both models and
  return identical results. Each of 8 planted user-side bugs was caught.
- **Live tau2-bench** (`base` split, 4 trials per task; `Qwen/Qwen3-8B` assistant on vLLM,
  Claude Opus 4.6 user simulator):

  | Domain | Original pass^1 | Modularized pass^1 | Difference (95% CI) |
  |---|---|---|---|
  | airline | 0.275 | 0.260 | −0.015 [−0.080, +0.050] |
  | retail, run 1 | 0.579 | 0.526 | −0.053 [−0.098, −0.008] |
  | retail, run 2 (replication) | 0.553 | 0.539 | −0.013 [−0.063, +0.037] |

  The first retail gap did not reproduce. Two runs of the unchanged original code differ by
  −0.026, so run-to-run noise is about ±0.03. Any difference is bounded to about ±0.06 pass^1.
- **Rollouts** (2,224 compared). Message structure is identical, the user simulator's replies
  are statistically indistinguishable, and the assistant's behavior and failure types match.

## Run RL App on ACR

The common RL-training setup serves the model being trained from a local vLLM
server and runs many parallel rollouts on ACR. The steps below deploy this
agent to ACR.

### 1. Configure the ACR agent

```bash
cd examples/strands_taubench_agent

# Network details from your vLLM instance (so ACR can reach the server inside the VPC)
SUBNET_ID="subnet-0123456789abcdefg"
SECURITY_GROUP_ID="sg-0123456789abcdefg"

agentcore configure --entrypoint rl_app.py \
  --name strands_taubench_agent_modularized_rl_test \
  --requirements-file pyproject.toml \
  --deployment-type container \
  --vpc --subnets $SUBNET_ID --security-groups $SECURITY_GROUP_ID \
  --disable-memory \
  --non-interactive
```

This writes config to `.bedrock_agentcore.yaml` and generates a Dockerfile at
`.bedrock_agentcore/strands_taubench_agent_modularized_rl_test/Dockerfile`.

> If your vLLM server is reachable at a public URL instead of inside a VPC, drop
> the `--vpc --subnets --security-groups` flags to deploy to the public network.

### 2. Customize the Dockerfile

The auto-generated Dockerfile installs this example's dependencies but **not
tau2-bench** (which isn't on PyPI). Add one section to clone and editable-install
tau2-bench at a pinned commit, before the `COPY . .` line:

```dockerfile
# --- tau2-bench: editable install at a pinned commit ---
# tau2-bench isn't on PyPI; its data/ dir must be reachable via the repo layout,
# so an editable install is required.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/sierra-research/tau2-bench.git /opt/tau2-bench && \
    cd /opt/tau2-bench && git checkout 70e700c && \
    uv pip install -e .
```

> **Pin tau2-bench to the commit your tasks were authored against.** The reward
> code is tightly coupled to a specific tau2 version: `_compute_db_reward`
> compares DB hashes, `_compute_env_assertion_reward` runs tau2 assertion
> functions, and tool schemas come from tau2's `openai_schema`. A different tau2
> version (e.g. a future tau-3) can silently change rewards with no error. Pin it.

The agent reads the stock tau2 DBs bundled at that commit. The reward stays
correct because `_compute_db_reward` builds both the gold and predicted
environments from the same `get_environment()` call, so both sides read the same
DB — the hash comparison is internally consistent.

### 3. Deploy to ACR

```bash
agentcore deploy --agent strands_taubench_agent_modularized_rl_test
```

This uploads the source, builds the (ARM64) image via CodeBuild, pushes it to
ECR, and registers the agent runtime.

Record two values from this step — both are in `.bedrock_agentcore.yaml` (and
printed in the deploy logs):

- **`agent_runtime_arn`** — the training engine needs it to invoke the agent.
- **`execution_role`** — the IAM role the agent runs as, e.g.
  `arn:aws:iam::123456789:role/AmazonBedrockAgentCoreSDKRuntime-us-west-2-abc123`.
  You grant it S3 + Bedrock permissions in [step 5](#5-set-up-iam-permissions).

### 4. Set up S3

```bash
aws s3 mb s3://agentcore-rl   # skip if the bucket already exists
```

Rollout results are written here as `s3://<bucket>/<exp_id>/<input_id>_<session_id>.json`.

### 5. Set up IAM permissions

Grant the ACR agent's execution role permission to write rollout results to S3:

```bash
# Find execution_role in .bedrock_agentcore.yaml, e.g.
# arn:aws:iam::123456789:role/AmazonBedrockAgentCoreSDKRuntime-us-west-2-abc123
# -> set ROLE_NAME to the part after "role/"
ROLE_NAME="YOUR_ROLE_NAME_HERE"

aws iam put-role-policy --role-name $ROLE_NAME \
  --policy-name TauBenchRLAccess \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": ["s3:PutObject", "s3:GetObject"],
        "Resource": "arn:aws:s3:::agentcore-rl/*"
      }
    ]
  }'
```

> The user simulator calls Bedrock (defaults to `global.anthropic.claude-opus-4-6-v1`;
> see `USER_MODEL_CONFIG` in `constants.py`). The auto-generated ACR execution
> role normally already grants Bedrock invoke access — if user turns fail with
> `AccessDeniedException`, add `bedrock:InvokeModel` permission to the role.

### 6. Set up the vLLM server

Serve the model being trained from a vLLM server reachable by ACR. For example:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --max-model-len 8192 --port 4000 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

The training engine (or your test) passes this server's address and model id to
the agent per-invocation via the `_rollout` payload:

```jsonc
"_rollout": {
  "base_url": "http://<vllm-host>:4000/v1",   // assistant inference endpoint
  "model_id": "Qwen/Qwen3-4B-Instruct-2507",  // must match --served-model-name
  "exp_id": "test",
  "s3_bucket": "agentcore-rl",
  "session_id": "session_123",
  "input_id": "task_123"
}
```

### 7. Test the agent

`test_local.py` runs a single tau-bench task end-to-end. Three sample tasks ship
in `tasks/` (one per domain) — each is a raw tau2-bench task with a single write
action, sampled from the stock tau2 DBs so they work out of the box. The domain
is read from the task file, so you only pass the task path and model config.

**Local server** — run `rl_app.py` directly, before (or instead of) deploying.
Because the agent runs in your own environment here, tau2-bench must be installed
locally. With the `.venv` from [Installation](#installation) active, clone
tau2-bench into this example folder and editable-install it at the pinned commit:

```bash
# from examples/strands_taubench_agent_modularized, with .venv active
git clone https://github.com/sierra-research/tau2-bench.git
cd tau2-bench && git checkout 70e700c && uv pip install -e .
cd ..
```

Then run the server in one terminal and the test in another:

```bash
python rl_app.py &        # serves on localhost:8080
python test_local.py --task tasks/airline_example.json \
  --base-url http://localhost:4000/v1 \
  --model-id Qwen/Qwen3-4B-Instruct-2507
```

**Deployed ACR agent** — add `--acr` and `--agent-name`. No local tau2-bench
needed here; the agent runs inside the container, which already has it:

```bash
python test_local.py --task tasks/retail_example.json --acr \
  --agent-name strands_taubench_agent_modularized_rl_test \
  --base-url http://localhost:4000/v1 \
  --model-id Qwen/Qwen3-4B-Instruct-2507 \
  --s3-bucket agentcore-rl
```

The result (rollout data, reward, conversation, metadata) is written to
`s3://<bucket>/test/task_123_session_123.json`. Swap `--task` to any of
`tasks/{airline,retail,telecom}_example.json` to exercise a different domain —
the same image serves all three.

### 8. Redeployment

After code changes, rebuild and re-register:

```bash
agentcore deploy --agent strands_taubench_agent_modularized_rl_test
```

The first invocation after a redeploy cold-starts (image pull + container init).
