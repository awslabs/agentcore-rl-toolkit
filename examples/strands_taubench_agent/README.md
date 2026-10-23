# Strands Tau-Bench Agent

RL-adapted [tau2-bench](https://github.com/sierra-research/tau2-bench) agent for
training with Bedrock AgentCore Runtime (ACR). Supports the **airline**,
**retail**, and **telecom** tau-bench domains from a single deployment.

Each rollout is a multi-turn conversation between two models:

- **Assistant** — the vLLM-served model being trained (a customer-service agent
  that follows the domain policy and calls tau-bench tools).
- **User simulator** — a Bedrock Claude model that plays the customer, driven by
  the task's scenario.

Tool calls execute against a fresh tau-bench environment (DB + tools); telecom
additionally exposes user-side tools (device actions). The reward follows
tau-bench's `reward_basis` (DB / COMMUNICATE / ENV_ASSERTION / ACTION) and is
computed deterministically at the end of the rollout.

## Files

| File | Purpose |
|------|---------|
| `rl_app.py` | Multi-turn agent and user simulator |
| `reward.py` | `TauBenchReward` — DB / COMMUNICATE / ENV_ASSERTION / ACTION axes, dispatched per task by `reward_basis` |
| `utils.py` | Strands tool wrapping, message-role conversion, thinking-token handling |
| `constants.py` | System prompts, user-model config, orchestrator config, `DOMAINS_USER_TOOLS` |
| `pyproject.toml` | Example dependencies (tau2-bench is installed separately via the Dockerfile) |
| `test_local.py` | Smoke test — runs one task against a local server or a deployed ACR agent |
| `tasks/` | Sample tau2-bench tasks, one per domain (airline / retail / telecom) |

## Prerequisites

- **AWS credentials** with access to Bedrock (user simulator) and AgentCore. Verify with `aws sts get-caller-identity`.
- A **vLLM server** serving the model being trained.

## Installation

```bash
# 1. Clone this repo to get the example
git clone https://github.com/awslabs/agentcore-rl-toolkit.git
cd agentcore-rl-toolkit/examples/strands_taubench_agent

# 2. Create and activate the example's environment
uv venv --python 3.13
source .venv/bin/activate

# 3. Install the example and its dependencies
uv pip install -e .
```

The [Dockerfile](Dockerfile) installs tau2-bench and its bundled datasets.

## Run RL App on ACR

The common RL-training setup serves the model being trained from a local vLLM
server and runs many parallel rollouts on ACR. The steps below deploy this
agent to ACR.

### 1. Deploy

Run from `examples/strands_taubench_agent`. Set the subnet and security group IDs for your
vLLM server:

```bash
uv tool install bedrock-agentcore-starter-toolkit

agentcore configure --entrypoint rl_app.py --name strands_taubench_agent_rl \
  --requirements-file pyproject.toml --deployment-type container --protocol HTTP \
  --vpc --subnets subnet-xxxxxxxxxxxxxxxxx --security-groups sg-xxxxxxxxxxxxxxxxx \
  --disable-memory --non-interactive
agentcore deploy --agent strands_taubench_agent_rl
```

Set `AGENT_RUNTIME_ARN` to the ARN printed by the deployment. For a publicly
reachable vLLM server, omit the VPC flags.
Run `agentcore deploy --agent strands_taubench_agent_rl` for subsequent deployments.

The user simulator uses Bedrock (`USER_MODEL_CONFIG` in `constants.py`), so the
runtime execution role needs `bedrock:InvokeModel` access to that model.

### 2. Set up the vLLM server

Serve the model being trained from a vLLM server reachable by ACR. For example:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --max-model-len 8192 --port 4000 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

The training engine (or your test) passes this server's address and model id to
the agent per-invocation via the `_config` payload:

```jsonc
"_config": {
  "base_url": "http://<vllm-host>:4000/v1",   // assistant inference endpoint
  "model_id": "Qwen/Qwen3-4B-Instruct-2507"   // must match --served-model-name
}
```

### 3. Test the agent

`test_local.py` runs a single tau-bench task end-to-end. Three sample tasks ship
in `tasks/` (one per domain) — each is a raw tau2-bench task with a single write
action, sampled from the stock tau2 DBs so they work out of the box. The domain
is read from the task file, so you only pass the task path and model config.

**Local server** — run `rl_app.py` directly, before (or instead of) deploying.
Because the agent runs in your own environment here, tau2-bench must be installed
locally. With the `.venv` from [Installation](#installation) active, clone
tau2-bench into this example folder and editable-install it at the pinned commit:

```bash
# from examples/strands_taubench_agent, with .venv active
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

**Deployed ACR agent** — run from the example directory with `--acr`. No local tau2-bench
needed here; the agent runs inside the container, which already has it:

```bash
python test_local.py --task tasks/retail_example.json --acr \
  --agent-arn "$AGENT_RUNTIME_ARN" \
  --base-url http://localhost:4000/v1 \
  --model-id Qwen/Qwen3-4B-Instruct-2507
```

The result contains the reward, conversation, and metadata. Swap `--task` to any of
`tasks/{airline,retail,telecom}_example.json` to exercise a different domain —
the same image serves all three.
