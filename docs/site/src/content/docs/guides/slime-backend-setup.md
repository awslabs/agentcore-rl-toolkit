---
title: slime training backend setup
description: Train an AgentCore Runtime-deployed agent with the slime training backend.
---

This doc describes how to train an AgentCore Runtime-deployed agent with the
[slime](https://github.com/THUDM/slime) training backend. The integration
plugs into slime's own `train.py` through two hook flags, so slime itself is
unmodified:

- `--custom-generate-function-path agentcore_rl_toolkit.backends.slime.integration.rollout.generate`
  runs each rollout on AgentCore Runtime and turns the captured trajectory
  into slime `Sample`s.
- `--custom-reward-post-process-path agentcore_rl_toolkit.backends.slime.integration.rewards.normalize_episode_rewards`
  applies GRPO group normalization that stays correct when one rollout
  forks into several trajectory leaves.

Each slime rollout process starts an in-repo rollout gateway in front of
slime's SGLang router. The agent's OpenAI-compatible client points at that
gateway, which records token IDs, logprobs, and loss masks for every model
turn; rewards come from the dict the agent returns.

## Prerequisites

- A GPU cluster with **CUDA 13** installed (the scripts default to
  `/usr/local/cuda-13.0`).
- Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).
- AWS credentials with permission to invoke and stop AgentCore Runtime
  sessions.
- An AgentCore Runtime deployment of an `AgentCoreRuntimeApp` agent, such as
  [`examples/strands_math_agent/rl_app.py`](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/examples/strands_math_agent/rl_app.py).
  The agent's model client must forward the `api_key` from the payload's
  `_config` field (`config.get("api_key") or "EMPTY"`), which carries the
  gateway session key, and return `{"reward": <float>}`. Save the resulting
  **runtime ARN**.
- Network routing from the AgentCore containers to the trainer node's
  gateway port.

## Installation

From the repo root, in an activated Python 3.12 environment:

```bash
uv venv --python 3.12 && source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0
bash src/agentcore_rl_toolkit/backends/slime/scripts/install_slime.sh
```

The script installs the toolkit with the `gateway` and `rollout` extras, the cu13
PyTorch stack (flash-attn, Transformer Engine, Apex, torch_memory_saver,
sglang), clones Megatron-LM and slime into the current
directory at pinned commits, and applies slime's official Megatron and
sglang patches.

## Configure

The hooks read their settings from a YAML file passed with
`--custom-config-path`; every key becomes an attribute on slime's `args`.
Start from the example:

```bash
cd src/agentcore_rl_toolkit/backends/slime/examples/math_agent
cp config.yaml.example config.yaml
```

```yaml
rollout_session_backend:
  backend: agentcore_rl_toolkit.rollout_session.agentcore_http_session.AgentCoreHttpSession
  agentcore_runtime_arn: "arn:aws:bedrock-agentcore:<region>:<account-id>:runtime/<runtime-name>-<id>"

gateway_port: 9090              # 0 = auto-assign
max_rollout_time: 900           # per-session rollout timeout (s)
tps_limit: 5                    # new rollout sessions per second
```

`rollout_session_backend` is required: `backend` is the import path of a
`RolloutSession` class, and the remaining keys are its constructor
arguments. `AgentCoreHttpSession` invokes through the process-wide
`bedrock-agentcore` client, which pools connections and retries throttled
requests. Optional keys:
`gateway_host` (defaults to the node's routable IP), `model_id` (defaults
to `--hf-checkpoint`).

## Prepare data

The training dataset is a JSONL file where each line is one rollout
request:

```json
{"prompt": "...", "metadata": {"payload": { /* whatever your agent expects */ }}}
```

- **`prompt`** — read by slime through `--input-key prompt` (length
  filtering, positional bookkeeping). The agent does not see it.
- **`metadata.payload`** — sent **verbatim** (plus the `_config` field) as
  the `payload` dict your `@app.entrypoint` function receives. Put every
  per-rollout field the agent needs here (user prompt, ground-truth answer,
  task IDs, etc.). A row without a `payload` dict, or whose payload already
  has a `_config` field, fails the rollout.

Example (GSM8K):

```json
{"prompt": "How many ...?", "metadata": {"payload": {"prompt": "How many ...?", "answer": "42"}}}
```

## Launch training

`examples/math_agent/train.sh` is a complete GSM8K GRPO launcher. It stops
stale SGLang/Ray processes, applies the CUDA 13 environment fixes, starts a
Ray head, and submits slime's `train.py` with the two hooks and your
config. Defaults target **8 × H100** (`NUM_GPUS=8`, tensor parallel 2, two
GPUs per rollout engine). It uses `--attention-backend fused` (cuDNN), which
supports slime's packed-sequence layout only on Hopper (sm90) and newer GPUs;
on Ampere, TE finds no usable attention backend and training fails at the
first log-prob computation.

```bash
SLIME_DIR=/path/to/slime \
MODEL_DIR=/path/to/Qwen3-0.6B \
TRAIN_DATA_PATH=/path/to/gsm8k_train.jsonl \
MODEL_TYPE=qwen3-0.6B \
bash train.sh
```

| Variable | Meaning |
|---|---|
| `SLIME_DIR` | slime checkout (the one `install_slime.sh` cloned) |
| `MODEL_DIR` | HF checkpoint, used for `--hf-checkpoint` and `--ref-load` |
| `TRAIN_DATA_PATH` | training JSONL |
| `VAL_DATA_PATH` | eval JSONL (defaults to `TRAIN_DATA_PATH`) |
| `MODEL_TYPE` | slime model-arch script name under `$SLIME_DIR/scripts/models/` |
| `CONFIG` | hook config YAML (defaults to `config.yaml` next to the script) |
| `CKPTS_DIR` | checkpoint directory; **cleared at start** |
| `NUM_GPUS` | GPUs on the node |

To adapt it to another agent, copy `train.sh`, change the slime
training arguments (batch sizes, response length, learning rate, wandb
project), and keep the three `--custom-*` flags.

## Behavior notes

- **Rewards are agent-owned.** The hook reads the scalar `reward` from the
  agent's returned dict. If the agent omits it, slime falls back to its own
  reward model hub; a non-numeric reward or a non-dict result raises.
- **Failed rollouts don't fail the step.** A timeout, invocation error, or
  agent exception keeps any captured turns with reward 0; a rollout with no
  captured turns becomes an aborted sample that slime removes from the
  batch. Each rollout's Runtime session is stopped when it ends.
- **Group normalization** runs only when `--advantage-estimator` is one of
  `grpo`, `gspo`, `cispo`, `reinforce_plus_plus_baseline` and reward
  normalization is enabled; otherwise rewards pass through unchanged.
