---
title: SageMaker backend setup
description: Train an AgentCore Runtime-deployed agent with Amazon SageMaker Training Sessions — no local GPUs required.
---

This doc describes how to train an AgentCore Runtime-deployed agent with
**Amazon SageMaker Training Sessions**. Unlike the
[slime](/agentcore-rl-toolkit/guides/slime-backend-setup/) /
[verl](/agentcore-rl-toolkit/guides/verl-backend-setup/) backends, there is
**no GPU cluster required for the training**: SageMaker hosts the policy weights,
the sampler, and the optimizer behind an SDK, and the RL loop itself runs as a
plain Python process on your laptop or a small EC2 box.

The loop is implemented in-repo at
[`src/agentcore_rl_toolkit/backends/experimental/sagemaker/`](https://github.com/awslabs/agentcore-rl-toolkit/tree/main/src/agentcore_rl_toolkit/backends/experimental/sagemaker)
and is driven by a single YAML config:

| File | Role |
|---|---|
| `train_grpo.py` | The GRPO training loop: rollout → advantages → `forward_backward` → `optim_step` → rebind sampler. |
| `config.py` / `config.yaml.example` | Typed config dataclass and a template to copy. |
| `rollout.py` | One ACR rollout: create gateway session, invoke the agent, await the S3 result, drain trajectories. |
| `datum.py` | Converts a gateway `TraceRecord` (+ advantage) into a SageMaker training datum. |
| `prepare_datasets/` | Dataset preprocessing scripts (GSM8K). |

Token capture uses the same in-repo
[rollout gateway](https://github.com/awslabs/agentcore-rl-toolkit/tree/main/src/agentcore_rl_toolkit/rollout_gateway)
as the experimental verl backend. The SageMaker-specific seam is
`SageMakerSdkBackend`
([`rollout_gateway/sampling_backends/sagemaker_sdk.py`](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/src/agentcore_rl_toolkit/rollout_gateway/sampling_backends/sagemaker_sdk.py)),
a `token_ids -> token_ids + logprobs` sampling backend over the SageMaker
`SamplingClient`.

## Prerequisites

- Python 3.12 and [`uv`](https://docs.astral.sh/uv/). No local GPU required.
- AWS credentials with permission to invoke an AgentCore Runtime, read/write an
  S3 bucket, and create SageMaker training sessions.
- An AgentCore Runtime deployment of your agent — follow the
  [Prepare agent for RL](/agentcore-rl-toolkit/guides/agent-adaptation/) guide.
  Save the resulting **runtime ARN** (`agent_runtime_arn` below). Your
  `rl_app.py` must read `api_key` from the `_rollout` payload
  (`payload["_rollout"].get("api_key") or "EMPTY"`) — the gateway keys token
  capture off the api-key / Bearer slot.
- An S3 bucket for rollout result delivery (`s3_bucket` below).
- SageMaker resources: an execution **role ARN**, an S3 **output path** for
  checkpoints, a **model package group ARN** for saved states, and the **hub
  content ARN** of the base model you want to fine-tune.

## Installation

```bash
cd /path/to/agentcore-rl-toolkit
uv venv --python=3.12
source .venv/bin/activate

uv pip install sagemaker-train          # SageMaker Training Sessions SDK
uv pip install -e ".[gateway]"          # rollout gateway (aiohttp + transformers)
uv pip install transformers==5.12       # version the SDK's tokenizer expects
uv pip install wandb                    # optional, for training curves
```

## Prepare data

The dataset is a Parquet or JSONL file with a single **`payload`** column /
field. Each row's `payload` is forwarded **verbatim** as the dict your
`@rollout_entrypoint` function receives — no trainer-side tokenization, no
prompt column. Put every per-rollout field your agent needs (prompt,
ground-truth answer, task IDs, resource URIs) inside `payload`.

Preprocessing scripts live in
[`prepare_datasets/`](https://github.com/awslabs/agentcore-rl-toolkit/tree/main/src/agentcore_rl_toolkit/backends/experimental/sagemaker/prepare_datasets).

## Configuration

Copy the template and fill in the required ARNs:

```bash
cd src/agentcore_rl_toolkit/backends/experimental/sagemaker
cp config.yaml.example config.yaml
```

Required — training fails fast without these:

| Key | Purpose |
|---|---|
| `role_arn` | SageMaker execution role. |
| `base_model_arn` | Base model to fine-tune, e.g. `arn:aws:sagemaker:<region>:aws:hub-content/SageMakerPublicHub/Model/<model-name>`. Also passed to the agent as `model_id`. |
| `s3_output_path` | S3 prefix for model checkpoints. |
| `model_package_group_arn` | Model package group that saved states / weights land in. |
| `agent_runtime_arn` | Your deployed AgentCore Runtime agent. |
| `s3_bucket` | Bucket the agent writes rollout results (rewards) to. |
| `dataset_path` | Training Parquet/JSONL file. |

The rest, grouped as in `config.yaml.example`:

```yaml
region: "us-east-1"
exp_id: "my-experiment"      # S3 key prefix for rollout results

# --- LoRA ---
lora_rank: 32
lora_alpha: 64
resume_from_model_package: ""   # model package ARN to resume optimizer+adapter from

# --- Training ---
batch_size: 32               # prompts per global step
responses_per_prompt: 4      # GRPO group size (K)
learning_rate: 1.0e-5
max_context_tokens: 32768    # gateway-enforced context budget per session
max_new_tokens: 8192
temperature: 1.0
epochs: 1
loss: ppo                    # ppo | importance_sampling | cispo

# --- Dataset ---
max_prompts: 0               # 0 = use all

# --- Evaluation ---
eval_dataset_path: "/path/to/test.parquet"
eval_every: 10               # 0 = only at the end
eval_max_prompts: 0
eval_temperature: 0.0

# --- Gateway ---
gateway_port: 0              # 0 = OS-assigned

# --- Checkpointing ---
save_every: -1               # -1 = only at the end
save_weights_at_end: false   # also publish an inference-ready model package

# --- Rollout ---
max_rollout_time: 1800.0     # per-rollout deadline, seconds
acr_tps_limit: 5             # AgentCore invoke TPS ceiling (per-account limit is 25)

# --- Logging ---
wandb_project: ""            # empty = console only
wandb_run_name: ""           # defaults to the SageMaker training session name
rollout_log_path: ""         # JSONL of per-rollout diagnostics; auto-named if empty
```

## Launch training

```bash
cd src/agentcore_rl_toolkit/backends/experimental/sagemaker
python train_grpo.py --config config.yaml
```

## What the loop does

Per global step, `train_grpo.py`:

1. **Rollout** — pops `batch_size` payloads; for each, runs `responses_per_prompt`
   ACR rollouts concurrently. Each rollout gets a fresh UUID used as the ACR
   `runtimeSessionId`, the gateway capture session id, and the `api_key` — so one
   id ties the container, the trajectory, and the S3 result together.
2. **Capture** — the agent's OpenAI/Anthropic calls go to the gateway
   (`base_url`), which renders messages to tokens, samples through the SageMaker
   `SamplingClient`, and records token ids, logprobs, and a loss mask per turn.
3. **Reward** — the agent writes `{"rewards": ...}` to S3; the loop awaits it
   (`max_rollout_time`) and stamps it onto the trajectory. A timeout, an ACR
   error, a non-200 agent status, or a missing/non-numeric reward all score
   **0.0** and are logged rather than raised — one bad rollout never kills a run.
4. **Advantages** — rewards are centered and normalized within each GRPO group;
   every trajectory-tree leaf becomes one datum (multi-turn agents and
   sub-agent forks yield several datums per rollout). Datums with an empty or
   all-zero loss mask are dropped; a step with no valid datums is skipped.
5. **Update** — `forward_backward(datums, loss_fn=<loss>)` then
   `optim_step(AdamParams(learning_rate=...))`.
6. **Rebind** — `save_weights_and_get_sampling_client()` returns a *new* sampling
   client, which is swapped into `SageMakerSdkBackend` so the next step samples
   on-policy. In-flight generations finish against the old weights.

Evaluation runs every `eval_every` steps (and once at the end) over
`eval_dataset_path` at `eval_temperature`, logging `eval/accuracy`. Checkpoints:
`save_state()` every `save_every` steps and once at the end, returning a model
package ARN you can feed back as `resume_from_model_package`; set
`save_weights_at_end: true` to also publish an inference-ready model package.

Metrics go to the console and, if `wandb_project` is set, to Weights & Biases:
`sampler/mean_reward`, `sampler/num_datums`, `time/rollout`, `time/train`,
`eval/accuracy`, plus the SDK's own `train/*` metrics.

## Notes and current limits

- **Rewards are agent-side.** The agent returns `{"rewards": ...}` from its
  `@rollout_entrypoint`; there is no trainer-side reward model. A list is read as
  its last element.
- **LoRA only.** Sessions are created via `create_lora_training_client`
  (`lora_rank` / `lora_alpha`); full fine-tuning is not exposed here.
- **GRPO only.** `train_grpo.py` is the only algorithm implemented; `loss`
  selects the surrogate (`ppo`, `importance_sampling`, `cispo`). For a different
  algorithm, write your own loop using SageMaker training APIs.
  `trace_record_to_datum()`, and the gateway assembly are all reusable as-is.
- **Single process, one gateway.** The loop is a single asyncio process serving
  one gateway on a background thread — it scales by ACR concurrency, not by local
  hardware.
