# Slime Backend Setup Guide

How to train an ACR-deployed agent with the [slime](https://github.com/THUDM/slime)
training backend. The guide splits concerns into three parts:

- **[Part 1 — Slime environment](#part-1--slime-environment-upstream)**:
  get a working slime runtime. This is slime's own concern; we only
  describe what we expect at the end.
- **[Part 2 — Install ART + backend](#part-2--install-art--slime-backend)**:
  install this toolkit and apply the two SGLang patches we own.
- **[Part 3 — Run training and evaluation](#part-3--run-training-and-evaluation)**:
  deploy the agent, prepare data, run `train.sh` / `eval.sh`.
- **[Common Problems & Fixes](#common-problems--fixes)** covers known
  gotchas (Megatron-LM regression on 32B, norm-epsilon mismatch, etc.).


## Prerequisites

- Hardware and CUDA requirements: see
  [slime's README](https://github.com/THUDM/slime#installation) and the
  [slime docker README](https://github.com/THUDM/slime/blob/main/docker/README.md)
  for tested GPU configurations per model size.
- Python 3.10+ and [`uv`](https://docs.astral.sh/uv/).
- AWS credentials with permission to invoke an ACR runtime and
  read/write an S3 bucket (`aws sts get-caller-identity` works).
- An ACR deployment of your agent — `rl_app.py` configured and deployed
  per
  [`examples/strands_math_agent/README.md`](../../../../examples/strands_math_agent/README.md).

---

## Part 1 — Slime environment (upstream)

Follow [slime's own installation docs](https://github.com/THUDM/slime#installation)
— either the container path (`slimerl/slime:latest`, what we validated)
or a bare-metal install. Part 2 and Part 3 below run inside this
environment.

Inside the `slimerl/slime:latest` container, slime and Megatron-LM ship
pre-installed at `/root/slime` and `/root/Megatron-LM` — use those paths
for `SLIME_DIR` / `MEGATRON_DIR` in step 3.4. For a bare-metal install,
point at wherever you cloned slime + Megatron-LM.

---

## Part 2 — Install ART + slime backend

Inside the slime environment from Part 1:

```bash
# From a clone of this repo
cd /path/to/agentcore-rl-toolkit

# Install the toolkit plus the slime-backend extras
# (the `[slime]` extra pulls in rllm-model-gateway, which the backend imports)
uv pip install -e ".[slime]"
```

Then apply the SGLang `token_ids` patch — it adds prompt_token_ids /
token_ids fields to chat completion responses so the gateway can capture
RL training trace data without requiring `logprobs=True` on every
request. The patch is idempotent (second run is a no-op):

```bash
python -m agentcore_rl_toolkit.backends.slime.patches.sglang_token_ids

# Verify the patch round-trips under greedy decoding (any HF checkpoint
# works; Qwen2.5-0.5B-Instruct is the fastest to download + load)
python -m agentcore_rl_toolkit.backends.slime.patches.verify_sglang_token_ids \
    --model-path /path/to/Qwen2.5-0.5B-Instruct
# Expect: "OK: 4/4 checks passed"
```

The verification script launches its own SGLang server, runs four
greedy-decoded correctness checks (non-streaming basic, streaming basic,
cross-mode consistency, tool-call consistency), and tears the server
down.

> **Note on the container path:** the patch persists only within that
> container layer. Bake it into your image or re-apply on every
> container start.

---

## Part 3 — Run training and evaluation

### 3.1 Deploy the agent to ACR

Follow the "Run RL App Hosted on ACR" section in
[`examples/strands_math_agent/README.md`](../../../../examples/strands_math_agent/README.md)
— it covers the `agentcore configure`/`agentcore deploy` flow plus VPC
and IAM setup.

Save the resulting **runtime ARN** and **S3 bucket name** — they go
into `config.yaml` in step 3.3.

### 3.2 Download model and data

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-3B-Instruct', local_dir='/path/to/Qwen2.5-3B-Instruct')
"

python -c "
from datasets import load_dataset
import json
ds = load_dataset('openai/gsm8k', 'main', split='train')
with open('/path/to/gsm8k_tiny.jsonl', 'w') as f:
    for i, row in enumerate(ds):
        if i >= 64: break
        answer = row['answer'].split('####')[-1].strip()
        f.write(json.dumps({'prompt': row['question'], 'label': answer}) + '\n')
"
```

### 3.3 Configure deployment settings

```bash
cd src/agentcore_rl_toolkit/backends/slime/examples/math_agent

cp config.yaml.example config.yaml   # gitignored
# Edit config.yaml:
#   agent_runtime_arn: "arn:aws:bedrock-agentcore:..."   (from step 3.1)
#   s3_bucket: "your-bucket-name"
#   (other fields have sensible defaults — leave as-is)

cp .wandb.env.example .wandb.env     # optional; skip to disable wandb
# Edit .wandb.env:
#   WANDB_API_KEY="..."
#   WANDB_ENTITY="your-org"
```

`config.yaml` and `.wandb.env` are gitignored.

### 3.4 Run training

train.sh defaults target **8 × H100** (NUM_GPUS=8, TP_SIZE=2,
ROLLOUT_GPUS_PER_ENGINE=2). For smaller clusters override via env
(e.g. `NUM_GPUS=1 TP_SIZE=1 ROLLOUT_GPUS_PER_ENGINE=1` for a single
GPU). Defaults also set `NUM_ROLLOUT=1` for smoke testing — bump to
`NUM_ROLLOUT=100` (slime's production value) for a real run.

`SLIME_DIR` /
`MEGATRON_DIR` need to point at the slime + Megatron-LM source trees
(inside the `slimerl/slime:latest` container these are `/root/slime`
and `/root/Megatron-LM`).

```bash
cd /path/to/agentcore-rl-toolkit

# Qwen2.5-3B, 8 GPUs, 1 rollout (smoke test — train.sh defaults)
export SLIME_DIR=/root/slime \
       MEGATRON_DIR=/root/Megatron-LM \
       MODEL_DIR=/path/to/Qwen2.5-3B-Instruct \
       DATA_PATH=/path/to/gsm8k_tiny.jsonl
bash src/agentcore_rl_toolkit/backends/slime/examples/math_agent/train.sh
```

For 32B on 8 GPUs:

```bash
export MODEL_DIR=/path/to/Qwen2.5-32B-Instruct \
       MODEL_TYPE=qwen2.5-32B \
       TP_SIZE=8 \
       ROLLOUT_GPUS_PER_ENGINE=8 \
       NUM_ROLLOUT=5
bash src/agentcore_rl_toolkit/backends/slime/examples/math_agent/train.sh
```

### 3.5 Run evaluation

`eval.sh` launches SGLang directly against an HF checkpoint and runs
`examples/strands_math_agent/evaluate.py` against it + the deployed ACR
runtime.

```bash
MODEL_DIR=/path/to/Qwen2.5-3B-Instruct \
EVAL_LIMIT=100 \
bash src/agentcore_rl_toolkit/backends/slime/examples/math_agent/eval.sh
```

To evaluate a trained checkpoint (when training was run with `SAVE_HF=1`),
point `MODEL_DIR` at the HF export inside the save directory.

---

## Common Problems & Fixes

### `LinearCrossEntropyModule` parallelism error on 32B (or any model with untied embeddings)

**Symptom:** During 32B training, the Megatron actor crashes with:

```
ValueError: Cannot determine parallelism type for module 'LinearCrossEntropyModule'
            at weight 'output_layer.weight'.
```

**Cause:** The Megatron-LM bundled in `slimerl/slime:latest` is several
hundred commits ahead of the sha pinned in slime's docker README
(`3714d81d`). Specifically, Megatron PR **#3226 "Reapply fix Linear CE
Fusion"** (2026-02-04) replaced `ColumnParallelLinear` with a new
`LinearCrossEntropyModule` that megatron-bridge's `AutoMapping` doesn't
recognize. Models with tied embeddings (0.5B, 3B, 7B) skip this code
path; models with `--untie-embeddings-and-output-weights` (32B and up)
hit it.

**Fix:** Inside the container, pin `/root/Megatron-LM` to the stable sha:

```bash
cd /root/Megatron-LM
# Stash any image-local patches first (can be restored later with `git stash pop`)
git stash -u -m "slime local patches"
git checkout 3714d81d418c9f1bca4594fc35f9e8289f652862
# Clear pyc caches that reference the old code
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
```

### `--norm-epsilon` mismatch on Qwen2.5-32B-Instruct

**Cause:** slime's `scripts/models/qwen2.5-32B.sh` hardcodes
`--norm-epsilon 1e-5` (matching Qwen2.5-32B **base**), but
**Qwen2.5-32B-Instruct** uses `1e-6`.

**Fix:** Edit the slime model script to `--norm-epsilon 1e-6`, or add
`--norm-epsilon 1e-6` explicitly in `train.sh`'s `python3 train.py` CLI
args. 0.5B/3B/7B Instruct variants match their base-model norm epsilons,
so this only affects 32B.
