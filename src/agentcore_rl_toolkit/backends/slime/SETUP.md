# Slime Backend Setup Guide

How to train an ACR-deployed agent with the [slime](https://github.com/THUDM/slime)
training backend. The guide splits concerns into three parts:

- **[Part 1 — Slime environment](#part-1--slime-environment)**:
  get a working slime runtime plus this toolkit, via either the
  official slime docker or the bare-metal install script.
- **[Part 2 — Run training and evaluation](#part-2--run-training-and-evaluation)**:
  deploy the agent, prepare data, run `train.sh` / `eval.sh`.
- **[Common Problems & Fixes](#common-problems--fixes)** covers known
  gotchas (Megatron-LM regression on 32B, norm-epsilon mismatch, etc.).
- **[Tested Versions](#tested-versions)** pins the exact environment
  this was validated against — check here if you want to reproduce our
  results.


## Prerequisites

- Hardware and CUDA requirements: see
  [slime's README](https://github.com/THUDM/slime#installation) and the
  [slime docker README](https://github.com/THUDM/slime/blob/main/docker/README.md)
  for tested GPU configurations per model size.
- A GPU cluster with **CUDA >= 12.9** installed.
- Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).
- AWS credentials with permission to invoke an ACR runtime and
  read/write an S3 bucket (`aws sts get-caller-identity` works).
- An ACR deployment of your agent — `rl_app.py` configured and deployed
  per
  [`examples/strands_math_agent/README.md`](../../../../examples/strands_math_agent/README.md).

---

## Part 1 — Slime environment

Choose **one** of the two paths below to install slime, then install the
toolkit into the same environment. Part 2 below runs inside this
environment.

### Option A: Official slime docker (recommended)

Follow [slime's own installation docs](https://github.com/THUDM/slime#installation)
and use the container image (`slimerl/slime:latest`). Inside the
container, slime and Megatron-LM ship pre-installed at `/root/slime` and
`/root/Megatron-LM` — use those paths for `SLIME_DIR` / `MEGATRON_DIR`
in step 2.4 (or `slime_dir` / `megatron_dir` on `SlimeRunner`).

Install the toolkit with the slime-backend extras inside the container
(the `[slime]` extra pulls in rllm-model-gateway, which the backend
imports):

```bash
# From a clone of this repo, inside the slime container
cd /path/to/agentcore-rl-toolkit
uv pip install -e ".[slime]"
```

We have only tested against official slime at commit
`fa3c990af6f18efd3fd9922698bf4bf4048d1263` (the commit pinned in
`scripts/install_slime.sh`).

### Option B: Bare-metal install script

Install slime and its heavyweight dependency stack (Megatron-LM,
Transformer Engine, Apex, flash-attn, sglang, torch_memory_saver) with
the provided script, which clones slime + Megatron-LM into the current
directory and applies slime's official patches. Run it inside your
activated python environment.

The script supports both CUDA 12 and CUDA 13. It defaults `CUDA_HOME`
to `/usr/local/cuda-13.0` (cu13) or `/usr/local/cuda-12.9` (cu12);
export `CUDA_HOME` yourself first if yours differs.

```bash
# From a clone of this repo, inside your activated python environment
cd /path/to/agentcore-rl-toolkit

# Install the toolkit with the slime-backend extras
uv pip install -e ".[slime]"

# For CUDA 13
export CUDA_HOME=/usr/local/cuda-13.0
bash src/agentcore_rl_toolkit/backends/slime/scripts/install_slime.sh cu13

# For CUDA 12
export CUDA_HOME=/usr/local/cuda-12.9
bash src/agentcore_rl_toolkit/backends/slime/scripts/install_slime.sh cu12
```

Point `SLIME_DIR` / `MEGATRON_DIR` (step 2.4) or `slime_dir` /
`megatron_dir` on `SlimeRunner` at the `slime` and `Megatron-LM`
directories the script cloned.

---

## Part 2 — Run training and evaluation

### 2.1 Deploy the agent to ACR

Follow the "Run RL App Hosted on ACR" section in
[`examples/strands_math_agent/README.md`](../../../../examples/strands_math_agent/README.md)
— it covers the `agentcore configure`/`agentcore deploy` flow plus VPC
and IAM setup.

Save the resulting **runtime ARN** and **S3 bucket name** — they go
into `config.yaml` in step 2.3.

### 2.2 Download model and data

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
        question = row['question']
        answer = row['answer'].split('####')[-1].strip()
        # Top-level 'prompt' is read by slime (tokenization, length filter).
        # 'metadata' is the agent payload verbatim — shape it however the agent expects.
        f.write(json.dumps({
            'prompt': question,
            'metadata': {'prompt': question, 'answer': answer},
        }) + '\n')
"
```

The agent-visible payload is exactly the contents of ``metadata``, so
different agents can use different payload shapes (e.g. ``{'task_id': ...}``
for AppWorld, ``{'repo_uri': ..., 'metadata_uri': ..., ...}`` for
migration) without any slime-side changes.

### 2.3 Configure deployment settings

```bash
cd src/agentcore_rl_toolkit/backends/slime/examples/math_agent

cp config.yaml.example config.yaml   # gitignored
# Edit config.yaml:
#   agent_runtime_arn: "arn:aws:bedrock-agentcore:..."   (from step 2.1)
#   s3_bucket: "your-bucket-name"
#   (other fields have sensible defaults — leave as-is)

cp .wandb.env.example .wandb.env     # optional; skip to disable wandb
# Edit .wandb.env:
#   WANDB_API_KEY="..."
#   WANDB_ENTITY="your-org"
```

`config.yaml` and `.wandb.env` are gitignored.

### 2.4 Run training

Two entry points, same job under the hood:

- **`SlimeRunner` (Python)** — recommended. Write a short `train.py`;
  the class handles Ray/SGLang plumbing and slime CLI flags.
- **`train.sh` (bash)** — escape hatch. Use when you need to override
  something the class doesn't surface, or to debug the raw slime CLI.

#### Python (`SlimeRunner`)

Defaults target **8 × H100** (`num_gpus=8`, `tp_size=2`,
`rollout_gpus_per_engine=2`). Override kwargs for other cluster sizes.
`.train(num_rollout=…)` defaults to 1 rollout for smoke testing — bump
to 100 for a real run.

`slime_dir` / `megatron_dir` default to the in-container paths
(`/root/slime` and `/root/Megatron-LM`); override for bare-metal
installs.

```python
# train.py — minimal 3B smoke test
from agentcore_rl_toolkit.backends.slime import SlimeRunner

SlimeRunner(
    exp_id="gsm8k-3b-smoke",
    agent_runtime_arn="arn:aws:bedrock-agentcore:...",
    s3_bucket="your-bucket",
    model_dir="/path/to/Qwen2.5-3B-Instruct",
    data_path="/path/to/gsm8k_tiny.jsonl",
    model_type="qwen2.5-3B",
).train(num_rollout=1)
```

```bash
cd /path/to/agentcore-rl-toolkit
python src/agentcore_rl_toolkit/backends/slime/examples/math_agent/train.py
```

32B on 8 GPUs:

```python
SlimeRunner(
    exp_id="gsm8k-32b-run",
    agent_runtime_arn="arn:aws:bedrock-agentcore:...",
    s3_bucket="your-bucket",
    model_dir="/path/to/Qwen2.5-32B-Instruct",
    data_path="/path/to/gsm8k_tiny.jsonl",
    model_type="qwen2.5-32B",
    tp_size=8,
    rollout_gpus_per_engine=8,
).train(num_rollout=5)
```

Any slime/Megatron-LM/SGLang CLI flag that isn't surfaced as a named
kwarg can be passed through `extra_flags`:

```python
SlimeRunner(..., extra_flags=["--num-epoch", "3"]).train(num_rollout=50)
```

If you prefer a YAML config, `SlimeRunner.from_yaml("config.yaml").train()`
accepts the same keys.

#### Bash (`train.sh`) — escape hatch

`train.sh` takes the same knobs via env vars. It's kept as the low-level
reference for what the Python class replicates, and as a debugging path
for slime flag experiments.

```bash
# 3B smoke test
export SLIME_DIR=/root/slime \
       MEGATRON_DIR=/root/Megatron-LM \
       MODEL_DIR=/path/to/Qwen2.5-3B-Instruct \
       DATA_PATH=/path/to/gsm8k_tiny.jsonl
bash src/agentcore_rl_toolkit/backends/slime/examples/math_agent/train.sh
```

For 32B on 8 GPUs, add `MODEL_TYPE=qwen2.5-32B TP_SIZE=8 ROLLOUT_GPUS_PER_ENGINE=8 NUM_ROLLOUT=5`.

### 2.5 Run evaluation

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

---

## Tested Versions

For reproducibility, here's the exact environment this integration was
validated against:

| Component | Version / SHA |
|---|---|
| Instance type | 8 × NVIDIA H100 80GB HBM3 |
| CUDA | `13.0` |
| PyTorch | `2.11.0+cu130` |
| slime | commit `fa3c990af6f18efd3fd9922698bf4bf4048d1263` |
| SGLang | `0.5.13` |
| Megatron-LM | commit `1dcf0dafa884ad52ffb243625717a3471643e087` ⚠️ see note |

> **Megatron-LM pin.** `install_slime.sh` pins Megatron-LM at `1dcf0dafa`
> (~500 commits ahead of slime's stable pin). This is fine for
> tied-embedding models (0.5B/3B/7B) but breaks 32B training — see
> [Common Problems & Fixes](#common-problems--fixes). For 32B, downgrade
> to `3714d81d` (slime's documented stable sha) via `git checkout`.
