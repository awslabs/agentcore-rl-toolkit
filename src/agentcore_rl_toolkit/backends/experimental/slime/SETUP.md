# Experimental Slime Backend Setup Guide

How to train an ACR-deployed agent with the [slime](https://github.com/THUDM/slime)
training backend, using `agentcore_rl_toolkit.rollout_gateway`.

The guide contains:

- **[Part 1 — Slime environment](#part-1--slime-environment)**: get a working
  slime runtime plus this toolkit, via the bare-metal install script.
- **[Part 2 — Run training and evaluation](#part-2--run-training-and-evaluation)**:
  deploy the agent, prepare data, configure `config.yaml`, run `train.sh`, evaluate.
- **[Tested Versions](#tested-versions)** pins the exact environment this was
  validated against — check here if you want to reproduce our results.

---

## Prerequisites

- Hardware requirements: see
  [slime's README](https://github.com/THUDM/slime#installation) for tested GPU
  configurations per model size. The defaults in `train.sh` target a single
  8-GPU node.
- A GPU cluster with **CUDA 13** installed (`/usr/local/cuda-13.0` by default;
  export `CUDA_HOME` if yours differs).
- Python==3.12 and [`uv`](https://docs.astral.sh/uv/).
- AWS credentials with permission to invoke an ACR runtime and read/write an S3
  bucket (`aws sts get-caller-identity` works).
- An ACR deployment of your agent — `rl_app.py` configured and deployed per
  [`examples/strands_math_agent/README.md`](../../../../../examples/strands_math_agent/README.md).
- **Network**: the training node's gateway port (`gateway_port`, default 9090)
  and the SGLang router port must be reachable *from the ACR VPC* — the deployed
  agent dials back into the gateway on every LLM call. Loopback won't work.

---

## Part 1 — Slime environment - Bare-metal

There is one supported path: the bare-metal install script.

```bash
# From a clone of this repo, inside your activated python environment
cd /path/to/agentcore-rl-toolkit

# Install the toolkit with the rollout-gateway extras (aiohttp + transformers).
# NOTE: this backend does NOT need the [slime] extra — that one pulls
# rllm-model-gateway, which only the legacy backends/slime uses.
uv pip install -e ".[gateway]"

export CUDA_HOME=/usr/local/cuda-13.0
bash src/agentcore_rl_toolkit/backends/experimental/slime/scripts/install_slime.sh
```

Notes:

- Expect a long build (the flash-attn / TE / apex source compiles dominate).
- Point `SLIME_DIR` (step 2.4) at the `slime` directory the script cloned.

---

## Part 2 — Run training and evaluation

### 2.1 Deploy the agent to ACR

Follow the "Run RL App Hosted on ACR" section in
[`examples/strands_math_agent/README.md`](../../../../../examples/strands_math_agent/README.md)
— it covers the `agentcore configure` / `agentcore deploy` flow plus VPC and IAM
setup.

Save the resulting **runtime ARN** — they go into `config.yaml` in step 2.3.

### 2.2 Download model and data

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-0.6B', local_dir='/path/to/Qwen3-0.6B')
"

python -c "
from datasets import load_dataset
import json
ds = load_dataset('openai/gsm8k', 'main', split='train')
with open('/path/to/gsm8k_train.jsonl', 'w') as f:
    for i, row in enumerate(ds):
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

### 2.3 Configure deployment settings

ACR deployment pointers and toolkit tunables live in `config.yaml`, passed to
slime via `--custom-config-path`. Slime merges every key into its args namespace,
where the rollout function reads them (`SlimeArtConfig.from_args` in
`integration/rollout.py`; each field also honors an uppercase env-var override).

```bash
cd src/agentcore_rl_toolkit/backends/experimental/slime/examples/math_agent

cp config.yaml.example config.yaml
# Edit config.yaml:
#   agent_runtime_arn: "arn:aws:bedrock-agentcore:..."   (from step 2.1)
#   s3_bucket: "your-bucket-name"

cp .wandb.env.example .wandb.env     # optional; skip to disable wandb
# Edit .wandb.env:
#   WANDB_API_KEY="..."
#   WANDB_ENTITY="your-org"
```

| `config.yaml` key | Env override | Default | Meaning |
|---|---|---|---|
| `agent_runtime_arn` | `ACR_AGENT_RUNTIME_ARN` | *(required)* | ACR runtime to invoke |
| `s3_bucket` | `ACR_S3_BUCKET` | *(required)* | bucket the agent writes results to |
| `exp_id` | `EXP_ID` | `slime-training` | S3 key prefix for this experiment |
| `gateway_port` | `GATEWAY_PORT` | `9090` | in-process rollout gateway port (must be reachable from the ACR VPC) |
| `acr_timeout` | `ACR_TIMEOUT` | `900` | per-session ACR invocation timeout (s) |
| `model_id` | `MODEL_ID` | `default` | OpenAI model id served to the agent |
| `acr_tps_limit` | `ACR_TPS_LIMIT` | `25` | ACR invocation rate limit (paces session *starts*) |
| `max_concurrent` | `MAX_CONCURRENT` | `100` | max concurrent in-flight ACR sessions |
| `max_pool_connections` | `MAX_POOL_CONNECTIONS` | `10` | boto3 conn-pool size — caps *reused* connections, not concurrency. Below `max_concurrent` it only logs "Connection pool is full" warnings, which are not errors. |
| `reward_postprocessing` | `REWARD_POSTPROCESSING` | `grpo` | `grpo` (group-relative) or `identity` |

### 2.4 Run training

`train.sh` is the only entry point. First fill in the values in
`src/agentcore_rl_toolkit/backends/experimental/slime/examples/math_agent/train.sh`

```bash
cd /path/to/agentcore-rl-toolkit/

bash src/agentcore_rl_toolkit/backends/experimental/slime/examples/math_agent/train.sh
```

---

## Tested Versions

For reproducibility, here's the exact environment this integration was validated
against:

| Component | Version / SHA |
|---|---|
| Instance type | 8 × NVIDIA B200 180GB |
| CUDA | `13.0` (driver 580.159.04) |
| Python | `3.12` |
| PyTorch | `2.11.0+cu130` |
| slime | commit `fa3c990af6f18efd3fd9922698bf4bf4048d1263` |
| SGLang | `0.5.13` (sglang-kernel `0.4.3`, sgl-deep-gemm `0.1.2`, sglang-router `0.3.2`) |
| Megatron-LM | commit `1dcf0dafa884ad52ffb243625717a3471643e087` + slime's `megatron.patch` |
| Megatron-Bridge | `0.5.0+6fde1c85` |
| TransformerEngine | `2.11.0` (`core-cu13`) |
| flash-attn | `2.8.3` |
| Apex | `10417ace` |
| transformers | `5.8.1` |
| numpy | `1.26.4` (`<2`, required by Megatron) |
