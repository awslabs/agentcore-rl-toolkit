# Tinker API backend

Train agents on Bedrock AgentCore Runtime using an existing Tinker-compatible
endpoint. The CPU client runs the training loop and a rollout gateway, which
uses the Hugging Face tokenizer and chat template to capture token IDs and
sampling logprobs. Model weights and GPU dependencies stay on the endpoint.
Endpoint provisioning and deployment are managed separately.

The loop uses LoRA, centered episode rewards (`r - mean(group)`), and Tinker
`importance_sampling` loss, following the
[tinker-cookbook RL recipe](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/rl_loop.py).
Each batch collects concurrent rollouts, performs one optimizer update when
there is a training signal, and synchronizes sampling weights before continuing.
There is no standard-deviation normalization or ratio clipping.

## Install

From a toolkit checkout:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[gateway,tinker_api]' 'transformers==5.12.1' datasets wandb
```

`datasets` is used to prepare the math example; `wandb` enables optional logging.

> **SkyRL:** Replace `.[gateway,tinker_api]` with `.[gateway,tinker_skyrl]` to use
> SDK 0.24.1. The tested SkyRL version does not return `sample_sequence_ids`,
> which SDK 0.30.1 requires. The two extras are mutually exclusive and share the
> same training loop.

## Connect an endpoint

For the official Tinker service, export the API key in the client's environment:

```bash
export TINKER_API_KEY='YOUR_TINKER_API_KEY'
```

Set these fields in the example config:

```json
{
  "endpoint": "https://tinker.thinkingmachines.dev/services/tinker-prod",
  "base_model": "Qwen/Qwen3.5-4B",
  "tokenizer": "Qwen/Qwen3.5-4B"
}
```

For a self-hosted service such as SkyRL, use its endpoint URL and model
identifier instead:

```json
{
  "endpoint": "http://GPU_NODE_PRIVATE_IP:18080",
  "base_model": "/models/Qwen3.5-4B",
  "tokenizer": "Qwen/Qwen3.5-4B"
}
```

Use the model identifier returned by `ServiceClient.get_server_capabilities()`.
A server-side model path does not need to exist on the client; the client
downloads only the matching tokenizer and chat template. When `TINKER_API_KEY`
is unset or empty, the client uses `tml-dummy` for unauthenticated endpoints.
Keep real keys out of the JSON config, which is logged to W&B.

The AgentCore runtime must reach the client's gateway through its VPC. The
client needs access to the Tinker endpoint and AWS APIs/S3. The endpoint must
support the client's Tinker SDK version, LoRA training, token sampling with
logprobs, sampler weight synchronization, and checkpoints.

## Run the math example

Deploy the [Strands math agent](../../../../examples/strands_math_agent/) first.
The [example config](examples/math_agent/config.json.example) runs full GSM8K
training with Qwen3.5-4B, thinking disabled, held-out evaluation, and periodic
checkpoints.

```bash
cd src/agentcore_rl_toolkit/backends/tinker_api/examples/math_agent
python prepare_data.py gsm8k-train.jsonl
python prepare_data.py gsm8k-test.jsonl --split test
cp config.json.example config.json
```

Fill in the endpoint, runtime ARN, S3 bucket, gateway address, dataset paths, and
output directory in `config.json`, then run:

```bash
python -m agentcore_rl_toolkit.backends.tinker_api.train config.json
```

Each dataset row is an agent invocation payload, for example
`{"prompt": "What is 7 + 5?", "answer": "12"}`. The agent returns a finite scalar
`rewards`. Failed rollouts with captured training tokens receive zero reward;
rollouts with no training tokens are excluded from scoring and training.
Groups with no reward variation are skipped.

The output directory contains rollout records, metrics, evaluation results
when enabled, and checkpoint paths on the endpoint. Set `wandb_project` to
publish metrics to W&B. Use `--log-level DEBUG` for detailed client logs.
