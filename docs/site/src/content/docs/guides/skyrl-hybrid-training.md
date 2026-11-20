---
title: SkyRL hybrid training
description: Train LoRA with SkyRL on GPU while rollouts sample on separate hardware, for example vLLM-Neuron on AWS Trainium.
---

Hybrid training splits SkyRL's [Tinker-compatible endpoint](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/src/agentcore_rl_toolkit/backends/tinker_api/skyrl/README.md)
across two kinds of hardware: SkyRL trains LoRA with FSDP on a GPU host, and a separate
server samples on other hardware. ART's
[Tinker API training loop](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/src/agentcore_rl_toolkit/backends/tinker_api/README.md)
is unchanged: its `endpoint` is SkyRL's API, and SkyRL forwards each `sample()` to the
sampling server.

The model, accelerators and instance types on this page are examples. Any model SkyRL
can train with LoRA works, with any sampling server that meets the
[sampling server contract](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/src/agentcore_rl_toolkit/backends/tinker_api/skyrl/serving/README.md#sampling-server-contract).
ART ships one: vLLM-Neuron for AWS Trainium, with an example profile for Qwen3.6-27B on
`trn2.48xlarge`.

![Hybrid training: AgentCore Runtime agents call the ART client, which samples and trains through SkyRL on the GPU host; SkyRL forwards samples to a sampling host, for example vLLM-Neuron on Trainium, and exports each LoRA checkpoint to S3, from which the sampler loads and merges it.](../../../assets/guides/skyrl-hybrid-training.svg)

## How one training step flows

1. Agents running on AgentCore Runtime call the model through the ART client's
   **rollout gateway**, which tokenizes each turn and records token-level trajectories.
2. The gateway samples, and the training loop trains (`forward_backward`, `optim_step`),
   through **SkyRL's Tinker API** on the GPU host, which trains LoRA with FSDP.
3. SkyRL forwards each sample to the **sampling server** as `/v1/completions` with
   `model=<model_id>_<ckpt>`, over a private network with a shared API key.
4. After each optimizer step, SkyRL exports the **LoRA adapter to S3** (41 MB in the
   Qwen3.6-27B example).
5. The next sample names the new checkpoint. The sampling server downloads it, merges it
   into its immutable base on CPU and refreshes its resident weights before replying;
   in the vLLM-Neuron server, compiled graphs and workers stay in place.

Only the adapter crosses between the GPU and sampling hosts, so they can be in different
regions. Each checkpoint name is immutable: a request naming a replaced version gets
HTTP 409 rather than newer weights.

## Components

| Host | Runs | Code |
|---|---|---|
| AgentCore Runtime | The agent, e.g. OfficeBench's `rl_app.py` | [`examples/strands_officebench_agent`](https://github.com/awslabs/agentcore-rl-toolkit/tree/main/examples/strands_officebench_agent) |
| ART client (CPU) | `tinker_api.train` and the rollout gateway | [`backends/tinker_api`](https://github.com/awslabs/agentcore-rl-toolkit/tree/main/src/agentcore_rl_toolkit/backends/tinker_api) |
| GPU host | Patched SkyRL, `--runtime-role trainer` with `--external-inference-url` | [`skyrl/serve.sh`](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/src/agentcore_rl_toolkit/backends/tinker_api/skyrl/serve.sh), [`skyrl/patches/`](https://github.com/awslabs/agentcore-rl-toolkit/tree/main/src/agentcore_rl_toolkit/backends/tinker_api/skyrl/patches) |
| Sampling host (example: `trn2.48xlarge`) | A sampling server; the vLLM-Neuron one is stock `vllm serve` with ART's middleware and worker extension | [`skyrl/serving/vllm_neuron`](https://github.com/awslabs/agentcore-rl-toolkit/tree/main/src/agentcore_rl_toolkit/backends/tinker_api/skyrl/serving/vllm_neuron) |

## Setup

| Step | Where | Guide |
|---|---|---|
| Check out SkyRL `a6c07705` and apply ART's two patches | GPU host | [SkyRL version and patches](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/src/agentcore_rl_toolkit/backends/tinker_api/skyrl/serving/README.md#skyrl-version-and-patches) |
| Grant S3 and AgentCore access and open the GPU host's route to the sampler | All hosts | [Network and permissions](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/src/agentcore_rl_toolkit/backends/tinker_api/skyrl/serving/README.md#network-and-permissions) |
| Start the trainer with `SKYRL_EXTERNAL_INFERENCE_URL` and a trainer-only FSDP config | GPU host | [Launch the trainer](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/src/agentcore_rl_toolkit/backends/tinker_api/skyrl/serving/README.md#launch-the-trainer) |
| Start the sampling server, e.g. `serving/vllm_neuron/serve.sh` with a profile for your model and instance | Sampling host | [vLLM-Neuron sampling server](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/src/agentcore_rl_toolkit/backends/tinker_api/skyrl/serving/vllm_neuron/README.md) |
| Run ART's training loop with `tinker>=0.25.0,<0.28.0` | Client host | [Tinker API training guide](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/src/agentcore_rl_toolkit/backends/tinker_api/README.md) |

For a complete run, follow the
[OfficeBench hybrid training example](https://github.com/awslabs/agentcore-rl-toolkit/blob/main/examples/trn_gpu_hybrid_training/README.md).

## Example run

One validated configuration, for reference: OfficeBench training of Qwen3.6-27B with
SkyRL FSDP LoRA on 8×A100 and vLLM-Neuron on `trn2.48xlarge` (TP8 × DP8, 64K context,
async scheduling with in-graph NKI sampling), four steps of 4 tasks × 8 rollouts.
Timings depend on the model size and hardware.

| Measurement | Result |
|---|---|
| Checkpoint refresh on the sampler (download, CPU merge, worker stage and apply) | 100–110 s |
| Inter-token latency (mean) | 43 ms |
| First reply after an optimizer step (export, refresh, first response) | About 164 s |
| Sampler export per update | About 53 s, 41 MB |
| Final LoRA training checkpoint | 105.7 MB |
| Rollout vs training logprob difference, generated tokens | About 0.006 |
