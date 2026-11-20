# Trn/GPU Hybrid Training

This example trains the [OfficeBench agent](../strands_officebench_agent/README.md) with SkyRL on GPU
while rollouts sample on Trn2. The agent itself is the original example's
[`rl_app.py`](../strands_officebench_agent/rl_app.py) (`AgentCoreRuntimeApp`, configured through
`_config`); this folder only adds the training configuration. Trn2 and the model in its
config are this example's choices; the [hybrid training section](../../src/agentcore_rl_toolkit/backends/tinker_api/skyrl/README.md#hybrid-training-gpu-trainer-with-separate-sampling-hardware)
covers other sampling hardware.

## Deploy the agent

Deploy the original example's runtime as described in its
[Deploy](../strands_officebench_agent/README.md#2-deploy) section.
The original example also provides task preprocessing, benchmark and local evaluation tools, and
documents known task-data issues.

Configure the Runtime's VPC and security group so it can reach the ART gateway. Its execution role
needs read access to the task/testbed S3 objects. Task instructions drive shell and office tools, so
use task/testbed objects in an S3 prefix you control. Set the deployed Runtime ARN in the training config.

## Prepare tasks

Use the original preprocessor with a local OfficeBench checkout:

```bash
python examples/strands_officebench_agent/preprocess.py \
  --officebench_dir /path/to/OfficeBench \
  --s3_bucket your-bucket \
  --s3_prefix officebench
```

Then create a JSONL dataset with one flat payload per line:

```json
{"task_uri":"s3://YOUR_BUCKET/officebench/1-2/0/config.json","testbed_uri":"s3://YOUR_BUCKET/officebench/1-2/testbed.tar.gz"}
{"task_uri":"s3://YOUR_BUCKET/officebench/1-6/0/config.json","testbed_uri":"s3://YOUR_BUCKET/officebench/1-6/testbed.tar.gz"}
{"task_uri":"s3://YOUR_BUCKET/officebench/1-16/0/config.json","testbed_uri":"s3://YOUR_BUCKET/officebench/1-16/testbed.tar.gz"}
{"task_uri":"s3://YOUR_BUCKET/officebench/1-19/0/config.json","testbed_uri":"s3://YOUR_BUCKET/officebench/1-19/testbed.tar.gz"}
```

## Train with SkyRL and Trn2

Start the SkyRL trainer as described in the
[external sampling guide](../../src/agentcore_rl_toolkit/backends/tinker_api/skyrl/serving/README.md#launch-the-trainer)
and the vLLM-Neuron server on Trn2 as described in its
[launch instructions](../../src/agentcore_rl_toolkit/backends/tinker_api/skyrl/serving/vllm_neuron/README.md#launch-the-server).
Run the training loop in an environment with ART's Tinker API dependencies
(see the [Tinker API recipe](../../src/agentcore_rl_toolkit/backends/tinker_api/README.md)),
separately from the Runtime's agent dependencies.

Copy `config.tinker.json.example` and set the runtime ARN, model paths, dataset and addresses.
`endpoint` points to the SkyRL API, which forwards sampling to Trn2. `base_model` is resolved by the
training host and `tokenizer` by the ART client. Bind the gateway to a private address reachable from
the Runtime's VPC, and allow its security group to reach that port.

Size the trainer for the example's 16,384-token context: set `trainer.max_prompt_length` to `16384`
and `generator.sampling_params.max_generate_length` to `512` in the JSON selected by
`SKYRL_BACKEND_CONFIG`. The [Qwen3.6-27B `trn2.48xlarge` profile](../../src/agentcore_rl_toolkit/backends/tinker_api/skyrl/serving/vllm_neuron/config/trn2.48xlarge/qwen3.6-27b.yaml)
serves up to 65,536 tokens and 64 concurrent sequences. Then run, from this directory with the
configured `officebench.jsonl` and `config.tinker.json`:

```bash
python -m agentcore_rl_toolkit.backends.tinker_api.train config.tinker.json
```

The example runs four tasks with four samples each in one batch, producing 16 concurrent rollouts.
`tps_limit`
controls new Runtime sessions per second. `rollout_timeout` bounds each agent execution, including
the first model call's wait for a new checkpoint to load on Trn2.

The four-row config is a smoke run, not a benchmark evaluation. All group members receive the same
original task and its original reward checks. A group with no reward variation produces no optimizer
update; inspect `train/optimizer_steps` in `metrics.jsonl` and the recorded rollouts when assessing
whether training occurred. Context exhaustion or invocation failure is recorded by the training loop.
