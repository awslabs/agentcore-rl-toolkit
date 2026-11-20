# vLLM-Neuron sampling server

This server implements the [external sampling contract](../README.md#sampling-server-contract)
on vLLM-Neuron (AWS Trainium) while SkyRL trains LoRA on GPU. The
[external sampling guide](../README.md) covers the design, versions, trainer launch,
network and permissions; this page covers the Neuron profiles, launch and refresh
implementation. The Neuron host needs no SkyRL install.

## Serving profiles

The shipped profile, Qwen3.6-27B on `trn2.48xlarge`, is an example: add a profile for
each model and instance type you serve, and a model module for a new architecture
(see [Resident checkpoint refresh](#resident-checkpoint-refresh)).

Serving profiles live under `config/<instance-type>/<model>.yaml`, with that profile's
Neuron runtime environment in `<model>.env` beside it, because
parallelism, device selection and compiler settings depend on the exact instance.
[`trn2.48xlarge/qwen3.6-27b.yaml`](config/trn2.48xlarge/qwen3.6-27b.yaml) serves
Qwen3.6-27B. Profiles contain inference parallelism, context limits, cache, sampling
and compiler settings for the model, serving workload and instance type. They are
passed to vLLM's `--config`, which accepts only vLLM settings, so the instance type is
recorded in the path and each file's header rather than as a key.

| Instance type | Model | Parallelism | Max sequences per replica | Context | Prefix caching |
|---|---|---|---|---|---|
| `trn2.48xlarge` | Qwen3.6-27B | TP8 / DP8, async scheduling, in-graph NKI sampling | 8 (64 total) | 65,536 | Enabled |

The profile uses all 64 logical cores (`NEURON_VISIBLE_DEVICES=0-63`) with
`NEURON_PLATFORM_TARGET_OVERRIDE=trn2` and `NEURON_LOGICAL_NC_CONFIG=2`. Each
replica is TP8, so a smaller allocation can lower `data-parallel-size` and
`data-parallel-size-local` with `NEURON_VISIBLE_DEVICES` covering TP × DP cores; the
compiled graphs are per replica and are reused. It requires a vLLM-Neuron build with
Qwen3.6 support (validated: `qwen36-trn2` at `76ca872`, vLLM 0.24.0, NeuronX Compiler
2.28.7631.0). In-graph sampling rejects top-k and top-p filtering, including the
model's `generation_config.json` defaults, so requests must send `top_k: -1` and
`top_p: 1.0`; SkyRL's external client does.

The first start compiles the graphs (about 18 minutes at DP1). Compile with one replica
(`NEURON_VISIBLE_DEVICES=0-7`, extra arguments `--data-parallel-size 1
--data-parallel-size-local 1`) and then start DP8 from the cache: eight replicas
compiling at once exceeded a 32,768-task systemd limit. Keep the compile cache.

Select the profile matching the actual checkpoint. Changing `--model` or
`ART_CHECKPOINT_DIR` does not select a different YAML automatically. The startup
command below selects it with `instance_type` and `profile`; other models and instance
types need their own validated serving profile. Deploy the whole folder: `serve.sh`,
`server.py`, its `model_*.py` modules and `config/`.

The server is stock `vllm serve` with this folder's plugins. Pass
`--config /path/to/serving/vllm_neuron/config/trn2.48xlarge/qwen3.6-27b.yaml`, the local
model checkpoint and `--served-model-name`; explicit CLI arguments override YAML settings.

## Launch the server

Start the trainer as described in [Launch the trainer](../README.md#launch-the-trainer).
Then, on the Neuron host (a `trn2.48xlarge` for the example profile), inside the Neuron
environment with that vLLM-Neuron build, run [`serve.sh`](serve.sh):

```bash
export ART_CHECKPOINT_DIR=/models/Qwen3.6-27B                    # immutable local base
export ART_WEIGHT_REFRESH_ROOT=/models/merged-cache              # existing dir, local NVMe
export ART_WEIGHT_REFRESH_CHECKPOINTS=s3://BUCKET/skyrl-checkpoints   # trainer's SKYRL_CHECKPOINTS_BASE
export ART_WEIGHT_REFRESH_S3_REGION=REGION
export VLLM_NEURON_BIND_HOST=PRIVATE_IP VLLM_API_KEY=SHARED_KEY   # or omit both for loopback
bash "$ART_REPO/src/agentcore_rl_toolkit/backends/tinker_api/skyrl/serving/vllm_neuron/serve.sh"
```

The script starts stock `vllm serve` with this folder on `PYTHONPATH`,
`--middleware server.ModelResolver` and `--worker-extension-cls server.WeightRefreshWorker`,
the profile selected by `ART_INSTANCE_TYPE` (default `trn2.48xlarge`) and
`ART_SERVING_PROFILE` (default `qwen3.6-27b`), and that profile's `.env` settings; values
already in the environment take precedence. It serves on port 8000 as model `initial`
(`VLLM_NEURON_PORT`, `ART_MODEL_VERSION`) and passes extra arguments to `vllm serve`. It
binds to `VLLM_NEURON_BIND_HOST`, default `127.0.0.1`; any other address requires
`VLLM_API_KEY`, which vLLM and ART's middleware both enforce, and which the trainer sends
as `SKYRL_EXTERNAL_INFERENCE_API_KEY` (see
[Network and permissions](../README.md#network-and-permissions)). Missing or invalid
settings stop the launch.

The compile cache defaults to `~/.cache/art-inference/<instance-type>/<profile>`; set
`NEURON_LIBTORCH_CACHE_ROOT` to reuse an existing one, and keep it stable across restarts.
With a warm compile cache, startup took about 10 minutes. Each refresh writes a merged
checkpoint of about 51 GB under `ART_WEIGHT_REFRESH_ROOT`.

## Resident checkpoint refresh

The opt-in text-serving extension retains Neuron workers and compiled graphs while
copying new merged checkpoint weights into the existing device tensors. It is enabled
by the two plugins and the settings above. The initial start still performs normal
compilation/cache loading and warmup.

| File | Contents |
|---|---|
| [`server.py`](server.py) | Model-independent plugins: settings, model-name resolution, single-flight refresh, S3 download, merge orchestration and validation, and the vLLM worker extension that stages and copies weights. Plugs into vLLM through its `--middleware` and `--worker-extension-cls` options, without patching vLLM |
| [`model_qwen3_5.py`](model_qwen3_5.py) | The Qwen3.5 architecture (`model_type: qwen3_5`, which Qwen3.6 uses): LoRA merge on CPU (keeping the vision weights of the original checkpoint) and the vLLM-Neuron model class |

`server.py` selects the model module by the base checkpoint's `model_type` and rejects
unsupported types at startup. To add a model, add a `model_<type>.py` providing
`MODEL_TYPE`, `load_merged(base_path, adapter_path)` and `neuron_model_class()`, and
register it in `MODEL_SUPPORT`.

A sampling request naming a new version starts one refresh. The server downloads
SkyRL's sampler archive, checks the adapter files and merges them on CPU into the
original base while the old version remains available. Every update is merged
with that same base, never with earlier merged weights. The server then blocks new
requests, drains existing responses, clears caches, stages every worker's shard,
copies and reads back every parameter, and advertises the new version only after
all replicas acknowledge. Waiting requests for that version then proceed. The
previous merged checkpoint under the refresh root is deleted after the switch;
the immutable base is never deleted. Each refresh logs a `weight_refresh` JSON line
with preparation and refresh timings.

Preparation failures leave the old version available, and a later request retries.
Failures after blocking admission leave serving unavailable (HTTP 503) until
restart. Architecture, tokenizer, and serving-shape changes require restart.
Install PEFT 0.18.1 and Accelerate 1.15.0 alongside the compatible
Neuron/Transformers environment (validated with Transformers 5.12.1) for CPU merging.

The worker implementation is limited to text serving without a vision encoder or
speculative decoder, for the registered model types. Resolver, controller and integrity tests live in
repository-root `tests/vllm_neuron/test_weight_refresh.py`.

Validated on `trn2.48xlarge` with Qwen3.6-27B in OfficeBench training with SkyRL FSDP
LoRA on GPU. With the sync profile (`27ba2dd`; TP8 × DP8, then DP4), two updates per run:
refreshes took about 100 s and rollout and training logprobs differed by about 0.004.
With this profile (`76ca872`, async, in-graph sampling; TP8 × DP8), four steps and three
updates: refreshes took 100–110 s with every worker acknowledging, inter-token latency
averaged 43 ms (about 70 ms before), and generated-token logprobs differed from the
trainer's by 0.006 on average (p99 0.14), within the attention q/k/v packing's numerical
noise.
