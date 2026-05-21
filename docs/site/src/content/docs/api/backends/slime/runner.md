---
title: SlimeRunner
description: One Python entry point for slime-backed training.
sidebar:
  order: 3
---


_module: `agentcore_rl_toolkit.backends.slime.runner`_

SlimeRunner — one Python entry point for slime-backed training.

Users instantiate ``SlimeRunner`` with a handful of per-experiment fields
and call ``.train()``; the runner reproduces what ``train.sh`` does today
(stop stale processes, start a Ray head, source the slime model script,
submit the slime training job) via subprocess.

``train.sh`` stays in the repo as the low-level escape hatch; this class
is the primary entry point.

## `class SlimeRunner`

One Python entry point for slime-backed training.

**Constructor**

```python
SlimeRunner(
    # --- Required: per-experiment ---
    exp_id: str,
    agent_runtime_arn: str,
    s3_bucket: str,
    model_dir: str,
    data_path: str,
    model_type: str,

    # --- Optional: cluster ---
    num_gpus: int = 8,
    tp_size: int = 2,
    rollout_gpus_per_engine: int = 2,
    slime_dir: str = '/root/slime',
    megatron_dir: str = '/root/Megatron-LM',

    # --- Optional: ACR / toolkit (forwarded to slime via custom-config yaml) ---
    model_id: str = 'default',
    acr_timeout: int = 900,
    acr_tps_limit: int = 25,
    max_concurrent: int = 100,
    gateway_port: int = 9090,
    reward_postprocessing: str = 'grpo',

    # --- Optional: training hyperparameters ---
    rollout_batch_size: int = 32,
    n_samples_per_prompt: int = 8,
    rollout_max_response_len: int = 1024,
    rollout_temperature: float = 1.0,
    lr: float = 1e-06,
    eps_clip: float = 0.2,
    eps_clip_high: float = 0.28,
    weight_decay: float = 0.1,
    adam_beta2: float = 0.98,
    sglang_mem_fraction_static: float = 0.7,
    max_tokens_per_gpu: int = 9216,

    # --- Wandb (opt-in; no defaults injected if unset) ---
    wandb_project: str | None = None,
    wandb_group: str | None = None,

    # --- Escape hatch ---
    extra_flags: list[str] = list(),
)
```

### Methods

#### `from_yaml(path: str | os.PathLike) -> 'SlimeRunner'`

Load kwargs from a YAML file (convenience for config-file workflows).

#### `train(num_rollout: int = 1) -> None`

Run the training job. Blocks until the slime job exits.

Mirrors ``train.sh`` step-by-step: stop stale sglang/ray, start a Ray
head, source the slime model script, submit the slime training job
via ``ray job submit``. Streams stdout/stderr to the parent process.

### Attributes

- `acr_timeout` *(int)*

- `acr_tps_limit` *(int)*

- `adam_beta2` *(float)*

- `agent_runtime_arn` *(str)*

- `data_path` *(str)*

- `eps_clip` *(float)*

- `eps_clip_high` *(float)*

- `exp_id` *(str)*

- `extra_flags` *(list[str])*

- `gateway_port` *(int)*

- `lr` *(float)*

- `max_concurrent` *(int)*

- `max_tokens_per_gpu` *(int)*

- `megatron_dir` *(str)*

- `model_dir` *(str)*

- `model_id` *(str)*

- `model_type` *(str)*

- `n_samples_per_prompt` *(int)*

- `num_gpus` *(int)*

- `reward_postprocessing` *(str)*

- `rollout_batch_size` *(int)*

- `rollout_gpus_per_engine` *(int)*

- `rollout_max_response_len` *(int)*

- `rollout_temperature` *(float)*

- `s3_bucket` *(str)*

- `sglang_mem_fraction_static` *(float)*

- `slime_dir` *(str)*

- `tp_size` *(int)*

- `wandb_group` *(str | None)*

- `wandb_project` *(str | None)*

- `weight_decay` *(float)*
