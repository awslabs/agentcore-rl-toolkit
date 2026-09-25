---
title: "SlimeRunner: one Python entry point for slime-backed training"
description: "Wrap train.sh + config.yaml behind a single Python class so users write `SlimeRunner(...).train()` instead of editing 121 lines of bash + YAML."
---
# SlimeRunner: one Python entry point for slime-backed training

## Summary

Today a slime-backend training run requires editing two files:
- `train.sh` (121 lines of bash, Ray/SGLang plumbing, 50+ slime CLI flags)
- `config.yaml` (ACR ARN, bucket, sampling settings)

Plus setting 4-5 env vars. Users who want to vary the model or dataset
either copy-edit train.sh or export environment variables for every
knob. This plan replaces the train.sh+config.yaml+env-vars combo with
a single Python class, `SlimeRunner`, whose constructor takes the
handful of fields that actually change per experiment and hides the
rest behind defaults. Under the hood it shells out to `ray job submit`
the same way `train.sh` does today.

**This is an additive convenience layer.** `train.sh` stays in the
repo (unchanged) as the low-level escape hatch for users who need to
customize what the class hides.

### Before / after

**Before (user edits train.sh + config.yaml, runs bash):**
```bash
# config.yaml: edit 8 fields
# train.sh: set env vars or edit defaults
export SLIME_DIR=/root/slime MEGATRON_DIR=/root/Megatron-LM \
       MODEL_DIR=/path/to/Qwen2.5-3B-Instruct \
       DATA_PATH=/path/to/gsm8k_tiny.jsonl
bash train.sh
```

**After (user writes one Python file):**
```python
# examples/math_agent/train.py
from agentcore_rl_toolkit.backends.slime import SlimeRunner

SlimeRunner(
    exp_id="gsm8k-3b-smoke",
    agent_runtime_arn="arn:aws:bedrock-agentcore:...",
    s3_bucket="my-bucket",
    model_dir="/path/to/Qwen2.5-3B-Instruct",
    data_path="/path/to/gsm8k_tiny.jsonl",
    model_type="qwen2.5-3B",
).train(num_rollout=1)
```

Runs identically; same Ray cluster, same SGLang engines, same GRPO.

## Why

**train.sh is intimidating and viral.** 121 lines of bash with ~50 flag
args, many of which are cargo-culted from slime's own reference
scripts. Users who want to change one thing end up reading all of it.

**config.yaml vs env-var vs train.sh-arg is a three-way split with no
good rule.** A user today has to know:
- ACR ARN goes in config.yaml
- MODEL_DIR goes in env var
- `--num-rollout` goes in train.sh `--num-rollout` flag directly
- `--optimizer-cpu-offload` lives inside train.sh's CLI and can't be
  overridden from outside

A single Python constructor gives one place for everything and lets
us pick sensible defaults per knob without surfacing the decision.

**The knobs that actually vary per experiment are few.** Walking
through train.sh:

| Varies per experiment | Fixed in practice |
|---|---|
| `agent_runtime_arn`, `s3_bucket` | `gateway_port=9090` |
| `model_dir`, `data_path`, `model_type` | `acr_timeout=900`, `acr_tps_limit=25`, `max_concurrent=100` |
| `num_rollout` | `rollout_batch_size=32`, `n_samples_per_prompt=8` |
| `num_gpus`, `tp_size` | `lr=1e-6`, `weight_decay=0.1`, `adam_beta2=0.98` |
| `model_id` (rare) | `reward_postprocessing="grpo"` |
|  | `--optimizer-cpu-offload` etc. (always set) |

Five-ish fields are genuinely per-experiment; the rest are "you'd
change them only if you know what you're doing." Natural home for
defaults-in-code + kwargs-for-escape.

**train.sh's hidden complexity is learnable once, not every run.**
The Ray start/stop, `--runtime-env-json` construction, `source
scripts/models/qwen2.5-3B.sh`, `--norm-epsilon` override — these are
setup steps that should happen once per training framework version,
not once per user.

## Post-change usage

**Minimal (3B smoke test):**

```python
SlimeRunner(
    exp_id="gsm8k-3b-smoke",
    agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:...:runtime/...",
    s3_bucket="my-bucket",
    model_dir="/workspace/slime_workdir/models/Qwen2.5-3B-Instruct",
    data_path="/workspace/slime_workdir/data/gsm8k_tiny.jsonl",
    model_type="qwen2.5-3B",
).train()   # num_rollout=1 by default (smoke test)
```

**32B full run:**

```python
SlimeRunner(
    exp_id="gsm8k-32b-grpo-2026-05-10",
    agent_runtime_arn="...",
    s3_bucket="...",
    model_dir="/workspace/slime_workdir/models/Qwen2.5-32B-Instruct",
    data_path="/workspace/slime_workdir/data/gsm8k_tiny.jsonl",
    model_type="qwen2.5-32B",
    num_gpus=8,
    tp_size=8,
    rollout_gpus_per_engine=8,
).train(num_rollout=100)
```

**Power user who needs to override a hidden default:**

```python
SlimeRunner(
    exp_id="gsm8k-3b-lr-sweep-5e7",
    agent_runtime_arn="...",
    s3_bucket="...",
    model_dir="...",
    data_path="...",
    model_type="qwen2.5-3B",
    # Override defaults; kwarg names match train.sh's flags
    n_samples_per_prompt=16,
    lr=5e-7,
    eps_clip_high=0.3,
).train(num_rollout=50)
```

**Native slime arg passthrough.** Any CLI flag slime or Megatron-LM
accepts can be passed through the runner using its snake-case form.
The runner converts `foo_bar=value` → `--foo-bar value` when building
the slime CLI, matching how slime itself maps argparse. If you want to
set a slime arg we didn't surface as a named kwarg, use `extra_flags`:

```python
SlimeRunner(
    ...,
    extra_flags=["--num-epoch", "3", "--use-rollout-routing-replay"],
).train()
```

Reference for the full set of accepted flags:
- **slime arguments:** [`slime/utils/arguments.py`](https://github.com/THUDM/slime/blob/main/slime/utils/arguments.py) — RL / rollout / training knobs.
- **Megatron-LM arguments:** [`megatron/training/arguments.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py) — model-parallelism, optimizer, checkpointing.
- **SGLang server args:** [`sglang.srt.server_args`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py) — inference-engine tuning; exposed by slime via `--sglang-*` flags.

Everything listed in those three files is reachable from `SlimeRunner`
without framework changes.

## Design

### Module layout and public surface

```
src/agentcore_rl_toolkit/backends/slime/
├── __init__.py           # exposes only `SlimeRunner`
├── runner.py             # SlimeRunner (new)
├── SETUP.md
├── integration/          # implementation detail — not part of public surface
│   ├── __init__.py
│   ├── gateway.py
│   ├── rewards.py
│   ├── rollout.py        # generate_rollout still referenced by slime via dotted path
│   └── traces.py
├── patches/              # SGLang patch scripts
└── examples/
    └── math_agent/
```

**Public API:**
```python
from agentcore_rl_toolkit.backends.slime import SlimeRunner  # only this
```

`slime/__init__.py`:
```python
from .runner import SlimeRunner

__all__ = ["SlimeRunner"]
```

Everything in `integration/` remains importable by its full path —
that's load-bearing because `train.sh` passes
`agentcore_rl_toolkit.backends.slime.integration.rollout.generate_rollout`
as a slime CLI arg, and slime loads it via `importlib`. But those
modules are **not** part of the documented public API — users shouldn't
import them directly. The `__init__` re-export is the contract; dotted
paths to `integration.*` are internal plumbing.

No renames, no file moves — `integration/` is already organized
correctly.

### Public class

```python
# src/agentcore_rl_toolkit/backends/slime/runner.py

from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class SlimeRunner:
    # --- Required: per-experiment, no sensible default ---
    exp_id: str             # experiment name; used as S3 prefix and wandb run group
    agent_runtime_arn: str
    s3_bucket: str
    model_dir: str          # HF checkpoint path (used for both --hf-checkpoint and --ref-load)
    data_path: str          # training JSONL
    model_type: str         # e.g. "qwen2.5-3B" — selects slime model script

    # --- Optional: cluster ---
    num_gpus: int = 8
    tp_size: int = 2
    rollout_gpus_per_engine: int = 2
    slime_dir: str = "/root/slime"            # container default
    megatron_dir: str = "/root/Megatron-LM"   # container default

    # --- Optional: ACR / toolkit ---
    model_id: str = "default"
    acr_timeout: int = 900
    acr_tps_limit: int = 25
    max_concurrent: int = 100
    gateway_port: int = 9090
    reward_postprocessing: str = "grpo"

    # --- Optional: training hyperparameters ---
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 1024
    rollout_temperature: float = 1.0
    lr: float = 1e-6
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    weight_decay: float = 0.1
    adam_beta2: float = 0.98
    sglang_mem_fraction_static: float = 0.7
    max_tokens_per_gpu: int = 9216

    # --- Escape hatch ---
    extra_flags: list[str] = field(default_factory=list)

    # --- Methods ---
    def train(self, num_rollout: int = 1) -> None:
        """Run the training job. Blocks until the slime job exits."""
        self._stop_stale_processes()
        self._start_ray()
        model_args = self._source_model_script()
        runtime_env = self._build_runtime_env()
        self._submit_ray_job(num_rollout, model_args, runtime_env)

    # Internal helpers (_stop_stale_processes, _start_ray, _source_model_script,
    # _build_runtime_env, _submit_ray_job) mirror train.sh's current bash steps,
    # one-to-one.
```

All kwargs default; users only pass what varies from the defaults.

### How it shells out

`.train()` uses `subprocess` to reproduce train.sh's current behavior
step-by-step:

1. `pkill -9 sglang || true; ray stop --force || true; sleep 3`
2. `ray start --head --num-gpus ${num_gpus} --disable-usage-stats`
3. `source ${slime_dir}/scripts/models/${model_type}.sh` — captured via
   `subprocess.check_output("bash -c '...; printf \"%s\\0\" \"${MODEL_ARGS[@]}\"'")`,
   split on null bytes into a Python list.
4. Build `runtime_env_json` dict in Python (we already have the
   python-in-bash snippet in train.sh; becomes native here).
5. `ray job submit --address http://127.0.0.1:8265 --runtime-env-json=... -- python3 ${slime_dir}/train.py <flags>`
6. Stream stdout/stderr to the parent process; non-zero exit raises.

No new Ray bindings, no new slime imports. If slime/Ray changes their
interface tomorrow, one subprocess command changes, not a refactor.

### config.yaml fate

`config.yaml` becomes **optional**. Users who prefer a config file can
continue to use it:

```python
SlimeRunner.from_yaml("config.yaml").train()
```

`from_yaml` is a 10-line classmethod that reads the file and passes
the dict to `SlimeRunner(**kwargs)`. Keeps the old path working for
anyone already using yaml. The class is the primary surface; yaml is
a convenience loader.

### train.sh fate

**Stays in the repo, unchanged.** Two reasons:

1. Users who need to customize something the class doesn't expose
   (e.g. debugging slime flags) can still drop to bash.
2. It serves as the executable reference for what the class has to
   replicate — we can diff it against the class's subprocess commands
   to catch drift.

The example dir ships `train.py` (using `SlimeRunner`) as the
primary entry point; `train.sh` as the "advanced / debugging" option.

## Downstream impact

### Math agent example

`examples/math_agent/` gains a `train.py`; existing `train.sh` /
`config.yaml` stay as-is. Users get pointed at train.py first; SETUP.md
is updated to show the 5-line Python snippet.

### New agents (hypothetical)

Instantiating `SlimeRunner` with a different `data_path` and
`agent_runtime_arn` is all they need. No copying train.sh.

### rllm / verl / out-of-tree backends

Untouched. This is a slime-backend convenience — rllm and verl have
their own entry points.

### Slime internals

No changes — we call slime through the same `ray job submit` CLI as
before.

## Backwards compatibility

**Fully backwards compatible.** train.sh and config.yaml are
unchanged. Adding the class doesn't remove anything.

## Out of scope

- **Replacing train.sh.** It stays as the escape hatch. We could
  remove it later if usage drops to zero, but not in this PR.
- **Live log streaming / progress bars.** `.train()` blocks and
  streams raw stdout; no UI work.
- **Distributed / multi-node support.** Current train.sh is
  single-node; the class mirrors that.
- **Changing the `RolloutLauncher`, `RewardFunction`, or payload
  contract.** Pure subprocess wrapper.

## Open questions

1. **Should `train()` be sync (blocks) or async / detached?** Recommend
   sync — matches what `bash train.sh` does today. Async can come
   later if users ask for it.

2. **Return value from `train()`.** None (fire-and-forget like bash
   today) vs. a results dict (path to wandb run, final rollout
   metrics, etc.). Recommend None for v1; add later if needed.

3. **Should hyperparameters be a nested `HyperParams` object
   (`SlimeRunner(..., hyper=HyperParams(lr=1e-6))`) or flat kwargs as
   proposed?** Recommend flat kwargs — 11 fields is manageable, nesting
   adds an import and a line of user code for small gain.

4. **What about `config.yaml` for users who like YAML?** Kept via
   `SlimeRunner.from_yaml(path)`. Not the primary path, but still
   supported.

## Task checklist (single PR)

**Core**

- [ ] Create `src/agentcore_rl_toolkit/backends/slime/runner.py` with
      the `SlimeRunner` dataclass + `.train()` that shells out.
- [ ] Update `src/agentcore_rl_toolkit/backends/slime/__init__.py` to
      `from .runner import SlimeRunner` and set `__all__ = ["SlimeRunner"]`.
- [ ] Leave `integration/` and `patches/` as they are — same files,
      same dotted paths. No re-exports from the slime package root.
- [ ] Add `SlimeRunner.from_yaml(path)` classmethod.

**Example**

- [ ] `examples/math_agent/train.py`: 8-line Python replacement for
      the common case.
- [ ] Leave `examples/math_agent/train.sh` + `config.yaml` untouched.

**Docs**

- [ ] SETUP.md 3.4: show `train.py` as the primary recipe; keep
      train.sh example as "advanced / debugging."
- [ ] Regenerate API reference with the new class.

**Tests**

- [ ] Unit: `SlimeRunner(**minimum_kwargs)._build_runtime_env()`
      produces the expected JSON (mirrors train.sh's python snippet).
- [ ] Unit: `SlimeRunner.from_yaml` round-trips sample config.yaml
      → equivalent kwargs.
- [ ] Integration smoke (manual, not CI): 3B smoke test via `train.py`
      → same rollout 0 metrics as today's train.sh.

**Release**

- [ ] Minor version bump, additive.

## Milestones

1. **M1** — draft the class, run a local smoke to confirm the
   shell-out produces a working rollout. 1 day.
2. **M2** — PR with tests + docs. Half day.
3. **M3** — land.
