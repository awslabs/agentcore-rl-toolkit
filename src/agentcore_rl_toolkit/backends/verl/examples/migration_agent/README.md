# MigrationBench agent (Megatron + LoRA, rollout-session agent loop)

GRPO for the Java 8 to 17
[`strands_migration_agent`](../../../../../../examples/strands_migration_agent), using
Qwen3-Coder-30B-A3B with Megatron expert parallelism and LoRA. Rollouts run as AgentCore
Runtime sessions driven by
[`RolloutSessionAgentLoop`](../../../experimental/verl/rollout_session_agent_loop.py) over
the `agentcore_s3` session backend — one fire-and-forget `InvokeAgentRuntime` per rollout,
with the result dict polled out of S3. **The agent container is unchanged**: same image,
same `payload`, and the same result contract a batch evaluation of it uses.

Like the `swe_agent` recipe (and unlike `math_agent`), this recipe has its own `main.py` —
it must install a custom Ray task runner — and a composed Hydra config, so a run is a
defaults list rather than a wall of `key=value` overrides.

## Files

| File | Purpose |
|---|---|
| `main.py` | Driver entrypoint. Installs `TaskRunnerWithRolloutSessionResources` and calls verl's `run_ppo`. |
| `megatron_lora_sync_grpo.sh` | Thin wrapper: sets `VERL_USE_EXTERNAL_MODULES`, checks the environment, runs `main.py`. Hydra overrides pass through. |
| `config/main.yaml` | The run. A defaults list over the groups below, plus this run's trainer settings. |
| `config/*.yaml` | One concern each — see [Config groups](#config-groups). |
| `preprocess_migrationbench.py` | Builds the train/val parquets (`task_id` + `payload` per row). |

## Install

Megatron-Bridge requires Python 3.12. From the repository root:

```bash
uv sync --python 3.12 --extra verl --extra rollout --group verl-megatron
```

`--extra rollout` is what the `math_agent` recipe does not need: this loop drives
`agentcore_rl_toolkit.rollout_session` and `.aws_tools` (aioboto3, backoff), which the
`[verl]` extra does not pull in.

## Prepare data

The agent example is a standalone uv project. From the repository root, upload its
repository tarballs and metadata using that project's environment, then build the parquets
with the root verl environment:

```bash
cd examples/strands_migration_agent
uv sync
uv run python preprocess.py --s3-bucket-name <data-bucket>

cd ../../src/agentcore_rl_toolkit/backends/verl/examples/migration_agent
uv run python preprocess_migrationbench.py \
    --s3-bucket-name <data-bucket> --output-dir "$dataset_path_prefix"
```

Every row carries a `task_id` (the repo slug) and a `payload` (the agent's exact invoke
payload). `task_id` is what rollouts are grouped by, what lands in the session record, and
what the agent's S3 results are keyed under, so a training rollout and a batch evaluation of
one repo meet on it; a row without it aborts the rollout loudly (`require_task_id`) rather
than falling back to the row index. A parquet built before this contract has to be rebuilt.
`data.custom_cls=PayloadDataset` synthesizes the chat-format `prompt` column verl's
dataloader needs from `payload.prompt`.

## Prerequisites

1. **Deploy the agent.** Follow
   [`examples/strands_migration_agent/README.md`](../../../../../../examples/strands_migration_agent/README.md).
   Nothing extra is needed for training: no capacity provider, no session table, and no new
   IAM surface on the runtime's execution role.
2. **Network.** Session containers must reach the trainer's CPU nodes on the rollout
   gateway's auto-assigned port — that is where the agent's OpenAI-protocol calls are
   captured. The runtime's egress has to allow it.
3. **Credentials.** The trainer calls AWS with the *ambient* credentials of the Ray
   processes (not the runtime's execution role): `bedrock-agentcore:InvokeAgentRuntime` and
   `StopRuntimeSession` on the runtime, and S3 reads *and* writes under `$rollout_output_s3`
   — reads because the client polls S3 for the agent's results.

## Environment

Every value below is read as `${oc.env:...}` during config composition, so a missing one
fails the launch rather than surfacing mid-run (the wrapper script checks them up front).

| Variable | Used by | Value |
|---|---|---|
| `AWS_REGION` | every AWS client this loop opens | the region of the runtime and the bucket |
| `agentcore_runtime_arn` | `rollout_session_backend` | `arn:aws:bedrock-agentcore:<region>:<acct>:runtime/<name>` |
| `rollout_output_s3` | per-rollout dump, and the agent's results | `s3://bucket/prefix` |
| `dataset_path_prefix` | `data.train_files` / `data.val_files` | directory holding the parquets, readable from every node |
| `model_path_prefix` | `actor_rollout_ref.model.path` | directory holding `Qwen3-Coder-30B-A3B-Instruct` |
| `checkpoint_path_prefix` | `trainer.default_local_dir` | checkpoints land in `<prefix>/<project_name>/<experiment_name>` |
| `VERL_USE_EXTERNAL_MODULES` | verl's import hook | set by the wrapper script — **required on every node** |

Optional: `HYDRA_FULL_ERROR=1` (the wrapper sets it) and `VERL_LOGGING_LEVEL` (the loop and
both metric mixins default their loggers to `WARN` from it). `trainer.logger` is
`console` + `wandb`, so `wandb login` first.

### `VERL_USE_EXTERNAL_MODULES`

`verl/__init__.py` `importlib.import_module`s each comma-separated path *as a side effect of
importing verl*. That is the only hook for code that must be registered in processes this
recipe does not own:

```bash
export VERL_USE_EXTERNAL_MODULES=agentcore_rl_toolkit.backends.experimental.verl,agentcore_rl_toolkit.backends.verl.trainer
```

The first registers `rollout_session_agent_loop`, the second the `agentcore_*` trainer
modes.
**verl does not forward this variable to Ray workers** — `get_ppo_ray_runtime_env()`
propagates a fixed list and this is not on it, so it must already be in each worker's
environment:

- Single node, Ray started implicitly by `ray.init()` inside `run_ppo` — the wrapper's export
  is enough, workers inherit it.
- Multi-node cluster started with `ray start` — export it in each node's `ray start` shell,
  or pass `++ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=...`.

The same applies to the `${oc.env:...}` names above: interpolations resolve lazily and the
`rollout_session_agent_loop` node is instantiated inside the agent-loop worker, so exporting
everything cluster-wide is the simple rule.

## Train

```bash
export AWS_REGION=us-west-2
export agentcore_runtime_arn=arn:aws:bedrock-agentcore:...:runtime/...
export rollout_output_s3=s3://.../migrationbench
export dataset_path_prefix=/fsx/.../datasets
export model_path_prefix=/fsx/.../models
export checkpoint_path_prefix=/fsx/.../checkpoints
wandb login

./megatron_lora_sync_grpo.sh                      # or: python main.py
./megatron_lora_sync_grpo.sh trainer.total_epochs=3 actor_rollout_ref.rollout.n=8
```

`main.py` also stamps two things at launch rather than letting the config state them:
`trainer.experiment_start_at` (the timestamp every S3 dump is grouped under) and
`ec2_instance_type` (from instance metadata, for the run's provenance).

## Config groups

`main.yaml`'s defaults list is composed from `pkg://verl.trainer.config` plus these:

| Group | What it fixes |
|---|---|
| `rollout_session_agent_loop` | The loop: `default_agent_loop`, gateway settings, session bounds, the `agentcore_s3` backend, storage. The one group that is not a verl concern. |
| `grpo` | GRPO with a KL loss against the reference policy, async vLLM rollout, validation sampling. |
| `dataset_migrationbench` | Train/val parquets, `PayloadDataset`, batch shape (`train_batch_size=32`, `n=16`), token budgets. |
| `qwen3_coder_30b_p6` | Model path, Megatron/vLLM parallelism and memory for one P6-B200 node. |
| `megatron`, `megatron_offload`, `megatron_recompute` | Engine strategy, grad/optimizer offload, recompute policy. |
| `lora` | Rank 64 / alpha 128, merged into the base weights, `lr=1e-5` (10x the full-FT rate). |
| `cluster_n1` | One node, 8 GPUs, trainer and rollout engine colocated. |

The trainer mode is `agentcore_sync` (`main.yaml`): verl's v1 sync trainer plus
`AgentLoopMetricsMixin`, `AdvantageZeroMetricsMixin`, `VariableRowBatchingMixin` and
`RolloutFailureIsolationMixin` (see
[`backends/verl/README.md`](../../README.md#trainer-modes)). Two of those are load-bearing
here: `rollout_session_agent_loop` emits one row per trajectory-tree leaf, so a session that
forks expands into several rows, and it reports a failed rollout as an inert flagged row
that `RolloutFailureIsolationMixin` moves into a GRPO group of its own instead of raising.
In `history_mode: linear` a session normally yields exactly one leaf, so in practice the
variable-row machinery mostly keeps the optimizer schedule stable if that changes.

## Concurrency and timeouts

`rollout_session_agent_loop.yaml` carries the run-wide bounds. They are enforced by named Ray
actors created once by the task runner and looked up by every agent-loop worker, so these are
cluster-wide numbers, not per-worker ones:

- `rollout_concurrency` (128) — concurrent trajectories. Keep it at or below rollout DP x
  `max_num_seqs` or the prefix KV-cache hit rate drops.
- `container_concurrency` (256) — concurrent AgentCore sessions; capacity and cost control.
- `session_create_rate` (5 TPS) — AgentCore session creation allows 25 TPS with no burst, and
  this backend spends roughly two calls per rollout (invoke + stop), which share that budget.
  It is the *only* place ACR throttling is enforced: the client's own limiter is per process,
  so it is deliberately disabled (there is no `tps_limit` knob any more).
- `agent_run_timeout` (1800s) — one invoke covers container cold start, the S3 repo fetch, the
  Maven environment setup and the agent run, so this bounds all of it. It replaces
  `agentcore_agent.yaml`'s `max_rollout_time`.
- `container_setup_timeout` — inert here (see [Limitations](#limitations)), kept equal to
  `agent_run_timeout` so it can never become the binding limit.

`max_pool_connections` (128) sizes the boto3 pools of the one `RolloutClient` the whole worker
process shares; invoke bursts queue there rather than on the API.

## Watching a run

- **S3** — `$rollout_output_s3/<experiment_start_at>/<session_id>` holds the loop's full
  `RolloutOutput` (task, loop outputs, exception, session meta, and the container's own
  result), and `$rollout_output_s3/<experiment_name>/<task_id>/<session_id>.json` holds the
  agent's result object as the container wrote it.
- **Tracker metrics** — `agent_loop/<name>/{mean,min,max,sum}` for every metric the loop
  reported (span timings, turn counts, generated length, reward), plus
  `critic/advantages/zero_mean` and `.../zero_pass_mean`, which say how much of the step
  carried no gradient and whether the collapsed GRPO groups were all-pass or all-fail. Both
  come from the mixins the `agentcore_sync` mode layers on, as do the per-update
  `batching/*` row counts.
- **Session table** — not used by this recipe (`dynamodb_table: null`), so the per-rollout
  record lives only in the S3 dump. Set the key to a table name to get a live item per
  rollout instead.

## Limitations

These come with wrapping the agent's production `RolloutClient` contract rather than the
four-POST training protocol the `agentcore_http` backend speaks; `docs/rollout_session_migration.md`
has the full list.

1. **The setup/run split is vacuous.** One invoke both creates the session and starts the
   rollout, so `setup` provisions nothing and `container_setup_timeout` bounds nothing. The
   container and rollout semaphores still cap concurrency, but no longer separate
   provisioning from generation.
2. **Coarser failure detection.** There is no status channel, so a container that dies without
   the SDK's error path running (OOM, process death) is only noticed when `agent_run_timeout`
   expires. Exceptions raised inside the handler still come back promptly, as
   `{status_code: 500, stop_reason: ...}` through S3.
3. **Reward and metrics stay convention-based** (`{"rewards": ...}` plus an optional `metrics`
   dict). The migration agent returns only rewards, so `agent_loop/*` will be sparse and
   verl's fixed `AgentLoopMetrics` scalars (`llm_latency_sum`, `tool_calls_time_s`,
   `eval_latency_s`) stay zero until the agent measures and returns them.
4. **Completion is polled.** `RolloutFuture` HEADs S3 with per-future exponential backoff, so
   a finished rollout is noticed up to 30s late.
5. A failed rollout still emits a masked, zero-reward sample so the GRPO group stays intact.
6. The gateway runs in `history_mode: linear` with `fork_threshold_tokens: 0`, because verl's
   v1 trainer does not yet handle one rollout producing several trainable trajectories.
   `num_records` in the session meta shows if that ever happens anyway.

Qwen3-Coder's chat-template hash maps to the gateway's `qwen3_5` response schema, so tool-call
and reasoning parsing happen in the renderer; engine-level parser flags have no effect on that
token-in/token-out path and are deliberately not set.
