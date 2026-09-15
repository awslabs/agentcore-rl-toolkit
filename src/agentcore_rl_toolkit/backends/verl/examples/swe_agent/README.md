# SWE agent (rollout-session agent loop)

GRPO for the [`openhands_swe_agent`](../../../../../../examples/openhands_swe_agent)
on SWE-Gym, validated against SWE-bench Verified. Rollouts run as AgentCore Runtime
sessions driven by
[`RolloutSessionAgentLoop`](../../../experimental/verl/rollout_session_agent_loop.py),
with Qwen3-Coder-30B-A3B on Megatron expert parallelism + LoRA and verl's
separate-async v1 trainer.

Unlike the `math_agent` and `migration_agent` recipes, this one does not shell out to
`python -m verl.trainer.main_ppo` with a wall of `key=value` overrides. It has its own
`main.py` (it must install a custom Ray task runner) and a composed Hydra config, so a
run is a defaults list rather than a command line.

## Files

| File | Purpose |
|---|---|
| `main.py` | Driver entrypoint. Installs `TaskRunnerWithRolloutSessionResources` and calls verl's `run_ppo`. |
| `trainer.py` | Registers this recipe's `trainer_mode` names (`art_sync`, `art_colocate_async`, `art_separate_async`) — verl's three v1 trainers plus the metric mixins. Loaded via `VERL_USE_EXTERNAL_MODULES`, never imported by `main.py`. |
| `config/main.yaml` | The run. A defaults list over the groups below, plus this run's trainer settings. |
| `config/*.yaml` | One concern each — see [Config groups](#config-groups). |

## Install

Megatron-Bridge requires Python 3.12. From the repository root:

```bash
uv sync --python 3.12 --extra verl --extra rollout --group verl-megatron
```

`--extra rollout` is what the other verl recipes do not need: this loop drives
`agentcore_rl_toolkit.rollout_session` and `.aws_tools` (aioboto3, backoff), which the
`[verl]` extra does not pull in. The gateway's own deps (aiohttp, transformers) are
already in `[verl]`.

## Prerequisites

1. **Deploy the agent.** Follow
   [`examples/openhands_swe_agent/README.md`](../../../../../../examples/openhands_swe_agent/README.md)
   — `./deploy.py` builds and pushes the agent image and creates the runtime, the
   execution role, the EC2 capacity provider, the ECR pull-through cache for task
   images, and the DynamoDB session table.
2. **Build the dataset parquet.** Same example: `./preprocess.py` for the whole split,
   then re-run it with `--noop-experiment / --oracle-experiment / --model-experiment`
   to write the difficulty-filtered training subset. `dataset_swe_gym.yaml` expects
   `$dataset_path_prefix/swe_gym_train_qwen3_coder_30b_challenged.parquet` (train) and
   `$dataset_path_prefix/swe_bench_verified.parquet` (val), both readable from every
   trainer node.
3. **Network.** Session containers must reach the trainer's CPU nodes on the rollout
   gateway's auto-assigned port — that is where the agent's OpenAI-protocol calls are
   captured. The capacity provider's subnets and security groups have to allow that
   egress.
4. **Credentials.** The trainer calls AWS with the *ambient* credentials of the Ray
   processes (not the runtime's execution role): `bedrock-agentcore:InvokeAgentRuntime`
   and `StopRuntimeSession` on the runtime, DynamoDB writes on the session table, S3
   `PutObject` under `$rollout_output_s3`, and `ec2:DescribeInstances` for the EC2
   monitor.

## Environment

Every value below is read as `${oc.env:...}` from the config, so a missing one fails
config composition rather than surfacing later. Names match the example's
`config.toml` keys, so a deploy and a training run can be given the same values.

| Variable | Used by | Value |
|---|---|---|
| `AWS_REGION` | every AWS client this loop opens | the region the runtime, table, bucket and cache live in |
| `agentcore_runtime_arn` | `rollout_session_backend` | `arn:aws:bedrock-agentcore:<region>:<acct>:runtime/<name>` |
| `agentcore_capacity_provider_arn` | `rollout_session_backend`, EC2 monitor | the pool `deploy.py` created |
| `agent_dynamodb_table` | per-session state | the session table; the same one `analyze.py` reports from |
| `rollout_output_s3` | per-rollout dump | `s3://bucket/prefix` |
| `docker_image_namespace` | forwarded to the container as a task field | ECR pull-through-cache prefix, joined with the parquet's `docker_image_uri` by `swe_unpack.sh` |
| `dataset_path_prefix` | `data.train_files` / `data.val_files` | directory holding the parquets |
| `checkpoint_path_prefix` | `trainer.default_local_dir` | checkpoints land in `<prefix>/<project_name>/<experiment_name>` |
| `VERL_USE_EXTERNAL_MODULES` | verl's import hook | see below — **required**, and required on every node |

Optional: `HYDRA_FULL_ERROR=1` for readable composition errors, `VERL_LOGGING_LEVEL`
(the loop and both mixins default their loggers to `WARN` from it), and `mlflow`
settings — `smart_ppo.yaml` logs to `console` + `mlflow`.

### `VERL_USE_EXTERNAL_MODULES`

`verl/__init__.py` reads this variable and `importlib.import_module`s each
comma-separated module path *as a side effect of importing verl*. That is the only
hook this recipe has for code that must be registered in a process it does not own:

```bash
export VERL_USE_EXTERNAL_MODULES=agentcore_rl_toolkit.backends.experimental.verl,agentcore_rl_toolkit.backends.verl.examples.swe_agent.trainer
```

**verl does not forward this variable to Ray workers.** `get_ppo_ray_runtime_env()`
propagates a fixed list (`VERL_FULL_DETERMINISM`, `PYTHONHASHSEED`, …) and this is not
on it, so the variable must already be in each worker's environment:

- Single node, Ray started implicitly by `ray.init()` inside `run_ppo` — exporting it
  in the launch shell is enough, workers inherit it.
- Multi-node cluster started with `ray start` — export it in each node's `ray start`
  shell, or pass it explicitly:
  `++ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=...`.

The same applies to the `${oc.env:...}` variables above: interpolations are resolved
lazily, and the `rollout_session_agent_loop` node is instantiated inside the agent-loop
worker, so those names must exist there too. Exporting everything cluster-wide is the
simple rule.

## Train

```bash
export HYDRA_FULL_ERROR=1
export AWS_REGION=us-west-2
export agentcore_runtime_arn=arn:aws:bedrock-agentcore:...:runtime/...
export agentcore_capacity_provider_arn=arn:aws:bedrock-agentcore:...
export agent_dynamodb_table=...
export rollout_output_s3=s3://.../...
export docker_image_namespace=<acct>.dkr.ecr.<region>.amazonaws.com/<cache-prefix>
export dataset_path_prefix=/fsx/.../datasets
export checkpoint_path_prefix=/fsx/.../checkpoints
export VERL_USE_EXTERNAL_MODULES=agentcore_rl_toolkit.backends.experimental.verl,agentcore_rl_toolkit.backends.verl.examples.swe_agent.trainer

python main.py main
```

**The config name is the last argument, not a flag.** `main.py` does
`config_name = sys.argv.pop()` before handing the rest to `@hydra.main`, so Hydra
overrides go *before* it:

```bash
python main.py trainer.total_epochs=3 actor_rollout_ref.rollout.n=8 main
```

`main.py` also derives three things at launch rather than letting the config state
them: `trainer.experiment_start_at` (the timestamp every session record and S3 key is
grouped under), `ec2_instance_type` (from instance metadata, for the run's provenance),
and `data.train_batch_size = separate_async.parameter_sync_step *
actor.ppo_mini_batch_size`, which keeps the async pipeline's batch arithmetic
consistent — so do not override `train_batch_size`.

## Config groups

`main.yaml`'s defaults list is composed from `pkg://verl.trainer.config` plus these:

| Group | What it fixes |
|---|---|
| `rollout_session_agent_loop` | The loop: `default_agent_loop`, gateway settings, session bounds, AgentCore backend, storage. The one group that is not a verl concern. |
| `smart_ppo` | GRPO algorithm settings, async vLLM rollout, logging, checkpoint dir. |
| `rollout_correction_decoupled` | Vanilla policy loss with rollout correction bypassed. |
| `dataset_swe_gym` | Train/val parquets, `n=16`, mini-batch size, 64k context. |
| `qwen3_30b_p6` / `qwen3_30b_p5` | Model path and Megatron/vLLM parallelism for P6-B200 / P5. |
| `megatron`, `megatron_offload`, `megatron_recompute` | Engine strategy, param/optimizer offload, recompute policy. |
| `lora` | Rank 64 / alpha 128, `lr=1e-5` (10x the full-FT rate, per Thinking Machines' LoRA note). |
| `async_separate_b2` | `trainer_mode: art_separate_async`, `parameter_sync_step: 2`. |
| `cluster_trainer_rollout_n{1,2,4}`, `cluster_trainer2_rollout1` | Node/GPU counts and the NCCL checkpoint engine. |

Swapping a group is a defaults edit: `qwen3_30b_p6` → `qwen3_30b_p5` for a P5 cluster,
`async_separate_b2` → nothing plus `trainer.v1.trainer_mode=art_sync` for a synchronous
run.

## Concurrency and timeouts

`rollout_session_agent_loop.yaml` carries the run-wide bounds. They are enforced by
named Ray actors created once by the task runner and looked up by every agent-loop
worker, so these are cluster-wide numbers, not per-worker ones:

- `rollout_concurrency` (128) — concurrent trajectories. Keep it at or below
  rollout DP x `max_num_seqs` or the prefix KV-cache hit rate drops.
- `container_concurrency` (256) — concurrent sessions. Capacity and cost control; a
  container is alive through setup and teardown, not just generation, so this is the
  larger number.
- `session_create_rate` (5 TPS) — AgentCore session creation allows 25 TPS with no
  burst and EC2 `CreateFleet` 5 TPS with 50 burst, so 5 is the binding limit.
- `container_setup_timeout` / `agent_run_timeout` (1200s each) — setup covers unpacking
  the multi-GB task image, so it is as generous as the run itself.

`ec2_monitor_poll_interval` (60s) controls the single poller that stamps each session's
EC2 instance id into the session table; one poller per run rather than a call per
session, which would exhaust the EC2 API's rate limit at this scale. It is skipped
unless the backend is `agentcore` and `dynamodb_table` is set.

## Watching a run

- **Session table** — one item per rollout, written through as it progresses:
  priorities, slot start times, span timings, timeout flags, EC2 instance, turn count,
  reward. `examples/openhands_swe_agent/analyze.py <experiment>` reports a training run
  the same way it reports a batch eval, including while it is still in flight.
- **S3** — `$rollout_output_s3/<experiment_start_at>/<session_id>` holds the full
  `RolloutOutput`: the task, the loop outputs, the exception if any, the session meta,
  and the container's own dump.
- **Tracker metrics** — `agent_loop/<name>/{mean,min,max,sum}` for every metric the
  harness reported (turn counts, generated length, latencies, healer counters), plus
  `critic/advantages/zero_mean` and `.../zero_pass_mean`, which say how much of the
  step carried no gradient and whether the collapsed GRPO groups were all-pass or
  all-fail. Both come from the mixins in `trainer.py`.

## Notes

- No `data.custom_cls` is set: verl's default `RLHFDataset` returns the whole parquet
  row, so the extra columns `preprocess.py` writes (`task_id`, `eval_script`,
  `docker_image_uri`, `repo_path`) reach the container as task fields, and
  `extra_info.index` becomes the `task_id` the loop groups rollouts by.
- The gateway runs in `history_mode: linear` with `fork_threshold_tokens: 0`, because
  verl's v1 trainer does not yet handle one rollout producing several trainable
  trajectories. `num_records` in the session meta shows if that ever happens anyway.
- A failed rollout still emits a masked, zero-reward sample so the GRPO group stays
  intact. Container-side failures are recorded from the container's own dump rather
  than re-raised.
