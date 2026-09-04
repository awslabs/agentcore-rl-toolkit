# Stable optimizer-step batching for variable-row verl rollouts

## Summary

Agentic rollouts can emit a variable number of training rows: a nominal rollout may
produce one row, or several rows after context compaction, trajectory forks, or sub-agent
branches. verl v1 currently treats those expanded rows as ordinary fixed-size PPO
mini-batches. Consequently, the number of optimizer steps in one outer training batch grows
with the number of emitted rows:

```
optimizer steps today
    = padded_expanded_rows / (ppo_mini_batch_size * rollout.n) * ppo_epochs
```

That couples optimization dynamics to agent trajectory shape. Two batches containing the
same number of source prompts can take different numbers of Adam steps solely because one
batch produced more branch rows.

The proposed fix makes optimizer-step count depend on the configured source-prompt batch,
while still training every emitted row:

```
num_actor_mini_batches
    = data.train_batch_size / actor.ppo_mini_batch_size

actor optimizer steps
    = num_actor_mini_batches * actor.ppo_epochs
```

The worker already exposes the right interface: `TrainingWorker.train_mini_batch` accepts
`num_mini_batch` as an alternative to a fixed `mini_batch_size`. The implementation should
use that interface, pad only enough for the configured number of actor optimizer partitions
and actor data-parallel ranks, and otherwise preserve V1's current whole-batch DP balancing.

The integration uses verl's public trainer registry and subclasses `PPOTrainerSync`.
Because the registry is process-local and trainer lookup happens inside a separate Ray
task-runner actor, the AgentCore launcher queues registration on the stock `TaskRunnerV1`
actor before queuing its stock `run()` method.

**Status:** implemented in this worktree. A one-step, eight-GPU Qwen3-4B/GSM8K run has
validated the full FSDP + ACR path; end-to-end MigrationBench validation remains.

## Goals

- Keep optimizer-step count stable when a rollout emits a variable number of rows.
- Support any integral `num_mini_batches >= 1`, not only the current recipes where it is
  one.
- Train every real branch row and preserve verl's current additive row-loss semantics.
- Reduce synthetic padding from the current fixed-row mini-batch multiple to the minimum
  required by DP and the configured optimizer-step schedule.
- Preserve V1's existing whole-batch token balancing across DP ranks.
- Use verl's public trainer customization surface rather than patching site-packages,
  replacing worker engines, or monkey-patching from the agent-loop manager.
- Keep FSDP and Megatron behavior behind the same `TrainingWorker.train_mini_batch`
  contract.
- Fail early on configurations outside the first implementation's intentionally narrow,
  tested contract instead of partially supporting them.

## Non-goals

- Keeping every row from one rollout/session in the same optimizer step. The current loss is
  additive over rows; crossing an optimizer boundary changes ordinary SGD/Adam ordering but
  is not a correctness violation.
- Changing how branch rows are weighted. More branches still contribute more additive loss
  mass, as they do today.
- Changing GRPO advantage grouping. verl computes and broadcasts multi-trajectory
  advantages before the actor update; this proposal only changes update partitioning.
- Changing dynamic micro-batching. `use_dynamic_bsz` decides how one optimizer partition is
  split into forward/backward micro-batches; it does not decide how many optimizer steps
  occur.
- Supporting async trainer modes in the first implementation. The AgentCore recipes use
  `PPOTrainerSync`; the batching helper itself should remain mode-agnostic so the mixin can
  be reused later.
- Supporting a critic in the first implementation. The checked-in AgentCore recipes use
  actor-only GRPO; critic scheduling should be handled by a generic upstream verl change.
- Supporting on-policy distillation in the first implementation.
- Restoring legacy V0's per-mini-batch DP token balancing. V0 implements
  `_balance_batch(..., keep_minibatch=True)`; V1 retains the argument but ignores it and
  balances only the whole batch across DP ranks. This proposal deliberately preserves the
  V1 behavior.

## Terminology

- **source prompt** — one item counted by `data.train_batch_size`.
- **nominal rollout** — one of the `rollout.n` samples generated for a source prompt.
- **expanded row** — one training row emitted by a nominal rollout. A rollout tree may emit
  several.
- **optimizer partition** — the global row chunk consumed by one optimizer step, before it
  is split over DP ranks.

In earlier discussion, “partition” referred to the global chunk before DP split. This
document uses **optimizer partition** for that concept; it does not introduce a separate
name for each rank's local slice.

## Current verl behavior

The relevant verl 0.9.0 path is:

1. `PPOTrainer._balance_batch` asks `_get_required_batch_multiple(dp_size)` for an
   upsampling multiple.
2. `_get_required_batch_multiple` computes the LCM of DP size and each fixed global row
   mini-batch:

   ```
   actor_global_mini_batch_size = actor.ppo_mini_batch_size * rollout.n
   ```

3. `_balance_batch` pads to that multiple and token-balances the entire expanded batch into
   `dp_size` contiguous rank chunks.
4. `_update_actor` passes both:

   ```
   global_batch_size = actor.ppo_mini_batch_size * rollout.n
   mini_batch_size   = actor.ppo_mini_batch_size * rollout.n
   ```

5. DP dispatch gives each rank one contiguous chunk.
6. `TrainingWorker.train_mini_batch` iterates fixed-size local mini-batches.
7. Every iterator item calls `BaseEngine.train_batch`, and every `train_batch` performs one
   `optimizer_step()`.

Legacy V0 has a `keep_minibatch=True` path that balances each configured mini-batch across
DP separately. V1's `_balance_batch` signature still contains `keep_minibatch`, but its
implementation does not branch on it: V1 always balances the whole padded batch into
`dp_size` contiguous rank chunks. This proposal does not change that behavior.

For the MigrationBench recipe:

```
data.train_batch_size                 = 32
actor.ppo_mini_batch_size             = 32
rollout.n                              = 16
actor DP size                          = 2
fixed global row mini-batch            = 512
configured source-prompt mini-batches  = 1
```

With 512 expanded rows, verl takes one actor step. With 1,536 expanded rows, it takes three
actor steps. The extra two steps come from trajectory expansion, not from the configured PPO
schedule.

The same fixed-row value also drives padding. A 513-row batch is padded to 1,024 rows even
though a one-step, DP=2 update only requires an even row count: 514.

## Required invariants

For the actor, define:

```
M = data.train_batch_size / actor.ppo_mini_batch_size
```

The configuration must make `M` an integer. For each `_step_once` in the sync trainer:

1. the actor performs exactly `M * actor.ppo_epochs` optimizer steps;
2. all real expanded rows are consumed once per PPO epoch;
3. every optimizer partition is evenly divisible over actor DP;
4. corresponding local iterator positions on all DP ranks refer to the same global
   optimizer partition;
5. padding rows have zero loss and do not affect reward/length metrics;
6. the loss denominator remains the configured nominal row mini-batch
   `actor.ppo_mini_batch_size * rollout.n`, preserving existing branch-row weighting.

## Design

### 1. Compute optimizer counts from source prompts

The number of optimizer partitions is based on configured source-prompt batching, not the
post-rollout row count:

```python
def configured_num_mini_batches(
    train_batch_size: int,
    ppo_mini_batch_size: int,
) -> int:
    if train_batch_size % ppo_mini_batch_size:
        raise ValueError(
            "data.train_batch_size must be divisible by ppo_mini_batch_size"
        )
    return train_batch_size // ppo_mini_batch_size
```

`rollout.n` intentionally does not appear in this count. It still defines the nominal row
count and the existing loss normalization, but it must not multiply the number of
optimizer steps.

For `ppo_epochs > 1`, verl's worker repeats the `M` partitions once per epoch, so the
result is `M * ppo_epochs` steps.

### 2. Keep the first implementation actor-only

The generic actor/critic case requires coordinating two potentially different DP sizes and
two mini-batch schedules. That is a valid upstream verl problem, but it is not needed to
validate the AgentCore training result.

The custom trainer should call verl's `need_critic(config)` during construction and reject
the configuration if it returns true:

```python
if need_critic(config):
    raise NotImplementedError(
        "agentcore_sync currently supports actor-only training; "
        "use a stock verl trainer for actor-critic training"
    )
```

This makes the scheduling model unambiguous:

```
D = actor data-parallel size
M = data.train_batch_size / actor.ppo_mini_batch_size
G = D * M
```

`G` is the required padding multiple.

### 3. Pad only to `actor_dp * num_mini_batches`

Let:

```
T = number of real expanded rows
G = actor_dp_size * actor_num_mini_batches
```

The target row count is:

```
T_padded = ceil(T / G) * G
```

This is the minimum size that gives every actor `(DP rank, optimizer partition)` balance
partition equal row cardinality.

Examples:

| Case | Real rows | Old required multiple | New required multiple | Padded rows |
|---|---:|---:|---:|---:|
| Migration recipe, `D=2`, `A=1` | 513 | 512 | 2 | 514 |
| `D=2`, actor `A=4` | 270 | depends on `rollout.n` | 8 | 272 |
| `D=4`, actor `M=3` | 601 | depends on `rollout.n` | 12 | 612 |

The existing verl `upsample_batch_to_divisible_size` helper remains suitable: its synthetic
rows have a two-token `[EOS, EOS]` shape, zero response/loss masks, zero rewards/logprobs,
fresh padding UIDs, and `is_padding=True`. Only the requested multiple changes.

Efficient padding and stable optimizer-step count are logically separable, but they should
land in the same implementation because both derive from the same schedule plan. Applying
only the `num_mini_batch` change would still leave unnecessary padding; applying only the
padding change would make the fixed-size worker iterator fail or continue producing a
variable step count.

### 4. Preserve V1's existing whole-batch DP balancing

No custom row-layout algorithm is needed.

The subclass changes `_get_required_batch_multiple(dp_size)` to return `D * M`, then calls
the inherited V1 `_balance_batch` unchanged. Stock V1 will:

1. pad the expanded batch to a multiple of `D * M`;
2. token-balance the whole padded batch into `D` equal-cardinality contiguous rank chunks;
3. let DP dispatch send one chunk to each rank.

Each rank therefore receives `N / D` rows. The worker receives
`num_mini_batch=M` and derives its local mini-batch size:

```
local_mini_batch_size = (N / D) / M = N / (D * M)
```

Padding to `D * M` is exactly what makes that quantity integral. Nothing needs to define or
store the per-rank, per-mini-batch row count separately.

For `M > 1`, individual local mini-batches may have uneven token workloads even though each
rank's whole chunk is balanced. That matches current V1 behavior and affects performance,
not optimizer-step correctness. Restoring V0's `keep_minibatch=True` behavior is left to a
generic verl change.

This design does not group rows by UID or session ID. Rows from one rollout may occupy
different optimizer partitions, which is acceptable for the current additive objective.

### 5. Select `num_mini_batch`, not `mini_batch_size`

For actor updates, the custom trainer should pass:

```python
nominal_global_batch_size = (
    config.actor_rollout_ref.actor.ppo_mini_batch_size
    * config.actor_rollout_ref.rollout.n
)

batch.extra_info.update(
    {
        "global_batch_size": nominal_global_batch_size,
        "num_mini_batch": actor_num_mini_batches,
        "epochs": config.actor_rollout_ref.actor.ppo_epochs,
        "seed": config.actor_rollout_ref.actor.data_loader_seed,
        "dataloader_kwargs": {"shuffle": config.actor_rollout_ref.actor.shuffle},
        # Existing entropy and temperature fields remain unchanged.
        # Distillation flags are fixed false by the supported-config contract.
    }
)
```

Do not pass `mini_batch_size` at the same time. `TrainingWorker.train_mini_batch` gives
`mini_batch_size` precedence and would return to fixed-row behavior.

The fixed `global_batch_size` denominator is intentional. For
`seq-mean-token-sum`, verl computes:

```
sum(real row token losses) / nominal_global_batch_size
```

Collapsing several expanded-row chunks into one optimizer step therefore behaves like
gradient accumulation of their additive losses before Adam, instead of silently averaging
away branch rows. Padding rows have zero loss masks and add no gradient.

**Scope note:** this PR changes optimizer-step scheduling, not objective weighting. It
preserves the nominal `global_batch_size` denominator and therefore preserves the current
additive branch-row semantics: a rollout that emits more real branch rows contributes more
aggregate loss mass. Normalizing by actual expanded rows or first reducing rows per
rollout/session would define a different objective and belongs in a separate design and
change.

### 6. Fail fast outside the validated first-version contract

`AgentCorePPOTrainerSync.__init__` should validate the whole supported surface before
calling `super().__init__` far enough to create workers:

```python
def validate_agentcore_sync_config(config) -> None:
    if not config.trainer.use_v1:
        raise ValueError("agentcore_sync requires trainer.use_v1=true")
    if need_critic(config):
        raise NotImplementedError("agentcore_sync does not support a critic yet")
    if is_distillation_enabled(config.get("distillation")):
        raise NotImplementedError("agentcore_sync does not support distillation yet")
    if config.actor_rollout_ref.actor.loss_agg_mode != "seq-mean-token-sum":
        raise NotImplementedError(
            "agentcore_sync currently requires loss_agg_mode=seq-mean-token-sum"
        )

    train_batch_size = config.data.train_batch_size
    ppo_mini_batch_size = config.actor_rollout_ref.actor.ppo_mini_batch_size
    if train_batch_size % ppo_mini_batch_size:
        raise ValueError(
            "data.train_batch_size must be divisible by actor.ppo_mini_batch_size"
        )
```

After the custom registry alias is normalized to stock `"sync"` semantics, also assert
`parameter_sync_step == 1`. The schedule is defined per sync `_step_once`; supporting
multi-trigger async schedules should be an upstream generalization.

These checks intentionally match the two checked-in, end-to-end-tested AgentCore recipes.
They do not reject unrelated actor features such as FSDP versus Megatron, LoRA, reference
policy/KL, rollout correction, dynamic micro-batching, or `ppo_epochs >= 1`.

## verl customization and integration cut

### Public interface to use: trainer registry

verl's registry accepts any `PPOTrainer` subclass, but the concrete lifecycle contract is
mode-specific. This implementation should directly subclass and register the stock sync
trainer:

```python
from omegaconf import open_dict
from verl.trainer.ppo.v1 import PPOTrainerSync, register_trainer


@register_trainer("agentcore_sync")
class AgentCorePPOTrainerSync(PPOTrainerSync):
    def __init__(self, config):
        # "agentcore_sync" is a registry selection alias. Internally this trainer
        # must retain verl's sync replay-buffer and dataloader semantics.
        with open_dict(config):
            config.trainer.v1.trainer_mode = "sync"
        super().__init__(config)

    # Override only the batching/update seams described below.
```

The normalization to `"sync"` is necessary in verl 0.9.0 because the base trainer uses the
literal mode string to select `ReplayBuffer` versus `ReplayBufferAsync`, exact-refill
behavior, and mode-specific config. The custom registry name is only a lookup key; runtime
semantics remain stock sync.

### Trainer registration on the stock task-runner actor

Stock `TaskRunnerV1.run()` performs:

```python
trainer_cls = get_trainer_cls(config.trainer.v1.trainer_mode)
```

before it loads `agent_loop_manager_class` or any AgentCore-owned class. Therefore this does
not work by itself:

```bash
python -m verl.trainer.main_ppo trainer.v1.trainer_mode=agentcore_sync
```

Nothing has imported the module containing `@register_trainer("agentcore_sync")` when the
lookup occurs.

Importing the trainer in the driver process is still insufficient. `run_ppo` creates
`TaskRunnerV1` as a separate Ray actor, and verl's trainer registry is ordinary
process-local Python state. Registration performed in the driver does not appear in that
actor.

The launcher uses verl's public `run_ppo(..., task_runner_class=...)` seam with a thin
factory. The factory creates stock `TaskRunnerV1`, then uses Ray's built-in
`__ray_call__` developer API to queue one import on that actor:

```python
# agentcore_rl_toolkit/backends/verl/main_ppo.py
def _register_agentcore_trainer(_task_runner):
    from . import trainer as _trainer  # noqa: F401


class _ConfiguredTaskRunnerV1:
    def remote(self):
        runner = self.task_runner.remote()
        runner.__ray_call__.remote(_register_agentcore_trainer)
        return runner
```

Calls submitted through one single-threaded actor handle execute in submission order.
`run_ppo` therefore queues stock `run()` after registration without replacing its body.

The full import/registration chain is:

1. `python -m agentcore_rl_toolkit.backends.verl.main_ppo` starts the Ray job.
2. Stock `run_ppo` asks the factory for a task-runner actor.
3. The factory creates stock `TaskRunnerV1` and queues `_register_agentcore_trainer`.
4. Importing `trainer.py` inserts the class into that actor's
   `TRAINER_REGISTRY`.
5. Stock `TaskRunnerV1.run()` performs the lookup and owns initialization, TransferQueue,
   fitting, logger shutdown, and queue shutdown.

The registry and module discovery are separate concerns. verl 0.9.0 provides the former
but has no config field that imports an external trainer module before
`get_trainer_cls(...)`. Running stock `python -m verl.trainer.main_ppo` therefore imports
only verl's built-in trainer modules; it has no reason to import this package's
`trainer.py`, so the external decorator never executes. Driver-side import alone also
cannot cross the Ray process boundary. Actor-specific setup hooks are not executed by the
pinned Ray version, while job-level setup hooks import trainer/torch before GPU workers are
assigned and break CUDA rank isolation. Queuing one actor call is the narrow compatibility
bridge that avoids both failures.

### Why not use `agent_loop_manager_class`

`agent_loop_manager_class` is a public FQN hook, but it is instantiated only after
`trainer.init()`. It could mutate the already-created trainer class before `fit()`, but that
would be a timing-dependent monkey patch unrelated to the manager's rollout responsibility.
The trainer registry expresses the ownership correctly and is easier to test.

### Trainer override surface

The custom subclass should override only:

- `_get_required_batch_multiple(dp_size)` — return
  `dp_size * actor_num_mini_batches`;
- `_update_actor(...)` — preserve stock actor metadata/metrics while replacing
  `mini_batch_size` with `num_mini_batch`.

The inherited `_balance_batch` remains unchanged. It discovers actor DP through verl's
existing `"actor"` mesh dispatch metadata, calls the overridden required-multiple method,
pads to `D * M`, and performs V1's normal whole-batch DP balancing.

Since distillation is explicitly unsupported, `_update_actor` only needs to preserve stock
entropy, epoch, seed, temperature, metric-reduction, and worker-call behavior while setting
the two distillation booleans to false. verl 0.9.0 does not expose a hook for constructing
update `extra_info`, so this small override is the unavoidable compatibility surface. The
repository pins verl exactly, and integration tests should detect upstream
signature/metadata drift when that pin is changed.

No worker, engine, dispatcher, TransferQueue, advantage, or loss implementation is
replaced.

## Files to change

### New files

| File | Purpose |
|---|---|
| `src/agentcore_rl_toolkit/backends/verl/trainer.py` | `AgentCorePPOTrainerSync`, config validation, inline schedule arithmetic, the two narrow overrides, and the Ray registration hook. |
| `src/agentcore_rl_toolkit/backends/verl/main_ppo.py` | Stock validation and `run_ppo`, plus a thin factory that queues trainer registration on stock `TaskRunnerV1`. |
| `tests/backends/verl/test_training_worker_batching.py` | Regression tests against the installed verl `TrainingWorker.train_mini_batch`, using a counting worker method only at the real optimizer boundary. |
| `tests/backends/verl/test_trainer_batching.py` | Trainer metadata, fail-fast configuration, and fresh-process registration tests; no fake TransferQueue. |
| `tests/backends/verl/test_main_ppo.py` | Verifies that the factory preserves task-runner options and queues registration before returning the actor. |

### Modified files

| File | Change |
|---|---|
| `src/agentcore_rl_toolkit/backends/verl/__init__.py` | Update the integration description; lazily expose the trainer only if a public import is useful. Keep ordinary package import verl-light. |
| `src/agentcore_rl_toolkit/backends/verl/agent_loop.py` | Document the AgentCore launcher contract. No rollout logic changes. |
| `src/agentcore_rl_toolkit/backends/verl/README.md` | Document stable optimizer steps, minimal padding, `agentcore_sync`, and preserved V1 balancing semantics. |
| `src/agentcore_rl_toolkit/backends/verl/examples/math_agent/fsdp_fft_sync_grpo.sh` | Select the AgentCore launcher and custom trainer registry name. |
| `src/agentcore_rl_toolkit/backends/verl/examples/migration_agent/megatron_lora_sync_grpo.sh` | Select the AgentCore launcher and custom trainer registry name. This is the primary regression case. |
| `docs/site/src/content/docs/guides/verl-backend-setup.md` | Document the AgentCore invocation and registration behavior. |

No change is planned for the rollout gateway, `AgentCoreAgentLoop`, trajectory grouping, or
verl site-packages.

## Validation plan

### Testing principle

Test observable behavior through as much real verl code as possible. Do not build a fake
worker group + fake TransferQueue pipeline and call it an integration test: if the test only
asserts that mocks received the same fields the implementation wrote, it is tautological.

The implementation does not modify TransferQueue, DP dispatch, or the body of
`_balance_batch`, so there is no reason to emulate those systems in unit tests. Mocking is
limited to expensive external boundaries:

- a counting worker method at the boundary where a real GPU optimizer would run;
- a recording actor worker-group call for the narrow trainer metadata seam;
- no fake TransferQueue.

### Trainer arithmetic tests

- `train_batch_size=32`, actor mini=32, `D=2`:
  - actor count is 1;
  - required multiple is 2;
  - 513 rows target 514, not 1,024.
- `train_batch_size=64`, actor mini=16, `D=2`:
  - actor count is 4;
  - required multiple is 8;
  - after inherited V1 DP dispatch, each rank's row count is divisible by four.
- Actor `D=4, M=3`:
  - required multiple is 12;
  - after inherited V1 DP dispatch, each rank's row count is divisible by three.
- Invalid non-integral source-prompt mini-batch ratios fail before training.

### Primary regression test: real verl worker iteration

Call the installed verl `TrainingWorker.train_mini_batch` implementation with real
`TensorDict` data and its real DataLoader iterator. Use a minimal worker harness whose
`train_batch()` records calls at the optimizer boundary; all split calculation and
iteration remain the installed verl implementation.

Required cases:

| Worker metadata | Local rows | Expected `train_batch()` calls per epoch |
|---|---:|---:|
| `num_mini_batch=1` | 256 | 1 |
| `num_mini_batch=1` | 768 | 1 |
| `num_mini_batch=3` | 300 | 3 |
| `num_mini_batch=3` | 900 | 3 |

Add a control demonstrating the old behavior: with fixed `mini_batch_size`, increasing
local rows increases `train_batch()` calls. This test exercises the actual verl split
calculation, DataLoader iteration, and worker update loop; the stub stops exactly where a
real engine would perform the expensive forward/backward/optimizer operation.

### Thin trainer contract tests

These tests protect wiring but are not presented as proof of the optimizer-step fix:

- `_get_required_batch_multiple(dp_size)` returns
  `dp_size * configured_num_mini_batches`;
- a recording `actor_rollout_wg.update_actor` call observes `num_mini_batch=A` and no
  `mini_batch_size`;
- `global_batch_size` remains `ppo_mini_batch_size * rollout.n`;
- actor entropy, epochs, seed, shuffle, temperature, and metric-prefix behavior remain
  stock-compatible; distillation flags are false;
- critic, distillation, and unsupported loss aggregation each fail before worker
  initialization;
- importing the trainer registers `agentcore_sync` in a fresh process;
- the task-runner factory queues registration before returning the stock actor.

### End-to-end smoke tests

For the MigrationBench configuration (`D=2`, actor count 1):

- force or replay batches with 512, 513, 1,024, and 1,536 expanded rows;
- assert exactly one actor optimizer step per PPO epoch in every case;
- assert padding counts are 0, 1, 0, and 0 respectively;
- compare FSDP and Megatron worker behavior through logged counters;
- confirm all real row keys appear in the update and padding keys contribute zero loss.

For a synthetic `num_mini_batches=4` configuration:

- assert four actor steps per PPO epoch for several expanded row counts;
- assert each rank's local row count is divisible by four;
- do not require per-mini-batch DP token balance, matching stock V1.

The final smoke run on September 4, 2026 uses the thin launcher above with stock
`TaskRunnerV1`, Qwen3-4B, FSDP, eight GPUs, two real GSM8K rows, and one configured
mini-batch. It exited successfully after one ACR rollout/update step with:

- `batching/real_rows=2`;
- `batching/total_rows=8`;
- `batching/padding_rows=6`;
- `batching/required_multiple=8`;
- `batching/configured_optimizer_steps=1`.

The worker teardown emitted the same post-training `DataLoader worker ... killed` weakref
traceback seen in the earlier copied-task-runner smoke, after `Training Progress: 100%` and
the step metrics. The launcher process still exited zero and all GPUs were released; this
is existing verl teardown noise rather than a registration or batching failure.

## Observability

Add the following numeric metrics directly to the existing `metrics` dict inside the custom
`_update_actor(batch, metrics)` override:

- `batching/real_rows`
- `batching/total_rows`
- `batching/padding_rows`
- `batching/num_mini_batches`
- `batching/required_multiple`
- `batching/configured_optimizer_steps`
- the existing V1 whole-batch `global_seqlen/*` metrics

`_update_actor` is already part of the required compatibility override and receives both
the padded `batch` and the per-step `metrics` dict. It can count real versus synthetic rows
from `batch.tags`, read the current update plan saved by
`_get_required_batch_multiple(dp_size)`, and let V1's existing logger path publish the
values. Do not use `on_step_begin` / `on_step_end` or `_pending_sync_metrics`: those
lifecycle hooks do not receive the batch or metrics dict, and `_pending_sync_metrics` is
intended for weight-sync statistics.

`batching/configured_optimizer_steps` is the configured value
`num_mini_batches * ppo_epochs`, not a runtime count of optimizer calls. V1's worker does
not currently return `total_num_iterations`, so the metric must not be presented as proof
that the expected number of updates actually ran. Actual update count is verified by the
real-worker regression test and the Art/Migration smoke run; adding a production runtime
counter would require a separate worker-level change.

## Compatibility and rollout

- The new behavior is enabled by selecting `trainer.v1.trainer_mode=agentcore_sync` through
  the AgentCore launcher. Stock verl modes remain untouched.
- The confirmed Art scope is actor-only sync training with distillation disabled and
  `loss_agg_mode=seq-mean-token-sum`; unsupported configurations fail fast.
- Existing AgentCore recipes should switch immediately because variable trajectory rows are
  an inherent backend behavior, not an optional experiment.
- The implementation remains pinned to verl 0.9.0. A future pin bump must run the trainer
  contract tests, especially the copied `_update_actor` metadata path.

## Upstream plan

1. Implement and land the narrow `agentcore_sync` fix in Art/toolkit first.
2. Run a controlled MigrationBench before/after comparison with the same model, data,
   hyperparameters, rollout workload, and training budget. Report at least expanded-row
   counts, padding rows, optimizer steps, reward/quality curves, stability, and throughput.
3. Present the code path and empirical impact to verl so the optimizer-step coupling is
   visible as a practical multi-trajectory training issue rather than only a theoretical
   API concern.
4. Work with verl on a generic V1 fix covering the broader trainer surface that this local
   implementation intentionally rejects.
5. Once the upstream fix and any external-trainer discovery hook are available in the
   pinned verl version, adopt them and remove the local custom trainer, launcher, and
   copied `_update_actor` compatibility surface.
