# Stable optimizer-step batching for variable-row verl rollouts

| Field | Value |
| --- | --- |
| Status | Accepted |
| Implementation | Shipped |
| Date | 2026-09-10 |
| Pull request | [#131](https://github.com/awslabs/agentcore-rl-toolkit/pull/131) |

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

The implementation makes optimizer-step count depend on the configured source-prompt batch,
while still training every emitted row:

```
num_actor_mini_batches
    = data.train_batch_size / actor.ppo_mini_batch_size

actor optimizer steps
    = num_actor_mini_batches * actor.ppo_epochs
```

The worker already exposes the right interface: `TrainingWorker.train_mini_batch` accepts
`num_mini_batch` as an alternative to a fixed `mini_batch_size`. The custom trainer uses
that interface, pads only enough for the configured number of actor optimizer partitions
and actor data-parallel ranks, and otherwise preserves V1's whole-batch DP balancing.

The integration uses verl's public trainer registry and subclasses `PPOTrainerSync`.
The recipes set verl's official `VERL_USE_EXTERNAL_MODULES` variable before invoking
`python -m verl.trainer.main_ppo`, so both the driver and inherited Ray task-runner
actor import the registration module before their process-local trainer lookup.

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
- Fail early on configurations outside the tested contract instead of partially
  supporting them.

## Non-goals

- Keeping every row from one rollout/session in the same optimizer step. The current loss is
  additive over rows; crossing an optimizer boundary changes ordinary SGD/Adam ordering but
  is not a correctness violation.
- Changing how branch rows are weighted. More branches still contribute more additive loss
  mass, as they do today.
- Changing GRPO advantage grouping. verl computes and broadcasts multi-trajectory
  advantages before the actor update; this implementation only changes update partitioning.
- Changing dynamic micro-batching. `use_dynamic_bsz` decides how one optimizer partition is
  split into forward/backward micro-batches; it does not decide how many optimizer steps
  occur.
- Supporting async trainer modes. The AgentCore example scripts use `PPOTrainerSync`.
- Supporting a critic. The AgentCore example scripts use actor-only GRPO.
- Supporting on-policy distillation.
- Restoring legacy V0's per-mini-batch DP token balancing. V0 implements
  `_balance_batch(..., keep_minibatch=True)`; V1 retains the argument but ignores it and
  balances only the whole batch across DP ranks. This implementation preserves the
  V1 behavior.

## Terminology

- **source prompt** — one item counted by `data.train_batch_size`.
- **nominal rollout** — one of the `rollout.n` samples generated for a source prompt.
- **expanded row** — one training row emitted by a nominal rollout. A rollout tree may emit
  several.
- **optimizer partition** — the global row chunk consumed by one optimizer step, before it
  is split over DP ranks.

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
`dp_size` contiguous rank chunks. The custom trainer does not change that behavior.

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

### 2. Keep the supported contract actor-only

The generic actor/critic case requires coordinating two potentially different DP sizes and
two mini-batch schedules. It is outside this trainer's supported contract.

The custom trainer calls verl's `need_critic(config)` during construction and rejects
the configuration if it returns true:

```python
if need_critic(config):
    raise ValueError("agentcore_sync supports only actor-only training")
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

This is the minimum size that gives every `(DP rank, optimizer partition)` pair
the same number of rows.

Examples:

| Case | Real rows | Old required multiple | New required multiple | Padded rows |
|---|---:|---:|---:|---:|
| Migration recipe, `D=2`, `M=1` | 513 | 512 | 2 | 514 |
| `D=2`, actor `M=4` | 270 | depends on `rollout.n` | 8 | 272 |
| `D=4`, actor `M=3` | 601 | depends on `rollout.n` | 12 | 612 |

The existing verl `upsample_batch_to_divisible_size` helper remains suitable: its synthetic
rows have a two-token `[EOS, EOS]` shape, zero response/loss masks, zero rewards/logprobs,
fresh padding UIDs, and `is_padding=True`. Only the requested multiple changes.

Efficient padding and stable optimizer-step count are logically separable, but both are
implemented here because they derive from the same schedule. Applying only the
`num_mini_batch` change would leave unnecessary padding; applying only the padding
change would make the fixed-size worker iterator fail or continue producing a
variable step count.

### 4. Preserve V1's existing whole-batch DP balancing

The subclass changes `_get_required_batch_multiple(dp_size)` to return `D * M`, then calls
the inherited V1 `_balance_batch` unchanged. The inherited implementation:

1. pad the expanded batch to a multiple of `D * M`;
2. token-balance the whole padded batch into `D` equal-cardinality contiguous rank chunks;
3. let DP dispatch send one chunk to each rank.

Each rank therefore receives `N / D` rows. The worker receives
`num_mini_batch=M` and derives its local mini-batch size:

```
local_mini_batch_size = (N / D) / M = N / (D * M)
```

Because the padded row count is divisible by `D * M`, V1 can first split the rows
evenly across `D` ranks, and each worker can then split its local rows evenly into
`M` mini-batches.

For `M > 1`, individual local mini-batches may have uneven token workloads even though each
rank's whole chunk is balanced. That matches current V1 behavior and affects performance,
not optimizer-step correctness. Restoring V0's `keep_minibatch=True` behavior is left to a
generic verl change.

**Limitation when `M > 1`.** Whole-batch balancing treats expanded rows independently
and does not preserve rollout-session boundaries. With `M=1`, all rows are in the same
optimizer partition and, with `seq-mean-token-sum`, their additive token losses are
accumulated before the optimizer update. With `M > 1`, rows from the same rollout session
may land in different optimizer partitions and therefore be evaluated at different
parameter states. The trainer still guarantees the configured optimizer-step count and
additive loss weighting within each step, but it is not equivalent to aggregating all rows
from a session before one update. Exact equivalence for `M > 1` would require session-aware
optimizer partitioning, which is outside this design.

### 5. Select `num_mini_batch`, not `mini_batch_size`

For actor updates, the custom trainer passes:

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

The other verl aggregation modes change this normalization or weighting:

- `token-mean` divides by the actual token count after row expansion, so adding a branch
  also changes the weight of existing tokens;
- `seq-mean-token-mean` first divides each row by its own token count, so branch length
  changes its relative contribution;
- `token-sum` omits the nominal `global_batch_size` denominator; and
- `seq-mean-token-sum-norm` introduces an additional scale factor.

This implementation changes optimizer-step scheduling, not objective weighting. It
preserves the nominal `global_batch_size` denominator and therefore preserves the current
additive branch-row semantics: a rollout that emits more real branch rows contributes more
aggregate loss mass. Normalizing by actual expanded rows or first reducing rows per
rollout/session would define a different objective and belongs in a separate design and
change.

### 6. Fail fast outside the validated contract

`AgentCorePPOTrainerSync.__init__` validates the supported surface before
initializing the base trainer:

```python
def validate_agentcore_sync_config(config) -> None:
    if not config.trainer.use_v1:
        raise ValueError("agentcore_sync requires trainer.use_v1=true")
    if need_critic(config):
        raise ValueError("agentcore_sync supports only actor-only training")
    if is_distillation_enabled(config.get("distillation")):
        raise ValueError("agentcore_sync does not support distillation")
    if config.actor_rollout_ref.actor.loss_agg_mode != "seq-mean-token-sum":
        raise ValueError("agentcore_sync requires loss_agg_mode=seq-mean-token-sum")

    train_batch_size = config.data.train_batch_size
    ppo_mini_batch_size = config.actor_rollout_ref.actor.ppo_mini_batch_size
    if train_batch_size % ppo_mini_batch_size:
        raise ValueError(
            "data.train_batch_size must be divisible by actor.ppo_mini_batch_size"
        )
```

After the custom registry alias is normalized to `"sync"` semantics, the trainer
also requires `parameter_sync_step == 1`. The schedule is defined per synchronous
`_step_once`; multi-trigger asynchronous schedules are outside this contract.

These checks match the two end-to-end-tested AgentCore example scripts.
They do not reject unrelated actor features such as FSDP versus Megatron, LoRA, reference
policy/KL, rollout correction, dynamic micro-batching, or `ppo_epochs >= 1`.

## verl customization and integration cut

### Public interface to use: trainer registry

verl's registry accepts any `PPOTrainer` subclass, but the concrete lifecycle contract is
mode-specific. `AgentCorePPOTrainerSync` subclasses and registers `PPOTrainerSync`:

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
semantics remain synchronous.

### Trainer registration through verl external modules

`TaskRunnerV1.run()` performs:

```python
trainer_cls = get_trainer_cls(config.trainer.v1.trainer_mode)
```

before it loads `agent_loop_manager_class` or any AgentCore-owned class. The trainer module
must therefore be imported before that lookup. verl 0.9.0 provides the
`VERL_USE_EXTERNAL_MODULES` environment variable for this purpose:

```bash
export VERL_USE_EXTERNAL_MODULES=agentcore_rl_toolkit.backends.verl.trainer
python -m verl.trainer.main_ppo trainer.v1.trainer_mode=agentcore_sync
```

The full import/registration chain is:

1. The recipe exports `VERL_USE_EXTERNAL_MODULES` before starting verl.
2. verl initialization imports `trainer.py`; its `@register_trainer("agentcore_sync")`
   decorator updates the registry in that process.
3. The locally spawned Ray task-runner actor inherits the environment and performs the
   same external-module import during its own verl initialization.
4. `TaskRunnerV1.run()` performs the lookup and owns initialization, TransferQueue,
   fitting, logger shutdown, and queue shutdown.

The registry remains process-local, but the official environment-based module discovery
runs in every inherited process that imports verl. No custom launcher, task-runner factory,
Ray developer API, or venv patch is required.

### Why not use `agent_loop_manager_class`

`agent_loop_manager_class` is a public FQN hook, but it is instantiated only after
`trainer.init()`. It could mutate the already-created trainer class before `fit()`, but that
would be a timing-dependent monkey patch unrelated to the manager's rollout responsibility.
The trainer registry expresses the ownership correctly and is easier to test.

### Trainer override surface

The custom subclass overrides only:

- `_get_required_batch_multiple(dp_size)` — return
  `dp_size * actor_num_mini_batches`;
- `_update_actor(...)` — preserve verl's actor metadata and metrics while replacing
  `mini_batch_size` with `num_mini_batch`.

The inherited `_balance_batch` remains unchanged. It discovers actor DP through verl's
existing `"actor"` mesh dispatch metadata, calls the overridden required-multiple method,
pads to `D * M`, and performs V1's normal whole-batch DP balancing.

Since distillation is explicitly unsupported, `_update_actor` preserves verl's
entropy, epoch, seed, temperature, metric-reduction, and worker-call behavior while setting
the two distillation booleans to false. verl 0.9.0 does not expose a hook for constructing
update `extra_info`, so this small override is the unavoidable compatibility surface. The
repository pins `verl==0.9.0`. When that pin changes, the copied metadata path must
be compared manually with upstream `_update_actor` before running the contract
tests.

No worker, engine, dispatcher, TransferQueue, advantage, or loss implementation is
replaced.

## Implementation surface

- `src/agentcore_rl_toolkit/backends/verl/trainer.py` contains
  `AgentCorePPOTrainerSync`, configuration validation, the batching overrides,
  metrics, and external-module registration.
- `tests/backends/verl/test_training_worker_batching.py` exercises the installed
  verl worker's real mini-batch iterator.
- `tests/backends/verl/test_trainer_batching.py` covers trainer metadata,
  fail-fast configuration, metrics, and fresh-process registration.
- The FSDP and Megatron example scripts export `VERL_USE_EXTERNAL_MODULES` and
  select `trainer.v1.trainer_mode=agentcore_sync`.
- The backend README and public setup guide document the supported contract.

The implementation does not replace the rollout gateway, agent-loop behavior,
TransferQueue, worker engines, advantage computation, or loss implementation.

## Validation

Automated coverage lives in:

- [`test_training_worker_batching.py`](../tests/backends/verl/test_training_worker_batching.py),
  which exercises the installed verl worker's mini-batch iteration across
  different row counts and epochs;
- [`test_trainer_batching.py`](../tests/backends/verl/test_trainer_batching.py),
  which covers trainer registration, configuration constraints, batching
  metadata, and dynamic metrics.

### End-to-end validation

An eight-GPU Qwen3-4B FSDP smoke test completed one ACR rollout and actor update
with two real rows padded to the required multiple of eight:

- `batching/real_rows=2`;
- `batching/total_rows=8`;
- `batching/padding_rows=6`.

A second smoke test used `python -m verl.trainer.main_ppo` with
`VERL_USE_EXTERNAL_MODULES`; `TaskRunnerV1` resolved `agentcore_sync` and
completed an actor update.

Full end-to-end validation then completed:

- a 116-step Qwen3-4B GSM8K run through the FSDP path;
- a 101-step Qwen3-Coder-30B-A3B MigrationBench run through the Megatron + LoRA
  path.

## Observability

The `_update_actor(batch, metrics)` override adds:

- `batching/real_rows`
- `batching/total_rows`
- `batching/padding_rows`
- `training/rollout_failure/missing_sessions`

The inherited trainer continues to publish the whole-batch `global_seqlen/*`
metrics.

## Compatibility

- The behavior is enabled by selecting
  `trainer.v1.trainer_mode=agentcore_sync` through
  `VERL_USE_EXTERNAL_MODULES`; verl's built-in trainer modes are unchanged.
- The supported contract is actor-only synchronous training with distillation
  disabled, `parameter_sync_step=1`, and
  `loss_agg_mode=seq-mean-token-sum`.
- The implementation targets `verl==0.9.0`. A version bump must run the trainer
  contract tests, especially the copied `_update_actor` metadata path.
