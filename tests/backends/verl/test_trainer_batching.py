import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_module
from verl.trainer.ppo.v1 import get_trainer_cls
from verl.trainer.ppo.v1.utils import MetricsAggregator

from agentcore_rl_toolkit.backends.verl import trainer as trainer_module
from agentcore_rl_toolkit.backends.verl.trainer import AgentCorePPOTrainerSync

# The registered `agentcore_*` mode, the verl mode it normalizes to, and the extra config
# each verl trainer's own __init__ demands.
TRAINER_MODES = {
    "agentcore_sync": ("sync", ()),
    "agentcore_colocate_async": ("colocate_async", ()),
    "agentcore_separate_async": (
        "separate_async",
        (
            "actor_rollout_ref.rollout.nnodes=1",
            "actor_rollout_ref.rollout.n_gpus_per_node=8",
            "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl",
        ),
    ),
}
COLOCATED_MODES = ["agentcore_sync", "agentcore_colocate_async"]


def _make_config(*overrides, config_name="ppo_trainer"):
    base_overrides = [
        "trainer.v1.trainer_mode=agentcore_sync",
        "algorithm.adv_estimator=grpo",
        "critic.enable=false",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum",
        "data.train_batch_size=64",
        "actor_rollout_ref.actor.ppo_mini_batch_size=16",
    ]
    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        return compose(config_name=config_name, overrides=[*base_overrides, *overrides])


def _trainer(mode, *overrides):
    """Build the trainer registered under `mode`, with that mode's mandatory config."""
    _, mode_overrides = TRAINER_MODES[mode]
    return get_trainer_cls(mode)(_make_config(f"trainer.v1.trainer_mode={mode}", *mode_overrides, *overrides))


class _ActorWorkerGroup:
    def update_actor(self, batch):
        return {"metrics": {"mfu": [0.5], "loss": [1.0]}}


def test_import_registers_every_agentcore_mode_in_a_fresh_process():
    code = """
from verl.trainer.ppo.v1 import get_trainer_cls
assert get_trainer_cls("agentcore_sync").__name__ == "AgentCorePPOTrainerSync"
assert get_trainer_cls("agentcore_colocate_async").__name__ == "AgentCorePPOTrainerColocateAsync"
assert get_trainer_cls("agentcore_separate_async").__name__ == "AgentCorePPOTrainerSeparateAsync"
"""
    env = {
        **os.environ,
        "VERL_USE_EXTERNAL_MODULES": "agentcore_rl_toolkit.backends.verl.trainer",
    }
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.parametrize("mode", list(TRAINER_MODES))
def test_registry_alias_is_normalized_to_the_verl_trainer_mode(mode):
    """The base trainer branches on the literal mode string, so it must not see the alias."""
    verl_mode, _ = TRAINER_MODES[mode]

    assert _trainer(mode).trainer_mode == verl_mode


@pytest.mark.parametrize("config_name", ["ppo_trainer", "ppo_megatron_trainer"])
def test_trainer_uses_sync_semantics_and_computes_required_batch_multiple(config_name):
    trainer = AgentCorePPOTrainerSync(_make_config(config_name=config_name))

    assert trainer.trainer_mode == "sync"
    assert trainer.parameter_sync_step == 1
    assert trainer._get_required_batch_multiple(dp_size=2) == 8


@pytest.mark.parametrize("mode", COLOCATED_MODES)
def test_colocated_modes_fix_the_configured_partition_count(mode):
    """train_batch_size=64 / ppo_mini_batch_size=16 => 4 partitions per update trigger."""
    trainer = _trainer(mode)

    assert trainer.parameter_sync_step == 1
    assert trainer._num_actor_mini_batches == 4
    assert trainer._get_required_batch_multiple(dp_size=2) == 8


def test_separate_async_counts_partitions_per_sync_trigger():
    """Decoupled PPO spends the updates as triggers, so each trigger has one partition."""
    trainer = _trainer(
        "agentcore_separate_async",
        "data.train_batch_size=32",
        "trainer.v1.separate_async.parameter_sync_step=2",
    )

    assert trainer.parameter_sync_step == 2
    assert trainer._num_actor_mini_batches == 1
    # Only DP alignment is left: no padding to a fixed mini-batch row count.
    assert trainer._get_required_batch_multiple(dp_size=4) == 4


def test_update_actor_sends_count_not_fixed_mini_batch_size_and_records_metrics():
    trainer = AgentCorePPOTrainerSync(
        _make_config(
            "data.train_batch_size=2",
            "actor_rollout_ref.actor.ppo_mini_batch_size=1",
            "actor_rollout_ref.actor.ppo_epochs=2",
            "actor_rollout_ref.rollout.n=4",
        )
    )
    trainer.actor_rollout_wg = _ActorWorkerGroup()
    batch = SimpleNamespace(
        extra_info={},
        keys=[
            "uid0_0_0",
            "uid0_0_1",
            "pad_0_0",
            "uid1_2_0",
        ],
        tags=[
            {"is_padding": False},
            {},
            {"is_padding": True},
            {"is_padding": False},
        ],
    )
    metrics = {}

    assert trainer._update_actor(batch, metrics) is batch

    assert batch.extra_info["num_mini_batch"] == 2
    assert "mini_batch_size" not in batch.extra_info
    assert batch.extra_info["global_batch_size"] == 4
    assert batch.extra_info["epochs"] == 2
    assert metrics["perf/mfu/actor"] == 0.5
    assert metrics["batching/total_real_rows"] == 3
    assert metrics["batching/total_rows"] == 4
    assert metrics["batching/total_padding_rows"] == 1
    assert metrics["training/rollout_failure/total_missing_sessions"] == 6


@pytest.mark.parametrize("mode", list(TRAINER_MODES))
def test_every_mode_sends_partition_count_and_records_batching_metrics(mode):
    trainer = _trainer(
        mode,
        "data.train_batch_size=16",
        "actor_rollout_ref.actor.ppo_mini_batch_size=16",
        "actor_rollout_ref.rollout.n=4",
        # verl's separate-async default is 4 triggers; one keeps every mode comparable.
        "trainer.v1.separate_async.parameter_sync_step=1",
    )
    trainer.actor_rollout_wg = _ActorWorkerGroup()
    batch = SimpleNamespace(
        extra_info={},
        keys=["uid0_0_0", "uid0_0_1", "pad_0_0"],
        tags=[{"is_padding": False}, {}, {"is_padding": True}],
    )
    metrics = {}

    assert trainer._update_actor(batch, metrics) is batch

    assert batch.extra_info["num_mini_batch"] == 1
    assert "mini_batch_size" not in batch.extra_info
    assert batch.extra_info["global_batch_size"] == 64
    assert metrics["batching/total_real_rows"] == 2
    assert metrics["batching/total_padding_rows"] == 1
    # 16 prompts * n=4 nominal rollouts expected, one distinct session seen.
    assert metrics["training/rollout_failure/total_missing_sessions"] == 63


def test_batching_counters_are_summed_across_sync_triggers():
    """A step can hold several triggers; verl reduces by metric name, and only names
    containing "sum"/"total" are summed instead of sample-weighted-averaged."""
    trainer = AgentCorePPOTrainerSync(_make_config("actor_rollout_ref.rollout.n=1"))
    trainer.actor_rollout_wg = _ActorWorkerGroup()
    aggregator = MetricsAggregator()
    for trigger in range(2):
        metrics = {}
        trainer._update_actor(
            SimpleNamespace(
                extra_info={},
                keys=[f"uid{trigger}_0_0", "pad_0_0"],
                tags=[{"is_padding": False}, {"is_padding": True}],
            ),
            metrics,
        )
        aggregator.add_step_metrics(metrics, sample_count=2)

    aggregated = aggregator.get_aggregated_metrics()

    assert aggregated["batching/total_real_rows"] == 2
    assert aggregated["batching/total_rows"] == 4
    assert aggregated["batching/total_padding_rows"] == 2
    assert aggregated["training/rollout_failure/total_missing_sessions"] == 126


@pytest.mark.parametrize("mode", list(TRAINER_MODES))
def test_every_mode_layers_the_metric_mixins(mode):
    cls = type(_trainer(mode))

    assert issubclass(cls, trainer_module.AgentLoopMetricsMixin)
    assert issubclass(cls, trainer_module.AdvantageZeroMetricsMixin)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("trainer.use_v1=false", "trainer.use_v1=true"),
        ("critic.enable=true", "actor-only"),
        ("distillation.enabled=true", "distillation"),
        ("actor_rollout_ref.actor.loss_agg_mode=token-mean", "seq-mean-token-sum"),
        ("data.train_batch_size=65", "must be divisible"),
        ("++trainer.v1.sync.parameter_sync_step=2", "parameter_sync_step=1"),
    ],
)
def test_unsupported_configurations_fail_fast(override, message):
    with pytest.raises(ValueError, match=message):
        AgentCorePPOTrainerSync(_make_config(override))


@pytest.mark.parametrize("mode", list(TRAINER_MODES))
def test_every_mode_rejects_an_unsupported_loss_aggregation(mode):
    with pytest.raises(ValueError, match="seq-mean-token-sum"):
        _trainer(mode, "actor_rollout_ref.actor.loss_agg_mode=token-mean")


def test_colocated_modes_reject_multi_trigger_weight_sync():
    """Their engines sleep for the whole training pass, so a second trigger would hang."""
    with pytest.raises(ValueError, match="parameter_sync_step=1"):
        _trainer("agentcore_colocate_async", "++trainer.v1.colocate_async.parameter_sync_step=2")
