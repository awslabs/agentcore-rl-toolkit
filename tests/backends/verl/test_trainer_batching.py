import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_module

from agentcore_rl_toolkit.backends.verl.trainer import AgentCorePPOTrainerSync


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


class _ActorWorkerGroup:
    def update_actor(self, batch):
        return {"metrics": {"mfu": [0.5], "loss": [1.0]}}


def test_import_registers_agentcore_sync_in_fresh_process():
    code = """
from verl.trainer.ppo.v1 import get_trainer_cls
assert get_trainer_cls("agentcore_sync").__name__ == "AgentCorePPOTrainerSync"
"""
    env = {
        **os.environ,
        "VERL_USE_EXTERNAL_MODULES": "agentcore_rl_toolkit.backends.verl.trainer",
    }
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.parametrize("config_name", ["ppo_trainer", "ppo_megatron_trainer"])
def test_trainer_uses_sync_semantics_and_computes_required_batch_multiple(config_name):
    trainer = AgentCorePPOTrainerSync(_make_config(config_name=config_name))

    assert trainer.trainer_mode == "sync"
    assert trainer.parameter_sync_step == 1
    assert trainer._get_required_batch_multiple(dp_size=2) == 8


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
    assert metrics["batching/real_rows"] == 3
    assert metrics["batching/total_rows"] == 4
    assert metrics["batching/padding_rows"] == 1
    assert metrics["training/rollout_failure/missing_sessions"] == 6


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
