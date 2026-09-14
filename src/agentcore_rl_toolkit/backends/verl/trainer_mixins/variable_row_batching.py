"""Stable actor optimizer-step batching for variable-row AgentCore rollouts.

An agentic rollout can emit a variable number of training rows (context compaction,
trajectory forks, sub-agents). verl v1 partitions the actor update by a fixed *row count*,
so the emitted-row count silently changes how many optimizer steps one training batch
takes. This mixin fixes the partition *count* instead and lets the partition size float,
keeping the configured schedule stable while still training every emitted row. See
``designs/verl_variable_trajectory_batching.md`` for the derivation.

The mixin is trainer-mode agnostic. verl's ``PPOTrainer.step`` splits a training batch into
``parameter_sync_step`` sample->update triggers and calls ``_update_actor`` once per
trigger, so the partition count is derived from the batch *one trigger* consumes:

    num_mini_batch = data.train_batch_size / parameter_sync_step / actor.ppo_mini_batch_size

That reduces to ``train_batch_size / ppo_mini_batch_size`` for the synchronous and
colocate-async trainers (which this mixin holds to ``parameter_sync_step == 1``) and to
``1`` for the separate-async trainer, which requires
``train_batch_size == parameter_sync_step * ppo_mini_batch_size``.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig
from tensordict import TensorDict
from transfer_queue import KVBatchMeta
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo.utils import need_critic
from verl.trainer.ppo.v1 import PPOTrainerSeparateAsync
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict

from .base import TrainerMixinBase


def _validate_config(config: DictConfig, name: str) -> None:
    """Reject configurations outside the contract, before the base trainer initializes."""
    if not config.trainer.use_v1:
        raise ValueError(f"{name} requires trainer.use_v1=true")
    if need_critic(config):
        raise ValueError(f"{name} supports only actor-only training")
    if is_distillation_enabled(config.get("distillation")):
        raise ValueError(f"{name} does not support distillation")
    # Keep expanded rows additive: sum their token losses, but normalize by the
    # configured pre-expansion mini-batch size rather than the expanded row or
    # token count. See the variable-row batching contract in README.md.
    if config.actor_rollout_ref.actor.loss_agg_mode != "seq-mean-token-sum":
        raise ValueError(f"{name} requires loss_agg_mode=seq-mean-token-sum")


def _num_mini_batches(config: DictConfig, parameter_sync_step: int, name: str) -> int:
    """Actor optimizer partitions per ``_update_actor`` call, i.e. per sync trigger."""
    train_batch_size = config.data.train_batch_size
    ppo_mini_batch_size = config.actor_rollout_ref.actor.ppo_mini_batch_size
    if train_batch_size % parameter_sync_step:
        raise ValueError(f"{name} requires data.train_batch_size divisible by parameter_sync_step")
    sample_batch_size = train_batch_size // parameter_sync_step
    if sample_batch_size % ppo_mini_batch_size:
        raise ValueError(
            f"{name} requires data.train_batch_size / parameter_sync_step "
            "must be divisible by actor.ppo_mini_batch_size"
        )
    return sample_batch_size // ppo_mini_batch_size


class VariableRowBatchingMixin(TrainerMixinBase):
    """Keeps the actor optimizer-step count stable under rollout row expansion.

    Layer it onto any verl v1 PPO trainer, ahead of the trainer class in the bases.
    """

    def __init__(self, config: DictConfig):
        name = type(self).__name__
        _validate_config(config, name)
        super().__init__(config)
        # Every trainer but separate-async sleeps its rollout engines in ``on_sample_end``
        # for the whole training pass, so a second sample->update trigger would wait
        # forever for generation that can no longer happen.
        if not isinstance(self, PPOTrainerSeparateAsync) and self.parameter_sync_step != 1:
            raise ValueError(f"{name} requires parameter_sync_step=1")
        self._num_actor_mini_batches = _num_mini_batches(self.config, self.parameter_sync_step, name)

    def _get_required_batch_multiple(self, dp_size: int) -> int:
        # Pad only for the DP ranks and the configured optimizer partitions, not for a
        # fixed row count per partition.
        return dp_size * self._num_actor_mini_batches

    def _update_actor(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Mirror verl's actor-update path, using a fixed mini-batch count."""

        # Keep aligned with PPOTrainer._update_actor in verl 0.9.0.
        global_batch_size = (
            self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        )
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        extra_info: dict[str, Any] = {
            "calculate_entropy": calculate_entropy,
            "distillation_use_topk": False,
            "distillation_only": False,
            "global_batch_size": global_batch_size,
            "epochs": self.config.actor_rollout_ref.actor.ppo_epochs,
            "seed": self.config.actor_rollout_ref.actor.data_loader_seed,
            "dataloader_kwargs": {"shuffle": self.config.actor_rollout_ref.actor.shuffle},
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        }

        # AgentCore change: fix the number of optimizer partitions instead of
        # fixing their row size, so rollout row expansion does not add steps.
        extra_info["num_mini_batch"] = self._num_actor_mini_batches
        batch.extra_info.update(extra_info)

        # Keep aligned with verl's worker call and metric handling.
        output: TensorDict = self.actor_rollout_wg.update_actor(batch)
        output = rename_dict(output["metrics"], "actor/")
        output["perf/mfu/actor"] = output.pop("actor/mfu")
        metrics.update(reduce_metrics(output))

        # AgentCore-specific update metrics.
        total_rows = len(batch.tags)
        padding_rows = sum(tag.get("is_padding", False) for tag in batch.tags)
        # ReplayBuffer keys are {uid}_{session_id}_{trajectory_index}; multiple
        # trajectory rows from one rollout session count once.
        actual_sessions = {
            tuple(key.rsplit("_", 2)[:2])
            for key, tag in zip(batch.keys, batch.tags, strict=True)
            if not tag.get("is_padding", False)
        }
        # One trigger consumes train_batch_size / parameter_sync_step source prompts.
        expected_sessions = (
            self.config.data.train_batch_size // self.parameter_sync_step * self.config.actor_rollout_ref.rollout.n
        )
        # These are per-trigger counts, and a step can hold several triggers
        # (separate-async with parameter_sync_step > 1). verl reduces a step's metrics by
        # name: a key containing "sum" or "total" is summed, anything else is
        # sample-weighted-averaged. Name every counter "total_*" so all four aggregate the
        # same way and stay comparable across modes.
        metrics.update(
            {
                "batching/total_real_rows": total_rows - padding_rows,
                "batching/total_rows": total_rows,
                "batching/total_padding_rows": padding_rows,
                "training/rollout_failure/total_missing_sessions": expected_sessions - len(actual_sessions),
            }
        )
        return batch
