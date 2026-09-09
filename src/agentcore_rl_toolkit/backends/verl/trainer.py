"""verl v1 sync trainer for variable-row AgentCore rollouts."""

from omegaconf import DictConfig, open_dict
from tensordict import TensorDict
from transfer_queue import KVBatchMeta
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo.utils import need_critic
from verl.trainer.ppo.v1 import PPOTrainerSync, register_trainer
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict


def _num_mini_batches(config: DictConfig) -> int:
    train_batch_size = config.data.train_batch_size
    ppo_mini_batch_size = config.actor_rollout_ref.actor.ppo_mini_batch_size
    if train_batch_size % ppo_mini_batch_size:
        raise ValueError("data.train_batch_size must be divisible by actor.ppo_mini_batch_size")
    return train_batch_size // ppo_mini_batch_size


def _validate_config(config: DictConfig) -> None:
    if not config.trainer.use_v1:
        raise ValueError("agentcore_sync requires trainer.use_v1=true")
    if need_critic(config):
        raise ValueError("agentcore_sync supports only actor-only training")
    if is_distillation_enabled(config.get("distillation")):
        raise ValueError("agentcore_sync does not support distillation")
    if config.actor_rollout_ref.actor.loss_agg_mode != "seq-mean-token-sum":
        raise ValueError("agentcore_sync requires loss_agg_mode=seq-mean-token-sum")

    _num_mini_batches(config)


@register_trainer("agentcore_sync")
class AgentCorePPOTrainerSync(PPOTrainerSync):
    """Sync trainer whose optimizer-step count is stable under row expansion."""

    def __init__(self, config: DictConfig):
        _validate_config(config)
        with open_dict(config):
            config.trainer.v1.trainer_mode = "sync"
        super().__init__(config)
        if self.parameter_sync_step != 1:
            raise ValueError("agentcore_sync requires parameter_sync_step=1")
        self._num_actor_mini_batches = _num_mini_batches(self.config)
        self._required_batch_multiple = self._num_actor_mini_batches

    def _get_required_batch_multiple(self, dp_size: int) -> int:
        self._required_batch_multiple = dp_size * self._num_actor_mini_batches
        return self._required_batch_multiple

    def _update_actor(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Mirror verl's actor-update path, using a fixed mini-batch count."""

        # Keep aligned with PPOTrainer._update_actor in verl 0.9.0.
        global_batch_size = (
            self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        )
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        extra_info = {
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
        expected_sessions = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
        metrics.update(
            {
                "batching/real_rows": total_rows - padding_rows,
                "batching/total_rows": total_rows,
                "batching/padding_rows": padding_rows,
                "batching/num_mini_batches": self._num_actor_mini_batches,
                "batching/required_multiple": self._required_batch_multiple,
                "batching/configured_optimizer_steps": (
                    self._num_actor_mini_batches * self.config.actor_rollout_ref.actor.ppo_epochs
                ),
                "training/rollout_failure/missing_sessions": expected_sessions - len(actual_sessions),
            }
        )
        return batch
