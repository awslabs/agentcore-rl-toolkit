"""verl's three v1 PPO backends plus this package's mixins, one registered name each.

Every AgentCore recipe wants the same three mixins -- its agent loop reports metrics
through ``AgentLoopOutput.extra_fields``, produces GRPO groups that can collapse, and may
emit a variable number of training rows per rollout -- so the registrations live here
rather than in one recipe. Name this module in ``VERL_USE_EXTERNAL_MODULES`` to make the
``agentcore_*`` trainer modes available, in the driver and on every node.

Must stay importable by any verl worker: no driver-only imports at module scope.
"""

from omegaconf import DictConfig, open_dict
from verl.trainer.ppo.v1 import (
    PPOTrainerColocateAsync,
    PPOTrainerSeparateAsync,
    PPOTrainerSync,
    register_trainer,
)

from .trainer_mixins import (
    AdvantageZeroMetricsMixin,
    AgentLoopMetricsMixin,
    VariableRowBatchingMixin,
)

__all__ = [
    "AgentCorePPOTrainerColocateAsync",
    "AgentCorePPOTrainerSeparateAsync",
    "AgentCorePPOTrainerSync",
]


class _AgentCoreTrainerBase:
    """Normalizes the registry alias back to the verl trainer mode it wraps.

    An ``agentcore_*`` name is only a lookup key for ``register_trainer``, but
    ``PPOTrainer`` compares ``trainer.v1.trainer_mode`` *literally* to pick ``ReplayBuffer``
    over ``ReplayBufferAsync``, exact-refill behavior, async prompt persistence, TransferQueue
    checkpointing, and the ``trainer.v1.<mode>`` config node ``parameter_sync_step`` is read
    from. So each subclass declares which verl mode it actually is and writes it back before
    the base trainer initializes.
    """

    # Set by each concrete trainer below.
    _verl_trainer_mode: str

    def __init__(self, config: DictConfig):
        with open_dict(config):
            config.trainer.v1.trainer_mode = self._verl_trainer_mode
        super().__init__(config)


# One registered name per verl backend. VariableRowBatchingMixin comes last of the mixins so
# it stays closest to the trainer whose batching seams it overrides.
@register_trainer("agentcore_sync")
class AgentCorePPOTrainerSync(
    _AgentCoreTrainerBase,
    AgentLoopMetricsMixin,
    AdvantageZeroMetricsMixin,
    VariableRowBatchingMixin,
    PPOTrainerSync,
):
    _verl_trainer_mode = "sync"


@register_trainer("agentcore_colocate_async")
class AgentCorePPOTrainerColocateAsync(
    _AgentCoreTrainerBase,
    AgentLoopMetricsMixin,
    AdvantageZeroMetricsMixin,
    VariableRowBatchingMixin,
    PPOTrainerColocateAsync,
):
    _verl_trainer_mode = "colocate_async"


@register_trainer("agentcore_separate_async")
class AgentCorePPOTrainerSeparateAsync(
    _AgentCoreTrainerBase,
    AgentLoopMetricsMixin,
    AdvantageZeroMetricsMixin,
    VariableRowBatchingMixin,
    PPOTrainerSeparateAsync,
):
    _verl_trainer_mode = "separate_async"
