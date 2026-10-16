"""verl's three v1 PPO backends plus this package's mixins, one registered name each.

Every AgentCore recipe wants the same three mixins -- its agent loop reports metrics
through ``AgentLoopOutput.extra_fields``, produces GRPO groups that can collapse, and may
emit a variable number of training rows per rollout -- so the registrations live here
rather than in one recipe. Name this module in ``VERL_USE_EXTERNAL_MODULES`` to make the
``agentcore_*`` trainer modes available, in the driver and on every node.

Must stay importable by any verl worker: no driver-only imports at module scope.
"""

import ray
from omegaconf import DictConfig, OmegaConf, open_dict
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.trainer.ppo.v1 import (
    PPOTrainerColocateAsync,
    PPOTrainerSeparateAsync,
    PPOTrainerSync,
    register_trainer,
)

from agentcore_rl_toolkit.concurrency.rate_limiter import LocalRateLimiter
from agentcore_rl_toolkit.concurrency.ray_adapters import make_priority_semaphore

from .trainer_mixins import (
    AdvantageZeroMetricsMixin,
    AgentLoopMetricsMixin,
    RoundRobinAgentLoopDispatchMixin,
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

    def init(self):
        path = self.config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        budgets = set()
        if path:
            for entry in OmegaConf.load(resolve_config_path(path)):
                if entry.get("_target_") != "agentcore_rl_toolkit.backends.verl.agent_loop.AgentCoreAgentLoop":
                    continue
                limits = (entry.get("rollout_concurrency", 128), entry.get("tps_limit", 5))
                if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in limits):
                    raise ValueError("rollout_concurrency and tps_limit must be positive integers")
                budgets.add(limits)
        if len(budgets) > 1:
            raise ValueError("AgentCore agent loops must agree on the training job's concurrency and TPS limits")

        # Retain the shared actors before verl starts the agent-loop workers.
        self._agentcore_rollout_resources = []
        for concurrency, tps in budgets:
            self._agentcore_rollout_resources = [
                make_priority_semaphore("rollout_concurrency_semaphore", concurrency, 10_000),
                ray.remote(LocalRateLimiter)
                .options(name="session_create_rate_limiter", num_cpus=0, get_if_exists=True)
                .remote(tps_limit=tps),
            ]
        super().init()


# One registered name per verl backend. VariableRowBatchingMixin comes last of the mixins so
# it stays closest to the trainer whose batching seams it overrides.
@register_trainer("agentcore_sync")
class AgentCorePPOTrainerSync(
    _AgentCoreTrainerBase,
    RoundRobinAgentLoopDispatchMixin,
    AgentLoopMetricsMixin,
    AdvantageZeroMetricsMixin,
    VariableRowBatchingMixin,
    PPOTrainerSync,
):
    _verl_trainer_mode = "sync"


@register_trainer("agentcore_colocate_async")
class AgentCorePPOTrainerColocateAsync(
    _AgentCoreTrainerBase,
    RoundRobinAgentLoopDispatchMixin,
    AgentLoopMetricsMixin,
    AdvantageZeroMetricsMixin,
    VariableRowBatchingMixin,
    PPOTrainerColocateAsync,
):
    _verl_trainer_mode = "colocate_async"


@register_trainer("agentcore_separate_async")
class AgentCorePPOTrainerSeparateAsync(
    _AgentCoreTrainerBase,
    RoundRobinAgentLoopDispatchMixin,
    AgentLoopMetricsMixin,
    AdvantageZeroMetricsMixin,
    VariableRowBatchingMixin,
    PPOTrainerSeparateAsync,
):
    _verl_trainer_mode = "separate_async"
