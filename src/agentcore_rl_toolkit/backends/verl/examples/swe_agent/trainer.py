"""This recipe's PPO trainers: verl's v1 backends plus the metric mixins we want.

Must stay importable by any verl worker: no driver-only imports at module scope.
"""

from verl.trainer.ppo.v1 import (
    PPOTrainerColocateAsync,
    PPOTrainerSeparateAsync,
    PPOTrainerSync,
    register_trainer,
)

from agentcore_rl_toolkit.backends.experimental.verl.trainer_mixins import (
    AdvantageZeroMetricsMixin,
    AgentLoopMetricsMixin,
)

__all__ = ["ARTColocateAsync", "ARTSeparateAsync", "ARTSync"]


# One registered name per verl backend.
@register_trainer("art_sync")
class ARTSync(
    AgentLoopMetricsMixin,
    AdvantageZeroMetricsMixin,
    PPOTrainerSync,
):
    pass


@register_trainer("art_colocate_async")
class ARTColocateAsync(
    AgentLoopMetricsMixin,
    AdvantageZeroMetricsMixin,
    PPOTrainerColocateAsync,
):
    pass


@register_trainer("art_separate_async")
class ARTSeparateAsync(
    AgentLoopMetricsMixin,
    AdvantageZeroMetricsMixin,
    PPOTrainerSeparateAsync,
):
    pass
