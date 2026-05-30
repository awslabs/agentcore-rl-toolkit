"""AgentCore-specific configuration dataclasses for verl integration."""

from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig
from verl.workers.config.actor import FSDPActorConfig, McoreActorConfig
from verl.workers.config.rollout import RolloutConfig


@dataclass
class AgentCoreConfig(BaseConfig):
    """Configuration for AgentCore rollout mode.

    These fields live under ``actor_rollout_ref.rollout.agentcore.*``.
    """

    agent_runtime_arn: str = ""
    s3_bucket: str = ""
    reqs_per_sec: int = 25
    max_pool_connections: int = 10
    max_rollout_time: int = 1800
    gateway_port: int = 9090
    gateway_store: str = "memory"


@dataclass
class AgentCoreRolloutConfig(RolloutConfig):
    """RolloutConfig extended with an ``agentcore`` field.

    The official verl ``RolloutConfig`` doesn't know about AgentCore; this
    subclass adds the typed ``agentcore`` field so Hydra's
    ``omega_conf_to_dataclass`` can instantiate the rollout config without
    raising ``TypeError: unexpected keyword argument 'agentcore'``.
    """

    agentcore: AgentCoreConfig = field(default_factory=AgentCoreConfig)


# AgentCore mode can dispatches one agent trace into multiple sequences.
# However, verl split a global batch into ppo mini batches where each batch
# has ``ppo_mini_batch_size`` sequences, the number of ppo mini steps is
# not fixed under AgentCore mode.
# ``ppo_mini_steps`` enables user to explicitly control how many ppo mini
# steps per global batch.
_AGENTCORE_ACTOR_FIELDS = {"ppo_mini_steps"}


@dataclass
class AgentCoreMcoreActorConfig(McoreActorConfig):
    """McoreActorConfig extended with ``ppo_mini_steps``."""

    _mutable_fields = McoreActorConfig._mutable_fields | _AGENTCORE_ACTOR_FIELDS
    ppo_mini_steps: Optional[int] = None


@dataclass
class AgentCoreFSDPActorConfig(FSDPActorConfig):
    """FSDPActorConfig extended with ``ppo_mini_steps``."""

    _mutable_fields = FSDPActorConfig._mutable_fields | _AGENTCORE_ACTOR_FIELDS
    ppo_mini_steps: Optional[int] = None
