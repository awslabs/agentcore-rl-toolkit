"""Shared base class and helpers for the trainer mixins in this package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Static-analysis-only base so mixins can reference the concrete trainer's members;
    # at runtime the real base comes from the recipe's trainer class.
    from verl.trainer.ppo.v1 import PPOTrainer

    TrainerMixinBase = PPOTrainer
else:
    TrainerMixinBase = object


def rollout_server_addresses(trainer: PPOTrainer) -> list[str]:
    """Addresses of the vLLM servers actually generating this run's rollouts.

    In separate-async the hybrid ``llm_server_manager`` is colocated with the trainer and sits
    idle (frozen vLLM counters) after warmup, so prefer the standalone rollout servers when
    present. Both managers exist before ``on_init_end`` fires.
    """
    server_manager = getattr(trainer, "standalone_server_manager", None) or trainer.llm_server_manager
    return server_manager.get_addresses()
