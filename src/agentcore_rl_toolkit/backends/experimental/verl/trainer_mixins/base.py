"""Plumbing shared by the trainer mixins in this package.

Two things every mixin needs live here: the static base each of them declares,
and the address list of the vLLM servers actually serving a run's rollouts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Give the type checker the concrete trainer's members (self.config,
    # self.llm_server_manager, self.global_steps, super().on_init_end(), ...).
    # At runtime the base is ``object``; the real base is supplied by the
    # recipe's trainer class, so this only affects static analysis, not the MRO.
    from verl.trainer.ppo.v1 import PPOTrainer

    TrainerMixinBase = PPOTrainer
else:
    TrainerMixinBase = object


def rollout_server_addresses(trainer: PPOTrainer) -> list[str]:
    """Addresses of the vLLM servers actually generating this run's rollouts.

    Prefer the always-on generation servers. In separate-async the hybrid
    ``llm_server_manager`` is colocated with the trainer: it is slept at init and
    only wakes during rollout-mode switches, so with ``parameter_sync_step > 1``
    it sits idle in trainer mode after warmup and its vLLM counters freeze --
    every delta-based rollout metric then returns None and only the static
    kv_cache_token_capacity survives past step 1. The standalone rollout servers
    (created in ``PPOTrainerSeparateAsync._setup`` and used by
    ``get_llm_client``) are the ones actually generating, so scrape those when
    present. Both managers are created before ``on_init_end`` fires.
    """
    server_manager = getattr(trainer, "standalone_server_manager", None) or trainer.llm_server_manager
    return server_manager.get_addresses()
