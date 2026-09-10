"""Cluster-wide resources a ``RolloutSessionAgentLoop`` runs on.

The run-wide limits (trajectory/container concurrency, session creation rate) are
process-local ``concurrency`` objects hosted as named Ray actors so every agent-loop
worker shares one counter. Creation (driver, once) and lookup (every worker, per loop
instance) both live here so the actor names are written down only once.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any

import ray
from hydra.utils import instantiate
from omegaconf import DictConfig
from ray.actor import ActorProxy

from agentcore_rl_toolkit.aws_tools.ec2_monitor import EC2Monitor, start_ec2_monitor
from agentcore_rl_toolkit.concurrency.priority_assigner import LocalPriorityAssigner
from agentcore_rl_toolkit.concurrency.priority_semaphore import LocalPrioritySemaphore
from agentcore_rl_toolkit.concurrency.rate_limiter import ACRRateLimiter
from agentcore_rl_toolkit.concurrency.ray_adapters import (
    RayPriorityAssigner,
    RayPrioritySemaphore,
    RayRateLimiter,
)
from agentcore_rl_toolkit.rollout_session.lifecycle import RolloutSessionBounds

from .config import RolloutSessionAgentLoopConfig

__all__ = [
    "RolloutSessionAgentLoopResources",
    "start_rollout_session_agent_loop_resources",
    "get_rollout_session_bounds",
]

# the named-actor contract between the creation and lookup halves below
CONTAINER_SEMAPHORE = "container_concurrency_semaphore"
ROLLOUT_SEMAPHORE = "rollout_concurrency_semaphore"
CONTAINER_PRIORITY_ASSIGNER = "container_priority_assigner"
ROLLOUT_PRIORITY_ASSIGNER = "rollout_priority_assigner"
SESSION_RATE_LIMITER = "session_create_rate_limiter"

# headroom on the semaphore actors' async slots, which bound queued waiters not permits
CONCURRENCY_SAFETY_MARGIN = 2


@dataclasses.dataclass
class RolloutSessionAgentLoopResources:
    """The handles for one run; the creator must keep this alive for the run's length.

    The actors are not ``detached``, so dropping this would collect them mid-run, and the
    monitor holds the only reference to its polling task.
    """

    container_semaphore: ActorProxy[LocalPrioritySemaphore]
    rollout_semaphore: ActorProxy[LocalPrioritySemaphore]
    container_priority_assigner: ActorProxy[LocalPriorityAssigner]
    rollout_priority_assigner: ActorProxy[LocalPriorityAssigner]
    session_rate_limiter: ActorProxy[ACRRateLimiter]
    ec2_monitor: EC2Monitor | None


def _max_concurrency(config: DictConfig) -> int:
    """How many callers may be queued on a semaphore actor at once.

    Sized as every trajectory of every in-flight batch plus margin. ``num_warmup_batches``
    exists in all trainer modes, so non-async runs just get an over-generous ceiling.
    """
    trajectories_per_batch = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
    batches_inflight = config.trainer.v1.separate_async.num_warmup_batches + 1
    return trajectories_per_batch * batches_inflight * CONCURRENCY_SAFETY_MARGIN


async def start_rollout_session_agent_loop_resources(
    config: DictConfig,
) -> RolloutSessionAgentLoopResources:
    """Create this run's shared actors, or adopt the ones already there.

    Idempotent via ``get_if_exists=True``. Async only for the EC2 monitor, which polls on
    the caller's event loop, so the caller must be an async actor that outlives the run.
    """
    cfg: RolloutSessionAgentLoopConfig = instantiate(config.rollout_session_agent_loop, _convert_="all")
    bounds_cfg = cfg.rollout_session_bounds
    max_concurrency = _max_concurrency(config)

    return RolloutSessionAgentLoopResources(
        # the V1 trainer sets no rollout priority, so each semaphore gets an assigner that
        # numbers groups in arrival order to give it something to order waiters by
        rollout_semaphore=(
            ray.remote(LocalPrioritySemaphore)
            .options(
                name=ROLLOUT_SEMAPHORE,
                get_if_exists=True,
                max_concurrency=max_concurrency,
            )
            .remote(value=bounds_cfg["rollout_concurrency"])
        ),
        rollout_priority_assigner=(
            ray.remote(LocalPriorityAssigner).options(name=ROLLOUT_PRIORITY_ASSIGNER, get_if_exists=True).remote()
        ),
        container_semaphore=(
            ray.remote(LocalPrioritySemaphore)
            .options(
                name=CONTAINER_SEMAPHORE,
                get_if_exists=True,
                max_concurrency=max_concurrency,
            )
            .remote(value=bounds_cfg["container_concurrency"])
        ),
        container_priority_assigner=(
            ray.remote(LocalPriorityAssigner).options(name=CONTAINER_PRIORITY_ASSIGNER, get_if_exists=True).remote()
        ),
        session_rate_limiter=(
            ray.remote(ACRRateLimiter)
            .options(name=SESSION_RATE_LIMITER, get_if_exists=True)
            .remote(tps_limit=bounds_cfg["session_create_rate"])
        ),
        ec2_monitor=await _start_ec2_monitor(cfg),
    )


async def _start_ec2_monitor(cfg: RolloutSessionAgentLoopConfig) -> EC2Monitor | None:
    """One poller for the whole run, mapping running instances back to their sessions.

    A single poller avoids exhausting the EC2 API rate limit at rollout scale. Nothing calls
    into it, so it is a background task on the caller's event loop rather than its own actor.
    """
    backend_cfg = cfg.rollout_session_backend
    if backend_cfg["backend"] != "agentcore" or cfg.ec2_monitor_poll_interval <= 0 or cfg.dynamodb_table is None:
        return None
    return await start_ec2_monitor(
        backend_cfg["capacity_provider_arn"],
        cfg.dynamodb_table,
        poll_interval=cfg.ec2_monitor_poll_interval,
        region_name=cfg.aws_region,
    )


def get_rollout_session_bounds(bounds_cfg: Mapping[str, Any]) -> RolloutSessionBounds:
    """The bounds on one rollout, wired to this run's shared actors.

    Called per loop instance in every agent-loop worker; the actors must already exist
    (``ray.get_actor`` raises ``ValueError`` otherwise), which the task runner guarantees.
    Only the per-rollout timeouts come from ``bounds_cfg`` here.
    """
    return RolloutSessionBounds(
        container_semaphore=RayPrioritySemaphore(ray.get_actor(CONTAINER_SEMAPHORE)),
        rollout_semaphore=RayPrioritySemaphore(ray.get_actor(ROLLOUT_SEMAPHORE)),
        container_priority_assigner=RayPriorityAssigner(ray.get_actor(CONTAINER_PRIORITY_ASSIGNER)),
        rollout_priority_assigner=RayPriorityAssigner(ray.get_actor(ROLLOUT_PRIORITY_ASSIGNER)),
        session_rate_limiter=RayRateLimiter(ray.get_actor(SESSION_RATE_LIMITER)),
        container_setup_timeout=bounds_cfg["container_setup_timeout"],
        agent_run_timeout=bounds_cfg["agent_run_timeout"],
    )
