"""The cluster-wide resources a
:class:`~.rollout_session_agent_loop.RolloutSessionAgentLoop` runs on.

A rollout session is bounded by limits that have to hold across the whole run,
not per loop instance: how many trajectories may be in flight, how many
containers, and how fast new AgentCore sessions may be created. Each is one
process-local object from
:mod:`~agentcore_rl_toolkit.concurrency` hosted as a **named Ray actor**,
so every agent-loop worker in the cluster shares the same counter.

A named actor is reached by a string, which makes that string a contract between
whoever creates the actor and whoever looks it up. Both halves are in this module
for exactly that reason: :func:`start_rollout_session_agent_loop_resources`
creates them and :func:`get_rollout_session_bounds` fetches them, so the names are
written once and a second agent cannot drift from the first. They used to be bare
strings in ``swe_agent/main.py`` and in ``rollout_session_agent_loop.py``,
spelled twice on opposite sides of the trainer entry point.

Nothing here is specific to the SWE agent: the resources follow from the *loop's*
config node, so any agent that runs a ``RolloutSessionAgentLoop`` reuses this
module via :class:`~.task_runner.TaskRunnerWithRolloutSessionResources`.

The two halves run in different places even though they share a file: the lookup
runs in every verl worker (``rollout_session_agent_loop`` imports it, and that
module is star-imported by the package ``__init__`` that
``VERL_USE_EXTERNAL_MODULES`` loads everywhere), while the creation runs once, on
the driver. Keeping the names
together is worth the one import that buys -- :mod:`~..aws_tools.ec2_monitor` and
:mod:`~..aws_tools.ec2_tools` in a worker that has no use for them -- because the
boto3 stack underneath them is already loaded there by ``persistent_dict``.
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

# The named-actor contract. Created by start_rollout_session_agent_loop_resources,
# looked up by get_rollout_session_bounds; no other spelling of these names exists.
CONTAINER_SEMAPHORE = "container_concurrency_semaphore"
ROLLOUT_SEMAPHORE = "rollout_concurrency_semaphore"
CONTAINER_PRIORITY_ASSIGNER = "container_priority_assigner"
ROLLOUT_PRIORITY_ASSIGNER = "rollout_priority_assigner"
SESSION_RATE_LIMITER = "session_create_rate_limiter"

# Headroom on the semaphore actors' concurrency: a waiter occupies one of the
# actor's async slots for as long as it is blocked, so the ceiling has to exceed
# the number of callers that can be queued at once, not the number of permits.
CONCURRENCY_SAFETY_MARGIN = 2


@dataclasses.dataclass
class RolloutSessionAgentLoopResources:
    """The handles for one run; the creator must keep this alive for the run's length.

    Holding on to it is load-bearing rather than tidy. A named actor is not
    ``detached``, so it is garbage-collected once the last handle to it is gone --
    dropping this dataclass would tear down the semaphores mid-run. The monitor is
    here for a related reason: it owns the only reference to its polling task.
    """

    container_semaphore: ActorProxy[LocalPrioritySemaphore]
    rollout_semaphore: ActorProxy[LocalPrioritySemaphore]
    container_priority_assigner: ActorProxy[LocalPriorityAssigner]
    rollout_priority_assigner: ActorProxy[LocalPriorityAssigner]
    session_rate_limiter: ActorProxy[ACRRateLimiter]
    ec2_monitor: EC2Monitor | None


def _max_concurrency(config: DictConfig) -> int:
    """How many callers may be queued on a semaphore actor at once.

    Every blocked ``acquire`` holds one of the actor's async slots, so this bounds
    waiters rather than permits: every trajectory of every in-flight batch, plus
    margin.

    ``num_warmup_batches`` is read unconditionally. It is defined in verl's config
    schema for all trainer modes (default 1), so a sync or colocate-async agent
    gets a slightly over-generous ceiling rather than a ``KeyError`` -- and an
    over-generous ceiling costs nothing but actor threads.
    """
    trajectories_per_batch = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
    batches_inflight = config.trainer.v1.separate_async.num_warmup_batches + 1
    return trajectories_per_batch * batches_inflight * CONCURRENCY_SAFETY_MARGIN


async def start_rollout_session_agent_loop_resources(
    config: DictConfig,
) -> RolloutSessionAgentLoopResources:
    """Create this run's shared actors, or adopt the ones already there.

    Every actor is created with ``get_if_exists=True``, so calling this twice in a
    cluster is a no-op that returns the same handles -- which is what makes it safe
    for the caller to be a task runner that may be restarted.

    Async because of the EC2 monitor alone: it polls on the caller's event loop, so
    the caller has to be an async actor that outlives the run.
    """
    cfg: RolloutSessionAgentLoopConfig = instantiate(config.rollout_session_agent_loop, _convert_="all")
    bounds_cfg = cfg.rollout_session_bounds
    max_concurrency = _max_concurrency(config)

    return RolloutSessionAgentLoopResources(
        # A semaphore per dimension, each with an assigner that numbers groups in
        # arrival order: the V1 trainer does not set a rollout priority, so the
        # assigner is what gives the semaphore something to order waiters by.
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

    It replaces a per-session lookup that exhausted the EC2 API rate limit at
    rollout scale. Nothing calls into it, so it is a background task on the
    caller's event loop rather than an actor of its own.
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

    ``bounds_cfg`` is the loop config's ``rollout_session_bounds`` node, handed
    over whole: this function and
    :func:`start_rollout_session_agent_loop_resources` are the only readers of its
    keys -- the concurrency/rate ceilings there, the per-rollout timeouts here.

    Called per loop instance in every agent-loop worker. The actors must already
    exist -- ``ray.get_actor`` raises ``ValueError`` if not -- which they do because
    :class:`~.task_runner.TaskRunnerWithRolloutSessionResources` creates them
    before verl builds any agent loop.

    Only the timeouts come from the config: they bound a single rollout's phases,
    so unlike the limits above they need no cluster-wide state.
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
