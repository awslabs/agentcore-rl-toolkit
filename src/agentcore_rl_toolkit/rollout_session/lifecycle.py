"""The rollout session contract, and the bounded lifecycle that drives it.

A :class:`RolloutSession` is where one rollout happens -- a live Docker container, an
AgentCore session -- behind three calls (``setup``, ``run``, ``shutdown``) that say
nothing about which of those it is. Implementations live beside this module.

A session is an async context manager, so its underlying resources are scoped
lexically by the caller:

    async with session:
        await session.setup(task)
        rollout = await session.run(task)

Exiting the block -- normally, on error, or on cancellation between ``setup`` and
``run`` -- guarantees ``shutdown`` runs, so callers never hand-roll a ``finally``.
``shutdown`` is idempotent and may also be called explicitly.

:func:`run_rollout_with_bounds` is that lifecycle driven under the shared limits
every caller needs -- concurrency slots, group priorities, the session-creation rate
-- bundled as a :class:`ContainerBounds`. Both callers use it unchanged: a trainer
passes bounds backed by cluster-wide Ray actors, a single-process driver passes
process-local ones.
"""

import asyncio
import dataclasses
import datetime as dt
from typing import Protocol, runtime_checkable

from agentcore_rl_toolkit.aws_tools.persistent_dict import (
    PersistentDict,
    measure_span_persistent,
)
from agentcore_rl_toolkit.concurrency.priority_assigner import PriorityAssigner
from agentcore_rl_toolkit.concurrency.priority_semaphore import PrioritySemaphore
from agentcore_rl_toolkit.concurrency.rate_limiter import RateLimiter
from agentcore_rl_toolkit.rollout_session.wire import RolloutDumpResponse


@runtime_checkable
class RolloutSession(Protocol):
    async def setup(self, task: dict) -> None:
        """Provision the container and prepare the task environment inside it."""
        ...

    async def run(self, task: dict) -> RolloutDumpResponse:
        """Run one agent rollout in the prepared container and return its dump."""
        ...

    async def shutdown(self) -> None:
        """Tear down the container / session. Idempotent; safe to call anytime."""
        ...

    async def __aenter__(self) -> "RolloutSession":
        """Enter the session's lifetime scope; ``shutdown`` runs on exit."""
        ...

    async def __aexit__(self, exc_type, exc, tb) -> None:
        ...


@dataclasses.dataclass
class ContainerBounds:
    """Everything that bounds a container rollout: concurrency, priority, rate, time.

    The resources are interfaces, so the same bundle describes both deployments: a
    trainer fills it with Ray actors, so its bounds hold across every worker in the
    cluster, while a single-process driver fills it with the local implementations and
    needs no cross-process coordination. One instance is built per experiment and
    shared by every rollout -- that sharing is what makes the semaphores cap anything
    at all.

    The two assigners are separate so container and rollout priorities are numbered
    independently. ``session_rate_limiter`` is optional: ``None`` means container
    creation is throttled only by the semaphores.

    The two timeouts are per-phase deadlines rather than shared state, but they bound
    a rollout just as much as the semaphores do and are fixed for the same scope --
    one experiment -- so they travel with the rest instead of as loose arguments at
    every call site.
    """

    container_semaphore: PrioritySemaphore
    rollout_semaphore: PrioritySemaphore
    container_priority_assigner: PriorityAssigner
    rollout_priority_assigner: PriorityAssigner
    container_setup_timeout: float
    agent_run_timeout: float
    session_rate_limiter: RateLimiter | None = None


async def run_rollout_with_bounds(
    session_state: PersistentDict,
    bounds: ContainerBounds,
    priority_key: str,
    session: RolloutSession,
    task: dict,
) -> RolloutDumpResponse:
    # Each bounded resource is scoped by its own `async with`, so leaving this
    # function for any reason -- return, error, or cancellation between the nested
    # steps -- releases the concurrency slots and tears the container down in
    # reverse order of acquisition, with no manual `finally` bookkeeping.
    container_priority = await bounds.container_priority_assigner.get_priority(priority_key)
    async with bounds.container_semaphore.slot(container_priority):
        await session_state.update(
            {
                "container_priority": container_priority,
                "container_slot_start_at": dt.datetime.now(),
            }
        )

        # Cluster-wide throttle on container/session creation, applied once the
        # container slot is held and before the session provisions anything.
        if bounds.session_rate_limiter is not None:
            async with measure_span_persistent("rate_limit_wait", session_state):
                await bounds.session_rate_limiter.wait_async()

        async with session:
            # setup with timeout
            async with measure_span_persistent("container_setup", session_state):
                try:
                    async with asyncio.timeout(bounds.container_setup_timeout):
                        await session.setup(task)
                        await session_state.set("container_setup_timeout_exceeded", 0.0)
                except TimeoutError as e:
                    await session_state.set("container_setup_timeout_exceeded", 1.0)
                    raise e
            await session_state.set("container_created_at", dt.datetime.now())

            # the rollout slot is held only for the run itself, released as soon
            # as we leave this block (before the container is torn down).
            rollout_priority = await bounds.rollout_priority_assigner.get_priority(priority_key)
            async with bounds.rollout_semaphore.slot(rollout_priority):
                await session_state.update(
                    {
                        "rollout_priority": rollout_priority,
                        "rollout_slot_start_at": dt.datetime.now(),
                    }
                )

                # run with timeout
                async with measure_span_persistent("agent_run", session_state):
                    try:
                        async with asyncio.timeout(bounds.agent_run_timeout):
                            rollout = await session.run(task)
                            await session_state.set("agent_run_timeout_exceeded", 0.0)
                            # A returned dump does not yet imply the rollout worked,
                            # but that is not this function's call: a rollout that ran
                            # and failed inside the container is data, and raising
                            # would replace its dump with a stack trace of our own.
                            # Callers ask the dump (RolloutDumpResponse.is_successful).
                            # Only failures on this side -- timeouts, transport --
                            # raise.
                            return rollout
                    except TimeoutError as e:
                        await session_state.set("agent_run_timeout_exceeded", 1.0)
                        raise e
