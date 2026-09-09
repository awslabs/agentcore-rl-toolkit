"""The rollout session contract, and the bounded lifecycle that drives it.

A :class:`RolloutSession` runs one rollout behind ``setup`` / ``run`` / ``shutdown``,
as an async context manager whose exit always shuts the session down.
:func:`run_rollout_with_bounds` drives that lifecycle under the shared limits in
:class:`ContainerBounds`.
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

    One instance is built per experiment and shared by every rollout -- that sharing is
    what makes the semaphores cap anything. The two assigners are separate so container
    and rollout priorities are numbered independently. ``session_rate_limiter=None``
    means container creation is throttled only by the semaphores.
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
    # Each bounded resource is scoped by its own `async with`, so leaving this function
    # for any reason releases the slots and tears the container down in reverse order.
    container_priority = await bounds.container_priority_assigner.get_priority(priority_key)
    async with bounds.container_semaphore.slot(container_priority):
        await session_state.update(
            {
                "container_priority": container_priority,
                "container_slot_start_at": dt.datetime.now(),
            }
        )

        # Cluster-wide throttle on container/session creation.
        if bounds.session_rate_limiter is not None:
            async with measure_span_persistent("rate_limit_wait", session_state):
                await bounds.session_rate_limiter.wait_async()

        async with session:
            async with measure_span_persistent("container_setup", session_state):
                try:
                    async with asyncio.timeout(bounds.container_setup_timeout):
                        await session.setup(task)
                        await session_state.set("container_setup_timeout_exceeded", 0.0)
                except TimeoutError as e:
                    await session_state.set("container_setup_timeout_exceeded", 1.0)
                    raise e
            await session_state.set("container_created_at", dt.datetime.now())

            # The rollout slot is held only for the run, released before teardown.
            rollout_priority = await bounds.rollout_priority_assigner.get_priority(priority_key)
            async with bounds.rollout_semaphore.slot(rollout_priority):
                await session_state.update(
                    {
                        "rollout_priority": rollout_priority,
                        "rollout_slot_start_at": dt.datetime.now(),
                    }
                )

                async with measure_span_persistent("agent_run", session_state):
                    try:
                        async with asyncio.timeout(bounds.agent_run_timeout):
                            rollout = await session.run(task)
                            await session_state.set("agent_run_timeout_exceeded", 0.0)
                            # A dump may describe a failed rollout; that is data, so
                            # callers ask RolloutDumpResponse.is_successful. Only
                            # failures on this side (timeout, transport) raise.
                            return rollout
                    except TimeoutError as e:
                        await session_state.set("agent_run_timeout_exceeded", 1.0)
                        raise e
