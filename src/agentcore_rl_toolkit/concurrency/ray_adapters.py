"""Adapters turning Ray actor handles into this package's protocols.

Nothing here calls a Ray API -- the handle does the remoting and ``ActorProxy`` is only
an annotation -- so this module imports with Ray absent. ``slot`` must be rebuilt
locally rather than forwarded: Ray turns an ``@asynccontextmanager`` into an
``ObjectRefGenerator``, and drops ``@property`` accessors.
"""

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator

import ray

from agentcore_rl_toolkit.concurrency.priority_assigner import LocalPriorityAssigner
from agentcore_rl_toolkit.concurrency.priority_semaphore import LocalPrioritySemaphore
from agentcore_rl_toolkit.concurrency.rate_limiter import ACRRateLimiter

if TYPE_CHECKING:
    from ray.actor import ActorProxy

__all__ = ["RayPrioritySemaphore", "RayPriorityAssigner", "RayRateLimiter"]


class RayPrioritySemaphore:
    """Adapts a Ray actor handle to the :class:`~.priority_semaphore.PrioritySemaphore` interface."""

    def __init__(self, actor: "ActorProxy[LocalPrioritySemaphore]") -> None:
        self._actor = actor

    async def acquire(self, priority: int = 0) -> None:
        await self._actor.acquire.remote(priority)

    async def release(self) -> None:
        await self._actor.release.remote()

    @asynccontextmanager
    async def slot(self, priority: int = 0) -> AsyncGenerator[None, None]:
        await self.acquire(priority)
        try:
            yield
        finally:
            await self.release()


class RayPriorityAssigner:
    """Adapts a Ray actor handle to the :class:`~.priority_assigner.PriorityAssigner` interface."""

    def __init__(self, actor: "ActorProxy[LocalPriorityAssigner]") -> None:
        self._actor = actor

    async def get_priority(self, key: str) -> int:
        return await self._actor.get_priority.remote(key)


class RayRateLimiter:
    """Adapts a Ray actor handle to the :class:`~.rate_limiter.RateLimiter` interface."""

    def __init__(self, actor: "ActorProxy[ACRRateLimiter]") -> None:
        self._actor = actor

    async def wait_async(self) -> None:
        await self._actor.wait_async.remote()


class ConcurrencyGroupedSemaphore(LocalPrioritySemaphore):
    """``LocalPrioritySemaphore`` with ``release`` pinned to its own Ray concurrency group.
    Routing ``release`` to a dedicated group gives it slots that acquires cannot occupy.
    """

    @ray.method(concurrency_group="release")
    async def release(self) -> None:
        await super().release()


def make_priority_semaphore(name: str, semaphore_concurrency: int, acquire_concurrency: int) -> ActorProxy[LocalPrioritySemaphore]:
    return (
        ray.remote(concurrency_groups={"release": semaphore_concurrency})(ConcurrencyGroupedSemaphore)
        .options(
            name=name,
            get_if_exists=True,
            max_concurrency=acquire_concurrency,
        )
        .remote(value=semaphore_concurrency)
    )