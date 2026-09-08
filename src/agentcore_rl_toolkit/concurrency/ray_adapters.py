"""Ray-actor adapters for the protocols in this package -- the only Ray-dependent code here.

Each of :mod:`.priority_semaphore`, :mod:`.priority_assigner` and
:mod:`.rate_limiter` defines a protocol plus a process-local implementation, and none
of them imports Ray. A cluster-wide instance is that same local class run as a Ray
actor; the adapters below turn an actor handle back into something that satisfies the
protocol, so callers see one interface either way -- which is what lets one
configuration describe both the training cluster and a single-process driver.

They live together in one module because each is a handful of ``await
handle.method.remote(...)`` forwards -- too thin to be worth a file each -- and
because that keeps this package's whole relationship to Ray in one place. Nothing
here calls a Ray API: the handle does the remoting and ``ActorProxy`` appears only in
annotations, so its import sits under ``TYPE_CHECKING`` and even this module imports
with Ray absent. Ray is a dependency of whoever *hosts* the actors, not of the code
that talks to them -- and each adapter is parameterised by the concrete class hosted
as the actor rather than by the protocol, since that is what the handle on the other
end actually is.

The adapters are not redundant with the actor handle itself. A ``@ray.remote`` handle
does forward ``acquire``/``release``, but Ray turns the ``@asynccontextmanager``
``slot`` into an ``ObjectRefGenerator``, which does not implement the async context
manager protocol -- so ``slot`` has to be rebuilt on this side from the two
primitives. The same boundary drops ``@property`` accessors, which is why nothing
here forwards ``value`` and callers ask ``locked()`` instead.

Untested as of 2026-09-04: the tests that booted a local Ray cluster were removed
from the unit suite (~10s of a ~19s run, and the real subject was the actor wiring
rather than the semaphore). These adapters are covered again when Ray and the
trainer integration are tested together.
"""

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator

from agentcore_rl_toolkit.concurrency.priority_assigner import LocalPriorityAssigner
from agentcore_rl_toolkit.concurrency.priority_semaphore import LocalPrioritySemaphore
from agentcore_rl_toolkit.concurrency.rate_limiter import ACRRateLimiter

if TYPE_CHECKING:
    from ray.actor import ActorProxy

__all__ = ["RayPrioritySemaphore", "RayPriorityAssigner", "RayRateLimiter"]


class RayPrioritySemaphore:
    """Adapts a Ray actor handle to the :class:`~.priority_semaphore.PrioritySemaphore` interface.

    ``acquire``/``release`` forward to the actor's methods. ``slot`` is
    reconstructed locally from those two calls rather than forwarded, for the
    reason given in this module's docstring.
    """

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
