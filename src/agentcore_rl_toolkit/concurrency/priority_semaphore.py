import asyncio
import heapq
import itertools
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import AsyncGenerator, Protocol, runtime_checkable

__all__ = ["PrioritySemaphore", "LocalPrioritySemaphore"]


@runtime_checkable
class PrioritySemaphore(Protocol):
    """Async interface for a priority-ordered concurrency limiter.

    Callers ``await sem.acquire(priority)`` before entering a bounded region and
    ``await sem.release()`` on the way out, or use the :meth:`slot` async context
    manager to pair the two. Lower ``priority`` integers are served first.

    The in-process implementation is :class:`LocalPrioritySemaphore`; a shared,
    cluster-wide permit pool is that class run as a Ray actor behind an adapter. Both
    satisfy this interface, so nothing downstream depends on Ray -- which is what lets
    one configuration describe both the training cluster and a single-process driver,
    and why the Ray side lives elsewhere in this package rather than here.
    """

    async def acquire(self, priority: int = 0) -> None:
        """Block until a permit is available for the caller."""
        ...

    async def release(self) -> None:
        """Return a permit, waking the next-highest-priority waiter."""
        ...

    def slot(self, priority: int = 0) -> AbstractAsyncContextManager[None]:
        """Acquire a permit on enter and release it on exit."""
        ...


class LocalPrioritySemaphore:
    """
    asyncio semaphore where lower integer values have higher priority.

    Among equal-priority waiters, acquisition is FIFO.
    """

    def __init__(self, value: int = 1) -> None:
        if value < 0:
            raise ValueError("initial value must be >= 0")

        self._value = value
        self._waiters: list[tuple[int, int, asyncio.Future[None]]] = []
        self._sequence = itertools.count()

    async def acquire(self, priority: int = 0) -> None:
        # Do not let a new caller bypass already queued callers.
        if self._value > 0 and not self._waiters:
            self._value -= 1
            return

        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()

        heapq.heappush(
            self._waiters,
            (priority, next(self._sequence), future),
        )

        try:
            await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                # A permit was assigned to this waiter just before the task
                # was cancelled. Return it to the semaphore.
                await self.release()
            else:
                # Leave it in the heap; release() removes cancelled entries.
                future.cancel()
            raise

    async def release(self) -> None:
        while self._waiters:
            _priority, _sequence, future = heapq.heappop(self._waiters)

            if future.cancelled():
                continue

            future.set_result(None)
            return

        self._value += 1

    @asynccontextmanager
    async def slot(self, priority: int = 0) -> AsyncGenerator[None, None]:
        await self.acquire(priority)
        try:
            yield
        finally:
            await self.release()

    @property
    def value(self) -> int:
        return self._value

    def locked(self) -> bool:
        return self._value == 0
