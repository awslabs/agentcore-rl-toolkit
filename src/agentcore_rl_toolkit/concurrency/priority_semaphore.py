"""Priority-ordered concurrency limiting."""

import asyncio
import heapq
import itertools
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import AsyncGenerator, Protocol, runtime_checkable

__all__ = ["PrioritySemaphore", "LocalPrioritySemaphore"]


@runtime_checkable
class PrioritySemaphore(Protocol):
    """Async interface for a priority-ordered concurrency limiter; lower integers go first."""

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
    """asyncio semaphore where lower integers have higher priority; FIFO within a priority."""

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
                # A permit was assigned just before cancellation; give it back.
                await self.release()
            else:
                # Leave it in the heap; release() skips cancelled entries.
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
