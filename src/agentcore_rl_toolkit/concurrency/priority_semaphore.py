"""Priority-ordered concurrency limiting."""

import asyncio
import heapq
import itertools
import threading
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


class _Waiter:
    """One queued acquirer, mutated under the semaphore lock from any loop/thread."""

    __slots__ = ("future", "loop", "assigned", "cancelled")

    def __init__(self, future: "asyncio.Future[None]", loop: asyncio.AbstractEventLoop) -> None:
        self.future = future
        self.loop = loop
        self.assigned = False
        self.cancelled = False


class LocalPrioritySemaphore:
    """asyncio semaphore where lower integers have higher priority; FIFO within a priority.
    Thread-safe: All shared state is guarded by a ``threading.Lock``.
    """

    def __init__(self, value: int = 1) -> None:
        if value < 0:
            raise ValueError("initial value must be >= 0")

        self._value = value
        self._waiters: list[tuple[int, int, _Waiter]] = []
        self._sequence = itertools.count()
        self._lock = threading.Lock()

    async def acquire(self, priority: int = 0) -> None:
        with self._lock:
            # Do not let a new caller bypass already queued callers.
            if self._value > 0 and not self._waiters:
                self._value -= 1
                return

            loop = asyncio.get_running_loop()
            waiter = _Waiter(loop.create_future(), loop)
            heapq.heappush(self._waiters, (priority, next(self._sequence), waiter))

        try:
            await waiter.future
        except asyncio.CancelledError:
            with self._lock:
                if waiter.assigned:
                    # A permit was handed to us just before cancellation; give it back.
                    self._release_locked()
                else:
                    # Still queued: mark it dead so release() skips it.
                    waiter.cancelled = True
            raise

    async def release(self) -> None:
        with self._lock:
            self._release_locked()

    def _release_locked(self) -> None:
        """Hand a permit to the next live waiter, or return it to the pool.

        The caller must hold ``self._lock``. Waking is scheduled on the waiter's own loop
        so the future is only ever mutated from the thread that created it.
        """
        while self._waiters:
            _priority, _sequence, waiter = heapq.heappop(self._waiters)

            if waiter.cancelled:
                continue

            waiter.assigned = True
            waiter.loop.call_soon_threadsafe(self._wake, waiter.future)
            return

        self._value += 1

    @staticmethod
    def _wake(future: "asyncio.Future[None]") -> None:
        # Runs on the waiter's own loop. The awaiting task may have been cancelled between
        # assignment and this callback firing; acquire()'s cancel path reclaims that permit,
        # so here we simply skip a future that is already resolved.
        if not future.done():
            future.set_result(None)

    @asynccontextmanager
    async def slot(self, priority: int = 0) -> AsyncGenerator[None, None]:
        await self.acquire(priority)
        try:
            yield
        finally:
            await self.release()

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    def locked(self) -> bool:
        with self._lock:
            return self._value == 0
