"""Request-rate limiting interface and process-local implementation."""

import asyncio
import time
from typing import Protocol, runtime_checkable

__all__ = ["RateLimiter", "LocalRateLimiter"]


@runtime_checkable
class RateLimiter(Protocol):
    """Async interface for a request-rate limiter; the limiter does the waiting."""

    async def wait_async(self) -> None:
        """Block until the caller may issue its next rate-limited request."""
        ...


class LocalRateLimiter:
    """Space requests at a fixed rate within one process.

    Callers share an instance to share its request budget. Supports synchronous
    waiting and asynchronous waiting with a lock to queue concurrent callers.
    """

    def __init__(self, tps_limit: int = 25):
        self.tps_limit = tps_limit
        self._min_interval = 1.0 / tps_limit
        self._last_call_time = 0.0
        # Async lock and its event loop (lazily created)
        self._async_lock = None
        self._async_lock_loop = None

    def _get_async_lock(self) -> asyncio.Lock:
        """Lazily create and return the async rate-limiting lock.

        Detects when the running event loop has changed (e.g., due to a new
        ``asyncio.run()`` call) and recreates the lock for the current loop.
        """
        loop = asyncio.get_running_loop()
        if self._async_lock is None or self._async_lock_loop is not loop:
            self._async_lock = asyncio.Lock()
            self._async_lock_loop = loop
        return self._async_lock

    def wait_sync(self):
        """Block until the next call is allowed under the TPS limit."""
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_time = time.time()

    async def wait_async(self):
        """Async wait until the next call is allowed under the TPS limit.

        Uses a lock to serialize timing checks. The lock is held only during
        the timing check and sleep, so concurrent callers queue up properly.
        """
        async with self._get_async_lock():
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_call_time = time.time()
