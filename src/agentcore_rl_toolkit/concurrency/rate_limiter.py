"""Request-rate limiting interface, satisfied by :class:`ACRRateLimiter`."""

from typing import Protocol, runtime_checkable

from agentcore_rl_toolkit.client import ACRRateLimiter

__all__ = ["RateLimiter", "ACRRateLimiter"]


@runtime_checkable
class RateLimiter(Protocol):
    """Async interface for a request-rate limiter; the limiter does the waiting."""

    async def wait_async(self) -> None:
        """Block until the caller may issue its next rate-limited request."""
        ...
