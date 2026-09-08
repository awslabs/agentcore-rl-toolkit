from typing import Protocol, runtime_checkable

from agentcore_rl_toolkit.client import ACRRateLimiter

__all__ = ["RateLimiter", "ACRRateLimiter"]


@runtime_checkable
class RateLimiter(Protocol):
    """Async interface for a request-rate limiter.

    Callers ``await limiter.wait_async()`` before issuing a rate-limited request; the
    call returns only once the caller is clear to proceed -- the limiter does the
    waiting internally. The process-local :class:`ACRRateLimiter` and a shared Ray
    actor behind an adapter both satisfy it, so nothing downstream depends on Ray --
    and neither does this module, which is why the Ray side lives elsewhere in this
    package.
    """

    async def wait_async(self) -> None:
        """Block until the caller may issue its next rate-limited request."""
        ...
