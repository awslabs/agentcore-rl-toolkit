"""Per-key, arrival-order priority assignment."""

from typing import Protocol, runtime_checkable

__all__ = ["PriorityAssigner", "LocalPriorityAssigner"]


@runtime_checkable
class PriorityAssigner(Protocol):
    """Async interface for a per-key, arrival-order priority assigner."""

    async def get_priority(self, key: str) -> int:
        """Return the priority for ``key``."""
        ...


class LocalPriorityAssigner:
    """Assigns a monotonically increasing, arrival-order priority per key."""

    def __init__(self) -> None:
        self._next_priority = 0
        self._priorities: dict[str, int] = {}

    async def get_priority(self, key: str) -> int:
        """Return the priority for ``key``, assigning one on first sight (idempotent)."""
        priority = self._priorities.get(key)
        if priority is None:
            priority = self._next_priority
            self._next_priority += 1
            self._priorities[key] = priority
        return priority
