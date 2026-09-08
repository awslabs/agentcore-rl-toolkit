from typing import Protocol, runtime_checkable

__all__ = ["PriorityAssigner", "LocalPriorityAssigner"]


@runtime_checkable
class PriorityAssigner(Protocol):
    """Async interface for a per-key, arrival-order priority assigner.

    Callers ``await assigner.get_priority(key)`` to learn a key's priority. The
    process-local :class:`LocalPriorityAssigner` and a shared Ray actor fronted by an
    adapter both satisfy it, so nothing downstream depends on Ray -- and neither does
    this module, which is why the Ray side lives elsewhere in this package.
    """

    async def get_priority(self, key: str) -> int:
        """Return the priority for ``key``."""
        ...


class LocalPriorityAssigner:
    """Assigns a monotonically increasing, arrival-order priority per key.

    Given a key, the first caller is handed the next counter value; every later
    caller with the same key gets that same value back. Keys are opaque, so the
    first key seen always receives the lowest number.
    """

    def __init__(self) -> None:
        self._next_priority = 0
        self._priorities: dict[str, int] = {}

    async def get_priority(self, key: str) -> int:
        """Return the priority for ``key``, assigning one on first sight.

        Idempotent per key: repeated calls with the same key return the value
        assigned to the first call.
        """
        priority = self._priorities.get(key)
        if priority is None:
            priority = self._next_priority
            self._next_priority += 1
            self._priorities[key] = priority
        return priority
