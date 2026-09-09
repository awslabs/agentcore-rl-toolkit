"""A dict whose mutations are persisted to an external store.

Reads are ordinary synchronous dict access; mutators are async and await a composed
``Persister`` (dynamodb, null, ...) before returning.
"""

import datetime as dt
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Iterator, Mapping, Protocol, Self, runtime_checkable

from agentcore_rl_toolkit.aws_tools.boto3_tools import get_aioboto3_session
from agentcore_rl_toolkit.aws_tools.dynamodb_tools import update_dict


@runtime_checkable
class Persister(Protocol):
    """Sink for a :class:`PersistentDict`'s state.

    ``persist`` receives the complete current state, not a delta.
    """

    async def persist(self, data: Mapping[str, Any]) -> None:
        ...

    def connection(self) -> Any:
        """An async context manager reusing one connection for its whole block."""
        ...


class PersistentDict:
    """A dict-like facade that persists itself on every mutation.

    The initial ``data`` is adopted as the starting state and is *not* persisted
    on construction.
    """

    def __init__(self, data: Mapping[str, Any] | None = None, *, persister: Persister):
        self._data: dict[str, Any] = dict(data or {})
        self._persister = persister

    # --- sync readers -----------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def snapshot(self) -> dict[str, Any]:
        """Return a shallow copy of the current state."""
        return dict(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"

    # --- async mutators ---------------------------------------------------

    async def persist(self) -> None:
        """Write the current state to the persister without mutating it."""
        await self._persister.persist(dict(self._data))

    def connection(self) -> Any:
        """Hold the persister's connection open for the block; mutations must be awaited inside."""
        return self._persister.connection()

    @asynccontextmanager
    async def mutate(self) -> AsyncGenerator[dict[str, Any], None]:
        """Yield the live backing dict; persist once on clean exit (never if the block raises)."""
        yield self._data
        await self.persist()

    async def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        await self.persist()

    async def update(self, other: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        if other:
            self._data.update(other)
        if kwargs:
            self._data.update(kwargs)
        await self.persist()

    async def setdefault(self, key: str, default: Any = None) -> Any:
        if key in self._data:
            return self._data[key]
        self._data[key] = default
        await self.persist()
        return default

    async def pop(self, key: str, *default: Any) -> Any:
        value = self._data.pop(key, *default)
        await self.persist()
        return value

    async def delete(self, key: str) -> None:
        del self._data[key]
        await self.persist()

    async def clear(self) -> None:
        self._data.clear()
        await self.persist()


class DynamoDBPersister:
    """Persist a dict to a single dynamodb item via a key-scoped partial ``SET``.

    Only keys present in the snapshot are ``SET``, so a deleted key leaves its
    previously written attribute behind in the item.
    """

    def __init__(self, table_name: str, key: dict[str, Any], region_name: str):
        self._table_name = table_name
        self._key = key
        self._region_name = region_name
        self._table: Any = None

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[Self, None]:
        """Hold one dynamodb client open for every persist made inside the block.

        Avoids building a fresh connector and SSL context per write, which dominates
        event-loop CPU at high concurrency. Exiting closes the client, so every
        persist must be awaited inside. Blocks may nest.
        """
        session = await get_aioboto3_session()
        async with session.resource("dynamodb", region_name=self._region_name) as dynamodb:
            table = await dynamodb.Table(self._table_name)
            outer, self._table = self._table, table
            try:
                yield self
            finally:
                self._table = outer

    async def persist(self, data: Mapping[str, Any]) -> None:
        updates = {k: v for k, v in data.items() if k not in self._key}
        if not updates:
            return

        if self._table is not None:
            await update_dict(self._table, self._key, updates)
            return

        session = await get_aioboto3_session()
        async with session.resource("dynamodb", region_name=self._region_name) as dynamodb:
            table = await dynamodb.Table(self._table_name)
            await update_dict(table, self._key, updates)


class NullPersister:
    """A persister that discards everything -- useful for tests and dry runs."""

    async def persist(self, data: Mapping[str, Any]) -> None:
        return None

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[Self, None]:
        yield self


@asynccontextmanager
async def measure_span_persistent(span_name: str, metrics: PersistentDict):
    """The ``measure_span`` timing helper, for a :class:`PersistentDict`.

    Persists in two steps so the store reflects live progress: an open span has a
    ``*_start_at`` but no ``*_end_at``. The end is written even if the block raises.
    """
    assert span_name not in metrics
    start = dt.datetime.now()
    await metrics.set(f"{span_name}_start_at", start)
    try:
        yield
    finally:
        end = dt.datetime.now()
        await metrics.update(
            {
                f"{span_name}_end_at": end,
                span_name: (end - start).seconds,
            }
        )
