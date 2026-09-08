"""A dict whose mutations are persisted to an external store.

``PersistentDict`` folds the two into one. Reads are ordinary synchronous dict
access; every mutator is an async method that applies the change in memory and then
awaits the ``Persister`` before returning. Where the data lands -- dynamodb, a file,
nowhere -- is decided by the ``Persister`` composed in, not by subclassing the dict.

The ``Persister`` receives a snapshot of the full dict on every mutation, so a
dynamodb-backed persister writing a key-scoped partial ``SET`` (see
``dynamodb_tools.update_dict``) can share one item with another persister as long as
their key sets are disjoint. Note that dynamodb partial updates only ``SET`` the
keys present in the snapshot: deleting a key removes it in memory but leaves the
previously written attribute in the item.

Because every mutation is a write, the *client* setup -- not the write -- is what
dominates at high concurrency. :meth:`PersistentDict.connection` is the opt-in fix:
an async context that holds the persister's client open for every persist inside it.
Nothing has to use it; a persist outside such a block behaves exactly as before.

The ``Persister`` seam is store-agnostic in principle, but in practice this exists
for dynamodb and any further store it grows is likely to be cloud-native too --
hence its home under ``aws_tools`` rather than at the top of the package.
"""

import datetime as dt
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Iterator, Mapping, Protocol, Self, runtime_checkable

from agentcore_rl_toolkit.aws_tools.boto3_tools import get_aioboto3_session
from agentcore_rl_toolkit.aws_tools.dynamodb_tools import update_dict


@runtime_checkable
class Persister(Protocol):
    """Sink for a :class:`PersistentDict`'s state.

    Implementations receive a snapshot of the dict after every mutation and are
    responsible for durably storing it. ``persist`` must be idempotent -- it is
    called once per mutation with the complete current state, not a delta.

    ``connection`` is where an implementation holds whatever a persist would
    otherwise set up per call (a network client, an open file); persists made
    inside the block reuse it. It is part of the protocol so callers can open it
    without knowing which persister they got, and an implementation with nothing
    to keep open just yields.
    """

    async def persist(self, data: Mapping[str, Any]) -> None:
        ...

    def connection(self) -> Any:
        """An async context manager reusing one connection for its whole block."""
        ...


class PersistentDict:
    """A dict-like facade that persists itself on every mutation.

    Reads (``d[k]``, ``get``, ``in``, iteration, ``len``, ``items`` ...) are
    synchronous and never touch the persister. Mutators (``set``, ``update``,
    ``setdefault``, ``pop``, ``delete``, ``clear``) are ``async`` and await the
    persister after applying the change in memory, so awaiting the mutator is a
    guarantee the change was durably stored.

    The initial ``data`` is adopted as the starting state and is *not* persisted
    on construction; call an explicit mutator (or :meth:`persist`) to write it.
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
        """Hold the persister's connection open for the duration of the block.

        A pass-through to :meth:`Persister.connection`, so a caller handed only
        the dict -- with no reference to the persister behind it -- can still
        amortize the client setup over a run's worth of mutations::

            async with meta.connection():
                await run_everything()

        Every mutation inside has to be awaited inside, since the connection is
        released on exit.
        """
        return self._persister.connection()

    @asynccontextmanager
    async def mutate(self) -> AsyncGenerator[dict[str, Any], None]:
        """Batch several in-place mutations and persist once on clean exit.

        Yields the live backing dict, so a block of synchronous writes -- or any helper
        that mutates a plain dict in place -- is applied and flushed with a single
        persist. Nothing is persisted if the block raises.
        """
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

    Each ``persist`` writes every non-key attribute in the snapshot; the item's
    key fields are supplied by ``key`` and skipped by ``update_dict``. Two
    persisters targeting the same item coexist as long as their non-key attribute
    sets are disjoint.

    A persister opens a dynamodb client per ``persist`` and holds none between
    calls, so it stays cheap to create one per session and safe to keep across the
    credential lifetime of a run. :meth:`connection` opts out of that per-call
    setup for the duration of a block.

    Table and region are required, with no defaults: which item a dict lands in is
    the caller's config, not a property of this class. A shared module that knew a
    particular table name would make every caller of it depend on one deployment.
    """

    def __init__(self, table_name: str, key: dict[str, Any], region_name: str):
        self._table_name = table_name
        self._key = key
        self._region_name = region_name
        self._table: Any = None

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[Self, None]:
        """Hold one dynamodb client open for every persist made inside the block.

        Outside a block, :meth:`persist` opens its own aioboto3 resource per call and
        closes it again -- a fresh aiohttp connector and SSL context for every write,
        while a ``PersistentDict`` writes on every single mutation. At high
        concurrency that setup dominates: py-spy attributed ~38% of on-CPU event-loop
        time to ``_build_verify_context``, i.e. to constructing those SSL contexts
        rather than to the writes. Inside a block the resource is created once and
        every persist reuses it, concurrently -- the table is opened before the block
        starts, so there is nothing to synchronize.

        Wrap a whole run::

            async with persister.connection():
                await run_everything()   # every persist reuses one client

        Exiting closes the client, so every persist has to be awaited *inside* the
        block: a write still in flight at exit fails against a closed client.
        Blocks may nest -- an inner one opens its own resource and restores the
        outer one's on exit.
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
        """Nothing to keep open; present so callers need not special-case it."""
        yield self


@asynccontextmanager
async def measure_span_persistent(span_name: str, metrics: PersistentDict):
    """The ``measure_span`` timing helper, for a :class:`PersistentDict`.

    ``PersistentDict`` has no ``__setitem__`` and its mutators are async, so the
    synchronous ``measure_span`` cannot write to it. This variant persists in two
    steps so the store reflects live progress: the start is written on entry (an
    open span shows a ``*_start_at`` with no ``*_end_at`` yet), then the end and
    duration are written on exit -- on normal completion *and* on error, mirroring
    ``measure_span``'s ``finally``, so a span that raises (e.g. a timed-out
    rollout) still leaves a durable trace.

    It lives here rather than beside ``measure_span`` because it is specific to
    this class: it is the only reason ``time`` would have to know about
    ``PersistentDict`` at all, and keeping it here leaves that module a plain,
    dependency-free timing helper.
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
