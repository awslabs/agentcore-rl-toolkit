#!/usr/bin/env python
"""PersistentDict write semantics and DynamoDB client lifetime; no AWS calls."""

import asyncio
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, mock

from agentcore_rl_toolkit.aws_tools.persistent_dict import (
    DynamoDBPersister,
    NullPersister,
    PersistentDict,
)

MODULE = "agentcore_rl_toolkit.aws_tools.persistent_dict"
TABLE = "test_sessions"
REGION = "us-west-2"


class Recorder:
    """Counts what a run opened, and what it wrote."""

    def __init__(self):
        self.clients: list[SimpleNamespace] = []
        self.writes: list[tuple[str, dict, dict]] = []

    @asynccontextmanager
    async def create_client(self, service: str, region_name: str):
        assert service == "dynamodb"
        client = SimpleNamespace(region=region_name, closed=False)
        self.clients.append(client)
        try:
            yield client
        finally:
            client.closed = True

    async def update_dict(self, client, table_name, key, updates, **kwargs):
        await asyncio.sleep(0)
        assert not client.closed
        self.writes.append((table_name, dict(key), dict(updates)))

    def patch(self):
        return mock.patch.multiple(
            MODULE,
            get_async_session=mock.AsyncMock(return_value=SimpleNamespace(create_client=self.create_client)),
            update_dict=mock.AsyncMock(side_effect=self.update_dict),
        )


def session_dict(session_id: str = "s1", **data) -> PersistentDict:
    return PersistentDict(
        {"session_id": session_id, **data},
        persister=DynamoDBPersister(TABLE, {"session_id": session_id}, REGION),
    )


class PersistentDictTest(IsolatedAsyncioTestCase):
    """The dict facade: reads never persist, every mutator persists once."""

    async def test_initial_data_is_not_persisted(self):
        persister = mock.AsyncMock(spec=NullPersister)
        d = PersistentDict({"a": 1}, persister=persister)
        self.assertEqual(d["a"], 1)
        self.assertIn("a", d)
        self.assertEqual(d.snapshot(), {"a": 1})
        persister.persist.assert_not_awaited()

    async def test_each_mutator_persists_the_whole_snapshot(self):
        persister = mock.AsyncMock(spec=NullPersister)
        d = PersistentDict({"a": 1}, persister=persister)

        await d.set("b", 2)
        await d.update({"c": 3}, dd=4)
        async with d.mutate() as live:
            live["e"] = 5

        self.assertEqual(
            persister.persist.await_args_list,
            [
                mock.call({"a": 1, "b": 2}),
                mock.call({"a": 1, "b": 2, "c": 3, "dd": 4}),
                mock.call({"a": 1, "b": 2, "c": 3, "dd": 4, "e": 5}),
            ],
        )


class StandalonePersistTest(IsolatedAsyncioTestCase):
    """Without a connection, every persist opens and closes its own client."""

    async def test_persists_non_key_fields_and_closes_each_client(self):
        recorder = Recorder()
        with recorder.patch():
            d = session_dict()
            await d.set("a", 1)
            await d.set("b", 2)

        self.assertEqual(len(recorder.clients), 2)
        self.assertTrue(all(r.closed for r in recorder.clients))
        self.assertEqual(
            recorder.writes,
            [
                (TABLE, {"session_id": "s1"}, {"a": 1}),
                (TABLE, {"session_id": "s1"}, {"a": 1, "b": 2}),
            ],
        )

    async def test_nothing_to_write_opens_no_client(self):
        recorder = Recorder()
        with recorder.patch():
            # Only the key field is present, so the partial SET would be empty.
            await session_dict().persist()
        self.assertEqual(recorder.clients, [])
        self.assertEqual(recorder.writes, [])


class ConnectionTest(IsolatedAsyncioTestCase):
    """Inside a ``connection`` block, one client serves every persist on it."""

    async def test_concurrent_writes_reuse_client_until_connection_exits(self):
        recorder = Recorder()
        with recorder.patch():
            d = PersistentDict(persister=DynamoDBPersister(TABLE, {"session_id": "s1"}, region_name="us-east-1"))
            async with d.connection():
                await asyncio.gather(*(d.set(f"a{i}", i) for i in range(4)))
                self.assertEqual(len(recorder.clients), 1)
                held = recorder.clients[0]
                self.assertEqual(held.region, "us-east-1")
                self.assertFalse(held.closed)
                self.assertEqual(len(recorder.writes), 4)
                self.assertEqual(recorder.writes[-1], (TABLE, {"session_id": "s1"}, {f"a{i}": i for i in range(4)}))
            self.assertTrue(held.closed)
            await d.set("b", 2)

        self.assertEqual(len(recorder.clients), 2)
        self.assertIsNot(recorder.clients[1], held)
        self.assertTrue(recorder.clients[1].closed)

    async def test_each_persister_owns_its_own_client(self):
        recorder = Recorder()
        with recorder.patch():
            d1, d2 = session_dict("s1"), session_dict("s2")
            async with d1.connection():
                await d1.set("a", 1)
                await d2.set("a", 1)  # no connection of its own: persists per call

        self.assertEqual(len(recorder.clients), 2)
        self.assertEqual(len(recorder.writes), 2)

    async def test_exit_closes_the_client_when_the_block_raises(self):
        recorder = Recorder()
        with recorder.patch():
            d = session_dict("s1")
            with self.assertRaises(RuntimeError):
                async with d.connection():
                    await d.set("a", 1)
                    raise RuntimeError("boom")

        self.assertTrue(all(r.closed for r in recorder.clients))

    async def test_nested_blocks_restore_the_outer_client(self):
        recorder = Recorder()
        with recorder.patch():
            d = session_dict("s1")
            async with d.connection():
                async with d.connection():
                    await d.set("a", 1)
                    inner = recorder.clients[1]
                self.assertTrue(inner.closed)
                await d.set("b", 2)  # the outer client is still open and serving
                self.assertFalse(recorder.clients[0].closed)

        self.assertEqual(len(recorder.clients), 2)
        self.assertEqual(len(recorder.writes), 2)


if __name__ == "__main__":
    unittest.main()
