#!/usr/bin/env python
"""Unit tests for PersistentDict's dynamodb persistence, and the held-open connection.

The dict half is plain in-memory logic checked against a recording persister. The
dynamodb half is checked against a fake aioboto3 session: what matters is not the
request shape (``dynamodb_tools.update_dict`` owns that) but *how many clients get
opened* -- one per persist by default, exactly one for a whole
:meth:`DynamoDBPersister.connection` block. No AWS.
"""

import asyncio
import unittest
from unittest import IsolatedAsyncioTestCase, mock

from agentcore_rl_toolkit.aws_tools.persistent_dict import (
    DynamoDBPersister,
    NullPersister,
    PersistentDict,
)

MODULE = "agentcore_rl_toolkit.aws_tools.persistent_dict"
TABLE = "test_sessions"
REGION = "us-west-2"


class FakeTable:
    """Stands in for an aioboto3 dynamodb ``Table`` resource."""

    def __init__(self, name: str, region: str):
        self.name = name
        self.region = region


class FakeResource:
    """An open dynamodb resource; ``closed`` records that the scope released it."""

    def __init__(self, region: str, recorder: "Recorder"):
        self.region = region
        self.recorder = recorder
        self.closed = False

    async def Table(self, name: str) -> FakeTable:
        # A real Table() is awaitable and lets other tasks interleave here; keep
        # that suspension point so concurrent persists are genuinely interleaved.
        await asyncio.sleep(0)
        self.recorder.tables.append((self.region, name))
        return FakeTable(name, self.region)


class FakeResourceContext:
    """What ``session.resource(...)`` returns: an async context manager."""

    def __init__(self, resource: FakeResource):
        self.resource = resource

    async def __aenter__(self) -> FakeResource:
        await asyncio.sleep(0)
        return self.resource

    async def __aexit__(self, *exc) -> None:
        self.resource.closed = True


class Recorder:
    """Counts what a run of the code under test opened, and what it wrote."""

    def __init__(self):
        self.resources: list[FakeResource] = []
        self.tables: list[tuple[str, str]] = []
        self.writes: list[tuple[str, dict, dict]] = []

    def session(self):
        recorder = self

        class FakeSession:
            def resource(self, service: str, region_name: str) -> FakeResourceContext:
                assert service == "dynamodb"
                resource = FakeResource(region_name, recorder)
                recorder.resources.append(resource)
                return FakeResourceContext(resource)

        return FakeSession()

    async def update_dict(self, table, key, updates, **kwargs):
        self.writes.append((table.name, dict(key), dict(updates)))

    def patch(self):
        """Patch the module's aioboto3 session and dynamodb write with this recorder."""
        return mock.patch.multiple(
            MODULE,
            get_aioboto3_session=mock.AsyncMock(return_value=self.session()),
            update_dict=mock.AsyncMock(side_effect=self.update_dict),
        )


def session_dict(session_id: str = "s1", **data) -> PersistentDict:
    """A PersistentDict writing to the fake table under ``session_id``."""
    return PersistentDict(data, persister=DynamoDBPersister(TABLE, {"session_id": session_id}, REGION))


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

        self.assertEqual(persister.persist.await_count, 3)
        self.assertEqual(
            persister.persist.await_args_list[-1].args[0],
            {"a": 1, "b": 2, "c": 3, "dd": 4, "e": 5},
        )


class StandalonePersistTest(IsolatedAsyncioTestCase):
    """Without a scope, every persist opens and closes its own client."""

    async def test_client_per_persist(self):
        recorder = Recorder()
        with recorder.patch():
            d = session_dict(**{"session_id": "s1"})
            await d.set("a", 1)
            await d.set("b", 2)

        self.assertEqual(len(recorder.resources), 2)
        self.assertTrue(all(r.closed for r in recorder.resources))
        self.assertEqual(recorder.writes[-1], (TABLE, {"session_id": "s1"}, {"a": 1, "b": 2}))

    async def test_key_fields_are_not_written(self):
        recorder = Recorder()
        with recorder.patch():
            await session_dict(**{"session_id": "s1"}).set("a", 1)
        self.assertEqual(recorder.writes, [(TABLE, {"session_id": "s1"}, {"a": 1})])

    async def test_nothing_to_write_opens_no_client(self):
        recorder = Recorder()
        with recorder.patch():
            # Only the key field is present, so the partial SET would be empty.
            await PersistentDict(
                {"session_id": "s1"},
                persister=DynamoDBPersister(TABLE, {"session_id": "s1"}, REGION),
            ).persist()
        self.assertEqual(recorder.resources, [])
        self.assertEqual(recorder.writes, [])


class ConnectionTest(IsolatedAsyncioTestCase):
    """Inside a ``connection`` block, one client serves every persist on it."""

    async def test_one_client_for_many_persists(self):
        recorder = Recorder()
        with recorder.patch():
            d = session_dict("s1")
            async with d.connection():
                for i in range(5):
                    await d.set("a", i)

        self.assertEqual(len(recorder.resources), 1)
        self.assertEqual(recorder.tables, [("us-west-2", TABLE)])
        self.assertEqual(len(recorder.writes), 5)
        self.assertEqual(recorder.writes[-1], (TABLE, {"session_id": "s1"}, {"a": 4}))

    async def test_concurrent_persists_share_the_one_client(self):
        """The table is opened before the block, so a burst needs no synchronizing."""
        recorder = Recorder()
        with recorder.patch():
            d = session_dict("s1")
            async with d.connection():
                await asyncio.gather(*(d.set(f"a{i}", i) for i in range(50)))

        self.assertEqual(len(recorder.resources), 1)
        self.assertEqual(len(recorder.writes), 50)

    async def test_tasks_spawned_inside_share_the_one_client(self):
        """The scope is the persister, not the task, so spawned rollouts reuse it."""
        recorder = Recorder()
        with recorder.patch():
            d = session_dict("s1")
            async with d.connection():
                tasks = [asyncio.create_task(d.set(f"a{i}", i)) for i in range(4)]
                await asyncio.gather(*tasks)

        self.assertEqual(len(recorder.resources), 1)
        self.assertEqual(len(recorder.writes), 4)

    async def test_each_persister_owns_its_own_client(self):
        recorder = Recorder()
        with recorder.patch():
            d1, d2 = session_dict("s1"), session_dict("s2", **{"other": 1})
            async with d1.connection():
                await d1.set("a", 1)
                # d2 has no connection of its own; it still persists per call.
                await d2.set("a", 1)

        self.assertEqual(len(recorder.resources), 2)
        self.assertEqual(len(recorder.writes), 2)

    async def test_region_is_honored(self):
        recorder = Recorder()
        with recorder.patch():
            persister = DynamoDBPersister(TABLE, {"session_id": "s1"}, region_name="us-east-1")
            async with persister.connection():
                await PersistentDict({"a": 1}, persister=persister).persist()

        self.assertEqual(recorder.tables, [("us-east-1", TABLE)])

    async def test_exit_closes_the_client_and_restores_per_call_persists(self):
        recorder = Recorder()
        with recorder.patch():
            d = session_dict("s1")
            async with d.connection():
                await d.set("a", 1)
            held = recorder.resources[0]
            self.assertTrue(held.closed)

            # Outside again: back to a client per persist.
            await d.set("b", 2)

        self.assertEqual(len(recorder.resources), 2)
        self.assertIsNot(recorder.resources[1], held)

    async def test_exit_closes_the_client_when_the_block_raises(self):
        recorder = Recorder()
        with recorder.patch():
            d = session_dict("s1")
            with self.assertRaises(RuntimeError):
                async with d.connection():
                    await d.set("a", 1)
                    raise RuntimeError("boom")

        self.assertTrue(all(r.closed for r in recorder.resources))

    async def test_nested_blocks_restore_the_outer_client(self):
        recorder = Recorder()
        with recorder.patch():
            d = session_dict("s1")
            async with d.connection():
                async with d.connection():
                    await d.set("a", 1)
                    inner = recorder.resources[1]
                self.assertTrue(inner.closed)
                # The outer client is still open and still serving.
                await d.set("b", 2)
                self.assertFalse(recorder.resources[0].closed)

        self.assertEqual(len(recorder.resources), 2)
        self.assertEqual(len(recorder.writes), 2)

    async def test_null_persister_connection_is_a_no_op(self):
        d = PersistentDict({"a": 1}, persister=NullPersister())
        async with d.connection():
            await d.set("b", 2)
        self.assertEqual(d.snapshot(), {"a": 1, "b": 2})


if __name__ == "__main__":
    unittest.main()
