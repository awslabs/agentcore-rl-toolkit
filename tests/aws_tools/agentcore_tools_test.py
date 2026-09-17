#!/usr/bin/env python
"""Unit tests for the process-wide ``bedrock-agentcore`` client: one per region per event
loop however many callers ask for it, opened once under a burst, and reopened after a close.
Fake aioboto3 session, no AWS.
"""

import asyncio
import unittest
from typing import Any
from unittest import IsolatedAsyncioTestCase, mock

from agentcore_rl_toolkit.aws_tools.agentcore_tools import (
    MAX_POOL_CONNECTIONS,
    SESSION_CLIENT_CONFIG,
    close_shared_agentcore_clients,
    shared_agentcore_client,
)

MODULE = "agentcore_rl_toolkit.aws_tools.agentcore_tools"
REGION = "us-west-2"


class FakeClient:
    def __init__(self, region: str, config: Any):
        self.region = region
        self.config = config
        self.closed = False


class FakeClientContext:
    """What ``session.client(...)`` returns: one client, opened and closed by hand here."""

    def __init__(self, client: FakeClient, opens: list[FakeClient]):
        self.client = client
        self.opens = opens

    async def __aenter__(self) -> FakeClient:
        # A real client creation suspends, which is the window the creation lock closes.
        await asyncio.sleep(0)
        self.opens.append(self.client)
        return self.client

    async def __aexit__(self, *exc) -> None:
        self.client.closed = True


class FakeSession:
    def __init__(self):
        self.opened: list[FakeClient] = []

    def client(self, service: str, region_name: str, config: Any = None) -> FakeClientContext:
        assert service == "bedrock-agentcore"
        return FakeClientContext(FakeClient(region_name, config), self.opened)


class SharedClientTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.session = FakeSession()
        patch = mock.patch(f"{MODULE}.get_aioboto3_session", mock.AsyncMock(return_value=self.session))
        patch.start()
        self.addCleanup(patch.stop)
        self.addAsyncCleanup(close_shared_agentcore_clients)

    async def test_every_caller_in_one_region_gets_the_one_client(self):
        first = await shared_agentcore_client(REGION)
        for _ in range(5):
            self.assertIs(await shared_agentcore_client(REGION), first)
        self.assertEqual(len(self.session.opened), 1)

    async def test_a_burst_of_first_callers_still_opens_one_client(self):
        # Without the creation lock each of these would open a client during the other's
        # await, and all but the last would be dropped unclosed.
        clients = await asyncio.gather(*(shared_agentcore_client(REGION) for _ in range(16)))

        self.assertEqual(len(self.session.opened), 1)
        self.assertEqual(len(set(map(id, clients))), 1)

    async def test_each_region_gets_its_own_client(self):
        west = await shared_agentcore_client(REGION)
        east = await shared_agentcore_client("us-east-1")

        self.assertIsNot(west, east)
        self.assertEqual([c.region for c in self.session.opened], [REGION, "us-east-1"])

    async def test_the_client_carries_the_session_config(self):
        client = await shared_agentcore_client(REGION)
        self.assertIs(client.config, SESSION_CLIENT_CONFIG)
        self.assertEqual(client.config.max_pool_connections, MAX_POOL_CONNECTIONS)

    async def test_the_client_outlives_the_call_that_opened_it(self):
        client = await shared_agentcore_client(REGION)
        self.assertFalse(client.closed)

    async def test_closing_ends_the_clients_and_the_next_call_reopens(self):
        first = await shared_agentcore_client(REGION)
        await shared_agentcore_client("us-east-1")

        await close_shared_agentcore_clients()
        self.assertTrue(all(c.closed for c in self.session.opened))

        second = await shared_agentcore_client(REGION)
        self.assertIsNot(second, first)
        self.assertFalse(second.closed)
        self.assertEqual(len(self.session.opened), 3)

    async def test_closing_twice_is_not_an_error(self):
        await shared_agentcore_client(REGION)
        await close_shared_agentcore_clients()
        await close_shared_agentcore_clients()

    async def test_closing_with_nothing_open_is_not_an_error(self):
        await close_shared_agentcore_clients()

    async def test_another_loops_client_is_left_alone(self):
        # A client's aiohttp connector belongs to the loop that created it, so a loop must
        # neither hand its client to another nor close one it does not own.
        mine = await shared_agentcore_client(REGION)

        def in_its_own_loop():
            async def get():
                client = await shared_agentcore_client(REGION)
                await close_shared_agentcore_clients()
                return client

            return asyncio.run(get())

        theirs = await asyncio.to_thread(in_its_own_loop)

        self.assertIsNot(theirs, mine)
        self.assertTrue(theirs.closed)
        self.assertFalse(mine.closed)


class PoolSizeTest(unittest.TestCase):
    """The pool is what a whole worker's rollouts share, not one rollout's."""

    def test_the_pool_covers_the_container_concurrency_the_examples_configure(self):
        # `container_concurrency: 256` in the example configs, each container holding at
        # most one connection at a time, plus headroom for teardown calls beside them.
        self.assertGreaterEqual(MAX_POOL_CONNECTIONS, 256)
        self.assertEqual(SESSION_CLIENT_CONFIG.max_pool_connections, MAX_POOL_CONNECTIONS)


if __name__ == "__main__":
    unittest.main()
