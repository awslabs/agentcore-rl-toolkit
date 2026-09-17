#!/usr/bin/env python
"""Unit tests for the ACR data plane helpers: the process-wide ``bedrock-agentcore`` client
(one per region per event loop however many callers ask for it, opened once under a burst,
and reopened after a close), the session warm-up and its retry through the container
readiness race, and the best-effort session stop. Fake aioboto3 session, no AWS.
"""

import asyncio
import unittest
from typing import Any
from unittest import IsolatedAsyncioTestCase, mock

from botocore.exceptions import ClientError

from agentcore_rl_toolkit.aws_tools.agentcore_tools import (
    MAX_POOL_CONNECTIONS,
    SESSION_CLIENT_CONFIG,
    START_MAX_TRIES,
    close_shared_agentcore_clients,
    is_runtime_readiness_error,
    shared_agentcore_client,
    start_agentcore_session,
    stop_agentcore_microvm_session,
)

MODULE = "agentcore_rl_toolkit.aws_tools.agentcore_tools"
REGION = "us-west-2"
RUNTIME_ARN = f"arn:aws:bedrock-agentcore:{REGION}:123456789012:runtime/agent-abc"
SESSION_ID = "verl_" + "0" * 32


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


class RuntimeClientError(ClientError):
    """Stands in for ``botocore.errorfactory.RuntimeClientError``.

    botocore synthesizes that class per client out of the service model, so there is nothing
    importable to raise here; what the code under test matches on -- the class name, the
    ``ClientError`` base and the error code -- is reproduced exactly.
    """


def client_error(code: str, cls: type[ClientError] = ClientError) -> ClientError:
    return cls({"Error": {"Code": code, "Message": "Received error (500) from runtime"}}, "InvokeAgentRuntimeCommand")


class FakeCommandClient:
    """``invoke_agent_runtime_command``, scripted per call.

    ``errors`` is consumed one entry per call, so ``(err, None)`` is "fails once, then
    starts"; once it runs out every call succeeds.
    """

    def __init__(self, *errors: Exception | None, exit_code: int = 0, chunks: list | None = None):
        self.errors = list(errors)
        self.exit_code = exit_code
        self.chunks = chunks
        self.calls: list[dict] = []

    async def invoke_agent_runtime_command(self, **kwargs):
        self.calls.append(kwargs)
        error = self.errors.pop(0) if self.errors else None
        if error is not None:
            raise error
        return {"stream": self._stream()}

    async def _stream(self):
        default = [{"chunk": {"contentStop": {"exitCode": self.exit_code}}}]
        for chunk in default if self.chunks is None else self.chunks:
            yield chunk

    @property
    def session_ids(self) -> list[str]:
        return [call["runtimeSessionId"] for call in self.calls]


class StartSessionTest(IsolatedAsyncioTestCase):
    """The warm-up that creates the microVM, and the readiness race it retries through."""

    def setUp(self):
        # The real interval is the measured recovery time; the retry logic is what is tested.
        patch = mock.patch(f"{MODULE}.START_RETRY_INTERVAL", 0)
        patch.start()
        self.addCleanup(patch.stop)

    async def test_a_start_runs_one_command_in_the_session(self):
        acr = FakeCommandClient()
        await start_agentcore_session(RUNTIME_ARN, SESSION_ID, acr)

        self.assertEqual(len(acr.calls), 1)
        self.assertEqual(acr.calls[0]["agentRuntimeArn"], RUNTIME_ARN)
        self.assertEqual(acr.calls[0]["runtimeSessionId"], SESSION_ID)
        self.assertEqual(acr.calls[0]["body"]["command"], "echo hello")

    async def test_the_readiness_race_is_retried_against_the_same_session(self):
        # The failed try's microVM is the one that is now warm, so the retry must reuse the
        # session id rather than start a second cold container.
        acr = FakeCommandClient(client_error("RuntimeClientError", RuntimeClientError))
        await start_agentcore_session(RUNTIME_ARN, SESSION_ID, acr)

        self.assertEqual(acr.session_ids, [SESSION_ID, SESSION_ID])

    async def test_a_readiness_error_that_only_arrives_mid_stream_is_retried_too(self):
        # An error event inside the command stream surfaces as botocore's EventStreamError,
        # which carries the same code under a different class name.
        acr = FakeCommandClient(client_error("RuntimeClientError"))
        await start_agentcore_session(RUNTIME_ARN, SESSION_ID, acr)

        self.assertEqual(len(acr.calls), 2)

    async def test_a_runtime_that_never_becomes_ready_raises_after_the_last_try(self):
        error = client_error("RuntimeClientError", RuntimeClientError)
        acr = FakeCommandClient(*([error] * START_MAX_TRIES))

        with self.assertRaises(RuntimeClientError):
            await start_agentcore_session(RUNTIME_ARN, SESSION_ID, acr)
        self.assertEqual(len(acr.calls), START_MAX_TRIES)

    async def test_any_other_failure_is_raised_on_the_first_try(self):
        # A rollout cannot recover from these, and retrying an access-denied or validation
        # burst only multiplies it.
        for code in ("AccessDeniedException", "ValidationException", "ServiceQuotaExceededException"):
            acr = FakeCommandClient(client_error(code), None)
            with self.assertRaises(ClientError):
                await start_agentcore_session(RUNTIME_ARN, SESSION_ID, acr)
            self.assertEqual(len(acr.calls), 1, code)

    async def test_a_command_that_exits_nonzero_is_a_start_failure(self):
        acr = FakeCommandClient(exit_code=1)
        with self.assertRaises(AssertionError):
            await start_agentcore_session(RUNTIME_ARN, SESSION_ID, acr)
        # Not the readiness race: the container answered, and it answered badly.
        self.assertEqual(len(acr.calls), 1)

    async def test_a_stream_with_no_chunks_names_the_session(self):
        acr = FakeCommandClient(chunks=[])
        with self.assertRaises(AssertionError) as caught:
            await start_agentcore_session(RUNTIME_ARN, SESSION_ID, acr)
        self.assertIn(SESSION_ID, str(caught.exception))

    def test_a_readiness_error_is_recognised_by_class_name_alone(self):
        # The modelled error is the observed shape; the code check is the fallback, not the
        # other way round.
        self.assertTrue(is_runtime_readiness_error(type("RuntimeClientError", (Exception,), {})()))
        self.assertFalse(is_runtime_readiness_error(RuntimeError("nope")))
        self.assertFalse(is_runtime_readiness_error(client_error("ThrottlingException")))


class StopMicrovmSessionTest(IsolatedAsyncioTestCase):
    """The data-plane stop: by runtime arn, and never raising."""

    class FakeStopClient:
        def __init__(self, error: Exception | None = None):
            self.error = error
            self.calls: list[dict] = []

        async def stop_runtime_session(self, **kwargs):
            self.calls.append(kwargs)
            if self.error is not None:
                raise self.error

    async def test_a_stop_ends_the_session_of_that_runtime(self):
        acr = self.FakeStopClient()
        self.assertTrue(await stop_agentcore_microvm_session(RUNTIME_ARN, SESSION_ID, acr))
        self.assertEqual(acr.calls, [{"agentRuntimeArn": RUNTIME_ARN, "runtimeSessionId": SESSION_ID}])

    async def test_a_failed_stop_is_reported_rather_than_raised(self):
        # Teardown runs with the rollout's own exception in flight, and must not replace it.
        acr = self.FakeStopClient(client_error("AccessDeniedException"))
        with self.assertLogs(MODULE, level="WARNING"):
            self.assertFalse(await stop_agentcore_microvm_session(RUNTIME_ARN, SESSION_ID, acr))


class PoolSizeTest(unittest.TestCase):
    """The pool is what a whole worker's rollouts share, not one rollout's."""

    def test_the_pool_covers_the_container_concurrency_the_examples_configure(self):
        # `container_concurrency: 256` in the example configs, each container holding at
        # most one connection at a time, plus headroom for teardown calls beside them.
        self.assertGreaterEqual(MAX_POOL_CONNECTIONS, 256)
        self.assertEqual(SESSION_CLIENT_CONFIG.max_pool_connections, MAX_POOL_CONNECTIONS)


if __name__ == "__main__":
    unittest.main()
