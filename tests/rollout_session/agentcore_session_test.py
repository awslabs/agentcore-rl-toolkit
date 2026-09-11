#!/usr/bin/env python
"""Unit tests for :class:`AgentCoreSession`'s data-plane client: one client per session
(start, setup, every status poll, the dump, the delete), not one per call, and a client
per call for anyone calling the helpers standalone. Fake aioboto3 session, no AWS.
"""

import asyncio
import unittest
from typing import Any
from unittest import IsolatedAsyncioTestCase, mock

from agentcore_rl_toolkit.aws_tools.agentcore_tools import (
    SESSION_CLIENT_CONFIG,
    invoke_agentcore_session,
)
from agentcore_rl_toolkit.aws_tools.persistent_dict import NullPersister, PersistentDict
from agentcore_rl_toolkit.rollout_session.agentcore_session import AgentCoreSession
from agentcore_rl_toolkit.rollout_session.wire import (
    InvocationRequest,
    InvocationResponse,
    RolloutDumpResponse,
    RolloutSetupResponse,
    RolloutStartResponse,
    RolloutStatusRequest,
    RolloutStatusResponse,
)

MODULE = "agentcore_rl_toolkit.aws_tools.agentcore_tools"
REGION = "us-west-2"
RUNTIME_ARN = f"arn:aws:bedrock-agentcore:{REGION}:123456789012:runtime/agent-abc"
PROVIDER_ARN = f"arn:aws:bedrock-agentcore:{REGION}:123456789012:capacity-provider/pool-1"
STATUS_REQUEST = InvocationRequest(payload=RolloutStatusRequest()).model_dump_json().encode()


class ResourceNotFoundException(Exception):
    """Matched by name, the way ``stop_agentcore_session`` matches it."""


class FakeBody:
    """The streaming body of an ``invoke_agent_runtime`` response."""

    def __init__(self, body: str):
        self._body = body

    async def __aenter__(self) -> "FakeBody":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def read(self) -> bytes:
        return self._body.encode()


async def one_ok_chunk():
    """The ``invoke_agent_runtime_command`` stream of a container that started."""
    yield {"chunk": {"contentStop": {"exitCode": 0}}}


class FakeClient:
    """An open ``bedrock-agentcore`` client, recording the calls made on it."""

    def __init__(self, region: str, config: Any, recorder: "Recorder"):
        self.region = region
        self.config = config
        self.recorder = recorder
        self.closed = False
        self.calls: list[str] = []

    def _record(self, name: str) -> None:
        assert not self.closed, f"{name} was made on a closed client"
        self.calls.append(name)
        self.recorder.calls.append(name)

    async def invoke_agent_runtime_command(self, **kwargs):
        self._record("start")
        return {"stream": one_ok_chunk()}

    async def invoke_agent_runtime(self, *, payload: bytes, **kwargs):
        request = InvocationRequest.model_validate_json(payload)
        self._record(request.payload.request_type)
        reply = self.recorder.reply(request.payload.request_type)
        # A real call suspends here, so concurrent sessions genuinely interleave.
        await asyncio.sleep(0)
        return {"statusCode": 200, "response": FakeBody(InvocationResponse(payload=reply).model_dump_json())}

    async def delete_capacity_provider_session(self, **kwargs):
        self._record("delete")
        if self.recorder.deletes_missing:
            raise ResourceNotFoundException("session already gone")


class FakeClientContext:
    """What ``session.client(...)`` returns; the exit is the only thing that closes it."""

    def __init__(self, client: FakeClient):
        self.client = client

    async def __aenter__(self) -> FakeClient:
        await asyncio.sleep(0)
        return self.client

    async def __aexit__(self, *exc) -> None:
        self.client.closed = True
        self.client.recorder.calls.append("close")


class Recorder:
    """Counts the clients a run opened, and logs every call across all of them."""

    def __init__(self, pending_polls: int = 0, deletes_missing: bool = False):
        self.clients: list[FakeClient] = []
        self.calls: list[str] = []
        self.pending_polls = pending_polls
        self.deletes_missing = deletes_missing

    def session(self):
        recorder = self

        class FakeSession:
            def client(self, service: str, region_name: str, config: Any = None) -> FakeClientContext:
                assert service == "bedrock-agentcore"
                client = FakeClient(region_name, config, recorder)
                recorder.clients.append(client)
                return FakeClientContext(client)

        return FakeSession()

    def reply(self, request_type: str):
        if request_type == "rollout_setup_request":
            return RolloutSetupResponse()
        if request_type == "rollout_start_request":
            return RolloutStartResponse()
        if request_type == "rollout_status_request":
            # Stays not-done for `pending_polls` calls, so a rollout really does poll.
            done = self.pending_polls <= 0
            self.pending_polls -= 1
            return RolloutStatusResponse(done=done, exception=None)
        if request_type == "rollout_dump_request":
            return RolloutDumpResponse(task_output={"patch": "diff"}, reward=1.0, exception=None)
        raise AssertionError(f"unexpected request {request_type}")

    def patch(self):
        return mock.patch(f"{MODULE}.get_aioboto3_session", mock.AsyncMock(return_value=self.session()))


def no_sleep():
    """Drop the 10s status-poll interval: what is under test is the client, not timing."""

    async def _sleep(delay, result=None, **kwargs):
        return result

    return mock.patch("asyncio.sleep", _sleep)


def session(session_id: str = "s1") -> AgentCoreSession:
    return AgentCoreSession(
        session_id,
        session_state=PersistentDict({"session_id": session_id}, persister=NullPersister()),
        runtime_arn=RUNTIME_ARN,
        capacity_provider_arn=PROVIDER_ARN,
    )


async def run_rollout(s: AgentCoreSession) -> RolloutDumpResponse:
    """One whole rollout: the lifecycle ``run_rollout_with_bounds`` drives."""
    async with s:
        await s.setup({"index": 1})
        return await s.run({"index": 1})


class SharedClientTest(IsolatedAsyncioTestCase):
    """Inside ``async with session``, one client serves every call it makes."""

    async def test_one_client_for_a_whole_rollout_however_many_polls(self):
        recorder = Recorder(pending_polls=12)
        with recorder.patch(), no_sleep():
            dump = await run_rollout(session())

        self.assertEqual(dump.reward, 1.0)
        self.assertEqual(len(recorder.clients), 1)
        # setup: start + setup invoke + polls; run: start invoke + polls + dump; then the
        # delete -- 19 calls, all on the one client.
        client = recorder.clients[0]
        self.assertEqual(client.calls.count("rollout_status_request"), 14)
        self.assertEqual(len(client.calls), 19)
        self.assertTrue(client.closed)

    async def test_the_calls_are_the_rollout_protocol_in_order(self):
        recorder = Recorder()
        with recorder.patch():
            await run_rollout(session())

        self.assertEqual(
            recorder.calls,
            [
                "start",
                "rollout_setup_request",
                "rollout_status_request",
                "rollout_start_request",
                "rollout_status_request",
                "rollout_dump_request",
                "delete",
                "close",
            ],
        )

    async def test_the_client_closes_only_after_the_delete(self):
        # The teardown must ride the shared client too, hence closing it last.
        recorder = Recorder()
        with recorder.patch():
            await run_rollout(session())
        self.assertEqual(recorder.calls[-2:], ["delete", "close"])

    async def test_the_shared_client_keeps_the_deep_retry_budget(self):
        # start_agentcore_session's budget: session creation is the throttled call.
        recorder = Recorder()
        with recorder.patch():
            await run_rollout(session())
        self.assertIs(recorder.clients[0].config, SESSION_CLIENT_CONFIG)
        self.assertEqual(recorder.clients[0].config.retries, {"max_attempts": 16, "mode": "standard"})

    async def test_the_client_is_opened_in_the_region_of_the_arns(self):
        recorder = Recorder()
        with recorder.patch():
            await run_rollout(session())
        self.assertEqual(recorder.clients[0].region, REGION)

    async def test_each_session_owns_its_own_client(self):
        recorder = Recorder()
        with recorder.patch():
            await asyncio.gather(*(run_rollout(session(f"s{i}")) for i in range(4)))

        self.assertEqual(len(recorder.clients), 4)
        self.assertTrue(all(c.closed for c in recorder.clients))
        # Concurrent rollouts interleave without their calls landing on one another's client.
        self.assertTrue(all(len(c.calls) == 7 for c in recorder.clients), [c.calls for c in recorder.clients])

    async def test_a_failing_rollout_still_closes_the_client(self):
        recorder = Recorder()
        with recorder.patch():
            with self.assertRaises(RuntimeError):
                async with (s := session()):
                    await s.setup({})
                    raise RuntimeError("boom")

        self.assertEqual(len(recorder.clients), 1)
        self.assertTrue(recorder.clients[0].closed)
        # Teardown still happened, on the still-open client.
        self.assertEqual(recorder.calls[-2:], ["delete", "close"])


class ShutdownTest(IsolatedAsyncioTestCase):
    """``shutdown`` stays idempotent and safe whether or not a client is open."""

    async def test_shutdown_on_a_session_that_was_never_entered(self):
        recorder = Recorder()
        with recorder.patch():
            await session().shutdown()

        self.assertEqual(recorder.calls, ["delete", "close"])
        self.assertEqual(len(recorder.clients), 1)

    async def test_shutdown_after_the_scope_opens_a_fresh_client(self):
        # The session's client is gone by then, so the call must not reuse it.
        recorder = Recorder()
        with recorder.patch():
            s = session()
            async with s:
                pass
            held = recorder.clients[0]
            await s.shutdown()

        self.assertEqual(len(recorder.clients), 2)
        self.assertIsNot(recorder.clients[1], held)
        self.assertTrue(all(c.closed for c in recorder.clients))

    async def test_an_already_deleted_session_is_not_an_error(self):
        recorder = Recorder(deletes_missing=True)
        with recorder.patch():
            async with session():
                pass
            await session().shutdown()

        self.assertEqual(recorder.calls, ["delete", "close", "delete", "close"])

    async def test_exiting_twice_shuts_down_once_more_but_never_raises(self):
        recorder = Recorder()
        with recorder.patch():
            s = session()
            await s.__aenter__()
            await s.__aexit__(None, None, None)
            await s.__aexit__(None, None, None)  # the scope is gone: plain shutdown
        self.assertEqual(recorder.calls, ["delete", "close", "delete", "close"])


class StandaloneCallTest(IsolatedAsyncioTestCase):
    """Without a session, a helper opens and closes a client for its one call --
    what dev scripts and cleanup jobs rely on."""

    async def test_a_call_with_no_client_opens_its_own(self):
        recorder = Recorder()
        with recorder.patch():
            resp = await invoke_agentcore_session(RUNTIME_ARN, "s1", STATUS_REQUEST)

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(recorder.clients), 1)
        self.assertTrue(recorder.clients[0].closed)

    async def test_a_call_per_client_is_what_a_loop_of_them_costs(self):
        recorder = Recorder()
        with recorder.patch():
            for _ in range(3):
                await invoke_agentcore_session(RUNTIME_ARN, "s1", STATUS_REQUEST)
        self.assertEqual(len(recorder.clients), 3)


if __name__ == "__main__":
    unittest.main()
