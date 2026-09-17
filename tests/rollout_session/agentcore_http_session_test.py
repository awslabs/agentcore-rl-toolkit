#!/usr/bin/env python
"""Unit tests for :class:`AgentCoreHttpSession`'s data-plane client: every call a session
makes (start, setup, every status poll, the dump, the delete) rides on the process-wide
client rather than one of its own, no session closes it, and anyone calling the helpers
standalone still gets a client per call. Fake aioboto3 session, no AWS.
"""

import asyncio
import unittest
from typing import Any
from unittest import IsolatedAsyncioTestCase, mock

from agentcore_rl_toolkit.aws_tools.agentcore_tools import (
    SESSION_CLIENT_CONFIG,
    close_shared_agentcore_clients,
    invoke_agentcore_session,
)
from agentcore_rl_toolkit.aws_tools.persistent_dict import NullPersister, PersistentDict
from agentcore_rl_toolkit.rollout_session.agentcore_http_session import AgentCoreHttpSession
from agentcore_rl_toolkit.rollout_session.exception_utils import describe_with_root_cause, root_cause
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
    """Matched by name, the way ``stop_agentcore_instance_session`` matches it."""


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
        if self.recorder.delete_error is not None:
            raise self.recorder.delete_error


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

    def __init__(
        self,
        pending_polls: int = 0,
        deletes_missing: bool = False,
        delete_error: Exception | None = None,
    ):
        self.clients: list[FakeClient] = []
        self.calls: list[str] = []
        self.pending_polls = pending_polls
        self.deletes_missing = deletes_missing
        self.delete_error = delete_error

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


def session(session_id: str = "s1") -> AgentCoreHttpSession:
    return AgentCoreHttpSession(
        session_id,
        session_state=PersistentDict({"session_id": session_id}, persister=NullPersister()),
        runtime_arn=RUNTIME_ARN,
        capacity_provider_arn=PROVIDER_ARN,
    )


async def run_rollout(s: AgentCoreHttpSession) -> RolloutDumpResponse:
    """One whole rollout: the lifecycle ``run_rollout_with_bounds`` drives."""
    async with s:
        await s.setup({"index": 1})
        return await s.run({"index": 1})


class SharedClientCase(IsolatedAsyncioTestCase):
    """Base case: end the loop's shared client, so no client crosses into another test."""

    async def asyncSetUp(self) -> None:
        self.addAsyncCleanup(close_shared_agentcore_clients)


class SharedClientTest(SharedClientCase):
    """One process-wide client serves every call of every session."""

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
            ],
        )

    async def test_a_finished_session_leaves_the_client_open_for_the_next_one(self):
        # The session borrows the client; closing it here would tear the pool out from
        # under every rollout still running.
        recorder = Recorder()
        with recorder.patch():
            await run_rollout(session())
            self.assertFalse(recorder.clients[0].closed)
            await run_rollout(session("s2"))

        self.assertEqual(len(recorder.clients), 1)
        self.assertEqual(recorder.calls.count("close"), 0)

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

    async def test_concurrent_sessions_share_the_one_client(self):
        recorder = Recorder()
        with recorder.patch():
            await asyncio.gather(*(run_rollout(session(f"s{i}")) for i in range(4)))

        # Four rollouts, one client, and every call accounted for on it: 7 per rollout.
        self.assertEqual(len(recorder.clients), 1)
        self.assertEqual(len(recorder.clients[0].calls), 28)

    async def test_a_failing_rollout_still_deletes_its_session(self):
        recorder = Recorder()
        with recorder.patch():
            with self.assertRaises(RuntimeError):
                async with (s := session()):
                    await s.setup({})
                    raise RuntimeError("boom")

        self.assertEqual(len(recorder.clients), 1)
        # Teardown still happened, on the still-open client.
        self.assertEqual(recorder.calls[-1], "delete")
        self.assertFalse(recorder.clients[0].closed)


class TeardownDoesNotMaskTest(SharedClientCase):
    """A failing delete must not erase the error that ended the rollout."""

    def chain(self, exc: BaseException) -> list[str]:
        names, cur = [], exc
        while cur is not None:
            names.append(type(cur).__name__)
            cur = cur.__cause__ or cur.__context__
        return names

    async def test_the_rollouts_error_survives_under_the_delete_error(self):
        recorder = Recorder(delete_error=RuntimeError("403 Forbidden"))
        with recorder.patch():
            with self.assertRaises(RuntimeError) as caught:
                async with (s := session()):
                    await s.setup({})
                    raise ValueError("the real error")

        self.assertEqual(self.chain(caught.exception), ["RuntimeError", "ValueError"])
        self.assertEqual(root_cause(caught.exception).args, ("the real error",))

    async def test_a_cancelled_body_survives_too(self):
        # The tell for the other hypothesis: rollouts torn down en masse, not rejected.
        recorder = Recorder(delete_error=RuntimeError("403 Forbidden"))
        with recorder.patch():
            with self.assertRaises(RuntimeError) as caught:
                async with (s := session()):
                    await s.setup({})
                    raise asyncio.CancelledError()

        self.assertEqual(self.chain(caught.exception), ["RuntimeError", "CancelledError"])

    async def test_a_delete_error_over_a_clean_body_has_no_chain(self):
        recorder = Recorder(delete_error=RuntimeError("403 Forbidden"))
        with recorder.patch():
            with self.assertRaises(RuntimeError) as caught:
                async with (s := session()):
                    await s.setup({})

        self.assertEqual(self.chain(caught.exception), ["RuntimeError"])

    async def test_a_failed_delete_does_not_take_the_shared_client_down(self):
        # The other rollouts on this client are unaffected by one session's bad teardown.
        recorder = Recorder(delete_error=RuntimeError("403 Forbidden"))
        with recorder.patch():
            with self.assertRaises(RuntimeError):
                async with (s := session()):
                    await s.setup({})

        self.assertEqual(recorder.calls[-1], "delete")
        self.assertFalse(recorder.clients[0].closed)

    async def test_the_summary_line_names_the_error_under_the_mask(self):
        mask = RuntimeError("An error occurred (403) ... DeleteCapacityProviderSession")
        mask.__context__ = ValueError("the real error")

        described = describe_with_root_cause(mask)
        self.assertIn("403", described)
        self.assertIn("ValueError: the real error", described)
        # Nothing to add when the exception is its own root cause.
        self.assertEqual(describe_with_root_cause(ValueError("alone")), "alone")

    async def test_a_context_cycle_does_not_hang_the_walk(self):
        a, b = ValueError("a"), ValueError("b")
        a.__context__, b.__context__ = b, a
        self.assertIn(root_cause(a), (a, b))


class ShutdownTest(SharedClientCase):
    """``shutdown`` stays idempotent and safe whether or not the session was entered."""

    async def test_shutdown_on_a_session_that_was_never_entered(self):
        # No phase owns the client, so a bare shutdown needs no scope around it.
        recorder = Recorder()
        with recorder.patch():
            await session().shutdown()

        self.assertEqual(recorder.calls, ["delete"])
        self.assertEqual(len(recorder.clients), 1)

    async def test_shutdown_after_the_scope_reuses_the_shared_client(self):
        recorder = Recorder()
        with recorder.patch():
            s = session()
            async with s:
                pass
            await s.shutdown()

        self.assertEqual(len(recorder.clients), 1)
        self.assertEqual(recorder.calls, ["delete", "delete"])

    async def test_an_already_deleted_session_is_not_an_error(self):
        recorder = Recorder(deletes_missing=True)
        with recorder.patch():
            async with session():
                pass
            await session().shutdown()

        self.assertEqual(recorder.calls, ["delete", "delete"])

    async def test_exiting_twice_shuts_down_once_more_but_never_raises(self):
        recorder = Recorder()
        with recorder.patch():
            s = session()
            await s.__aenter__()
            await s.__aexit__(None, None, None)
            await s.__aexit__(None, None, None)
        self.assertEqual(recorder.calls, ["delete", "delete"])

    def test_two_regions_in_one_session_are_rejected_at_construction(self):
        # One client serves both arns, so a cross-region pair could never have worked.
        with self.assertRaises(AssertionError):
            AgentCoreHttpSession(
                "s1",
                session_state=PersistentDict({"session_id": "s1"}, persister=NullPersister()),
                runtime_arn=RUNTIME_ARN,
                capacity_provider_arn=PROVIDER_ARN.replace(REGION, "eu-west-1"),
            )


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
