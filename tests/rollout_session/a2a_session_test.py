#!/usr/bin/env python
"""End-to-end tests for the A2A rollout protocol."""

import json
from unittest import IsolatedAsyncioTestCase, mock

import httpx
from a2a.client import Client
from bedrock_agentcore.runtime.a2a import build_a2a_app
from bedrock_agentcore.runtime.models import PingStatus

from agentcore_rl_toolkit.aws_tools.persistent_dict import NullPersister, PersistentDict
from agentcore_rl_toolkit.rollout_session import a2a_client as client_mod
from agentcore_rl_toolkit.rollout_session.a2a_client import A2ARolloutSession, RolloutA2AError, build_a2a_client
from agentcore_rl_toolkit.rollout_session.a2a_executor import RolloutAgentExecutor, make_ping_handler
from agentcore_rl_toolkit.rollout_session.wire import RolloutDumpResponse

SESSION_ID = "verl_" + "0" * 32


def meta() -> PersistentDict:
    return PersistentDict(persister=NullPersister())


class RecordingExecutor(RolloutAgentExecutor):
    def __init__(self, *, rollout_dump: RolloutDumpResponse, setup_error=None, rollout_error=None):
        super().__init__()
        self._dump = rollout_dump
        self._setup_error = setup_error
        self._rollout_error = rollout_error
        self.setup_inputs: list[dict] = []
        self.rollout_inputs: list[dict] = []
        self.busy_during_rollout: bool | None = None
        self.cleaned_up = False

    def run_setup(self, task_input: dict) -> None:
        self.setup_inputs.append(task_input)
        if self._setup_error is not None:
            raise self._setup_error

    def run_rollout(self, task_input: dict) -> RolloutDumpResponse:
        self.rollout_inputs.append(task_input)
        self.busy_during_rollout = self.ping_status() == PingStatus.HEALTHY_BUSY
        if self._rollout_error is not None:
            raise self._rollout_error
        return self._dump

    def cleanup(self) -> None:
        self.cleaned_up = True


class ASGISession(A2ARolloutSession):
    def __init__(self, session_id: str, session_state: PersistentDict, app):
        super().__init__(session_id, session_state)
        self._httpx = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://a2a.test")

    async def _client(self) -> Client:
        return build_a2a_client(self._httpx, "http://a2a.test/")

    async def shutdown(self) -> None:
        await self._cancel_if_unfinished()
        await self._httpx.aclose()


def serve(executor: RolloutAgentExecutor):
    return build_a2a_app(executor, ping_handler=make_ping_handler(executor))


class FastPollMixin:
    def setUp(self):
        super().setUp()
        import asyncio

        real_sleep = asyncio.sleep

        async def fast_sleep(_seconds):
            await real_sleep(0)  # sleep(0) still yields so the executor task runs

        patcher = mock.patch("asyncio.sleep", fast_sleep)
        patcher.start()
        self.addCleanup(patcher.stop)


class A2ARolloutFlowTest(FastPollMixin, IsolatedAsyncioTestCase):
    async def test_setup_then_rollout_completes_with_dump(self):
        dump = RolloutDumpResponse(task_output={"diff": "patch"}, reward=1.0, exception=None)
        executor = RecordingExecutor(rollout_dump=dump)
        async with ASGISession(SESSION_ID, meta(), serve(executor)) as session:
            await session.setup({"phase": "setup", "repo": "org/repo"})
            self.assertEqual(executor.setup_inputs, [{"phase": "setup", "repo": "org/repo"}])
            self.assertEqual(executor.rollout_inputs, [])

            result = await session.run({"phase": "rollout"})

        self.assertEqual(executor.rollout_inputs, [{"phase": "rollout"}])
        self.assertEqual(result.task_output, {"diff": "patch"})
        self.assertEqual(result.reward, 1.0)
        self.assertIsNone(result.exception)
        self.assertIs(executor.busy_during_rollout, True)
        self.assertEqual(executor.ping_status(), PingStatus.HEALTHY)

    async def test_in_container_failure_returns_dump_with_exception(self):
        dump = RolloutDumpResponse(task_output=None, reward=0.0, exception="pytest crashed")
        executor = RecordingExecutor(rollout_dump=dump)
        async with ASGISession(SESSION_ID, meta(), serve(executor)) as session:
            await session.setup({"phase": "setup"})
            result = await session.run({"phase": "rollout"})
        self.assertEqual(result.exception, "pytest crashed")
        self.assertEqual(result.reward, 0.0)

    async def test_setup_that_raises_surfaces_as_error(self):
        executor = RecordingExecutor(
            rollout_dump=RolloutDumpResponse(task_output=None, reward=None, exception=None),
            setup_error=RuntimeError("unpack script exploded"),
        )
        async with ASGISession(SESSION_ID, meta(), serve(executor)) as session:
            with self.assertRaises(RolloutA2AError) as ctx:
                await session.setup({"phase": "setup"})
        self.assertIn("unpack script exploded", str(ctx.exception))

    async def test_rollout_that_raises_comes_back_as_dump_exception(self):
        executor = RecordingExecutor(
            rollout_dump=RolloutDumpResponse(task_output=None, reward=None, exception=None),
            rollout_error=RuntimeError("model endpoint unreachable"),
        )
        async with ASGISession(SESSION_ID, meta(), serve(executor)) as session:
            await session.setup({"phase": "setup"})
            result = await session.run({"phase": "rollout"})
        self.assertIsNotNone(result.exception)
        self.assertIn("model endpoint unreachable", result.exception)


class SessionHeaderTest(FastPollMixin, IsolatedAsyncioTestCase):
    async def test_session_id_rides_on_every_request(self):
        seen: list[str] = []

        dump = RolloutDumpResponse(task_output={"diff": "ok"}, reward=1.0, exception=None)

        class ContextRecordingExecutor(RecordingExecutor):
            async def execute(self, context, event_queue):
                seen.append(context.context_id)
                await super().execute(context, event_queue)

        executor = ContextRecordingExecutor(rollout_dump=dump)
        async with ASGISession(SESSION_ID, meta(), serve(executor)) as session:
            await session.setup({"phase": "setup"})
            await session.run({"phase": "rollout"})

        self.assertEqual(seen, [SESSION_ID, SESSION_ID])


class FlakyTransport(httpx.AsyncBaseTransport):
    """Answer the first requests with synthetic ACR error responses, then delegate to a real app."""

    def __init__(self, inner: httpx.AsyncBaseTransport, errors: list[tuple[int, int | None]]):
        self._inner = inner
        self._errors = list(errors)
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self._errors:
            status, code = self._errors.pop(0)
            request_id = json.loads(request.content).get("id") if request.content else None
            error = {"code": code, "message": "synthetic"} if code is not None else {"message": "synthetic"}
            body = {"jsonrpc": "2.0", "id": request_id, "error": error}
            return httpx.Response(status, json=body, request=request)
        return await self._inner.handle_async_request(request)


class TransientRetryTest(FastPollMixin, IsolatedAsyncioTestCase):
    """Retry classification driven through real HTTP responses, not hand-built exception strings."""

    async def _send_setup_over(self, errors: list[tuple[int, int | None]]) -> tuple[str, FlakyTransport]:
        executor = RecordingExecutor(rollout_dump=RolloutDumpResponse(task_output=None, reward=1.0, exception=None))
        transport = FlakyTransport(httpx.ASGITransport(app=serve(executor)), errors)
        httpx_client = httpx.AsyncClient(transport=transport, base_url="http://a2a.test")
        try:
            client = build_a2a_client(httpx_client, "http://a2a.test/")
            task_id = await client_mod.send_setup(client, {"phase": "setup"}, session_id=SESSION_ID)
        finally:
            await httpx_client.aclose()
        return task_id, transport

    async def test_retries_conflict_returned_as_http_409(self):
        # ACR returns RetryableConflictException as HTTP 409 with a -32054 body; raise_for_status
        # fires first, so this only retries if we classify off the underlying response, not the message.
        task_id, transport = await self._send_setup_over([(409, client_mod.SESSION_IN_PROGRESS_CODE)])
        self.assertTrue(task_id)
        self.assertEqual(transport.calls, 2)  # one 409, one success

    async def test_retries_throttle_returned_as_http_429(self):
        task_id, transport = await self._send_setup_over([(429, client_mod.THROTTLE_CODE)])
        self.assertTrue(task_id)
        self.assertEqual(transport.calls, 2)

    async def test_retries_conflict_from_http_200_error_body(self):
        # The other wire shape: HTTP 200 carrying a JSON-RPC error body, which the SDK stringifies
        # as "JSON-RPC Error -32054: ...". Classification falls back to the message here.
        task_id, transport = await self._send_setup_over([(200, client_mod.SESSION_IN_PROGRESS_CODE)])
        self.assertTrue(task_id)
        self.assertEqual(transport.calls, 2)

    async def test_fatal_error_is_not_retried(self):
        with self.assertRaises(RolloutA2AError):
            await self._send_setup_over([(400, -32000)])
