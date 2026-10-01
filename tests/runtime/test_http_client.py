"""Exercise the public client through the app's real HTTP endpoint."""

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from bedrock_agentcore.runtime import BedrockAgentCoreApp

from agentcore_rl_toolkit import AgentCoreHttpClient, AgentCoreRuntimeApp, InvocationError
from agentcore_rl_toolkit.rollout_session.agentcore_http_session import AgentCoreHttpSession

SID = "session-00000000-0000-0000-000000000001"
ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/test"


class Body:
    def __init__(self, content):
        self.content = content
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def read(self):
        return self.content


@asynccontextmanager
async def connected(app):
    bodies, requests = [], []
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://agent") as http:

        async def invoke(**kwargs):
            requests.append(json.loads(kwargs["payload"]))
            response = await http.post(
                "/invocations",
                content=kwargs["payload"],
                headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": kwargs["runtimeSessionId"]},
            )
            body = Body(response.content)
            bodies.append(body)
            return {"statusCode": response.status_code, "response": body}

        client = AgentCoreHttpClient(ARN)
        aws = SimpleNamespace(invoke_agent_runtime=invoke, stop_runtime_session=AsyncMock())
        client._aws_client = aws
        yield client, aws, requests
    assert all(body.closed for body in bodies)


@pytest.mark.asyncio
async def test_handle_recovery_wait_timeout_and_payload_passthrough(tmp_path, monkeypatch):
    monkeypatch.setattr("agentcore_rl_toolkit.runtime.client.POLL_INTERVAL", 0.05)
    app = AgentCoreRuntimeApp(state_dir=tmp_path)

    @app.entrypoint
    async def handler(payload):
        # The SDK runs the handler on another loop, so use a thread event.
        if payload.get("blocked"):
            await asyncio.to_thread(gate.wait, 5)
        return payload

    gate = threading.Event()
    async with connected(app) as (client, aws, requests):
        payload = {"blocked": True, "_config": {"model_id": "first"}}
        handle = await client.invoke(payload, session_id=SID, invocation_id="same", background=True)
        try:
            with pytest.raises(TimeoutError):
                await handle.result(timeout=0.01)
            other = await client.invoke({"_config": {"model_id": "second"}}, session_id=SID)
            assert other == {"_config": {"model_id": "second"}}
            assert (await handle.status())["status"] == "in_progress"
            gate.set()
            restored = client.get_invocation(session_id=SID, invocation_id="same")
            assert await restored.result(timeout=5) == payload
            assert payload == {"blocked": True, "_config": {"model_id": "first"}}
            aws.stop_runtime_session.assert_not_called()
        finally:
            gate.set()


@pytest.mark.asyncio
async def test_plain_app_rollout_returns_result_without_polling():
    app = BedrockAgentCoreApp()

    @app.entrypoint
    def handler(payload):
        return {"status": "ok", "reward": 0.5, "metrics": {"turns": 2}, "prompt": payload["prompt"]}

    async with connected(app) as (client, aws, requests):
        async with AgentCoreHttpSession(SID, client=client) as session:
            await session.setup({})
            assert requests == []
            dump = await session.run({"prompt": "math"})
        assert dump.reward == 0.5
        assert dump.task_output == {"status": "ok", "reward": 0.5, "metrics": {"turns": 2}, "prompt": "math"}
        assert len(requests) == 1
        await session.shutdown()
        aws.stop_runtime_session.assert_awaited_once_with(agentRuntimeArn=ARN, runtimeSessionId=SID)


@pytest.mark.asyncio
async def test_failed_and_cancelled_rollouts_stop_their_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr("agentcore_rl_toolkit.runtime.client.POLL_INTERVAL", 0.05)
    app = AgentCoreRuntimeApp(state_dir=tmp_path)

    @app.entrypoint
    def handler(payload):
        raise ValueError("bad input")

    async with connected(app) as (client, aws, _), asyncio.timeout(5):
        async with AgentCoreHttpSession(SID, client=client) as session:
            dump = await session.run({})
        assert "bad input" in dump.exception
        assert dump.reward is None
        with pytest.raises(InvocationError, match="not_found"):
            await client.get_invocation(session_id=SID, invocation_id="unknown").result()

        client.invoke = AsyncMock(side_effect=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            async with AgentCoreHttpSession("another-session", client=client) as session:
                await session.run({})
        assert [call.kwargs["runtimeSessionId"] for call in aws.stop_runtime_session.await_args_list] == [
            SID,
            "another-session",
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["get", "stop"])
async def test_cancelled_rate_limit_wait_does_not_send_to_aws(operation):
    limiter = SimpleNamespace(wait_async=AsyncMock(side_effect=asyncio.CancelledError()))
    client = AgentCoreHttpClient(ARN, request_rate_limiter=limiter)
    aws = SimpleNamespace(invoke_agent_runtime=AsyncMock(), stop_runtime_session=AsyncMock())
    client._aws_client = aws
    with pytest.raises(asyncio.CancelledError):
        if operation == "get":
            await client.get_invocation(session_id=SID, invocation_id="inv").status()
        else:
            await client.stop_session(SID)
    aws.invoke_agent_runtime.assert_not_awaited()
    aws.stop_runtime_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_client_does_not_resubmit_after_a_lost_response():
    client = AgentCoreHttpClient(ARN)
    invoke = AsyncMock(side_effect=TimeoutError("response lost"))
    client._aws_client = SimpleNamespace(invoke_agent_runtime=invoke)
    with pytest.raises(TimeoutError, match="response lost"):
        await client.invoke({}, session_id=SID)
    invoke.assert_awaited_once()
