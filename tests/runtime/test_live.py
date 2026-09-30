"""Opt-in tests for live_agent.py deployed from this checkout.

Set RUNTIME_TEST_ARN and deploy with idleRuntimeSessionTimeout=60. Tests create
fresh sessions and stop them afterward; they do not deploy or delete the Runtime.
The caller needs InvokeAgentRuntime, StopRuntimeSession, and GetAgentRuntime.
"""

import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import boto3
import pytest
from botocore.config import Config
from botocore.exceptions import ReadTimeoutError

from agentcore_rl_toolkit import AgentCoreHttpClient

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUNTIME_TEST_ARN"), reason="set RUNTIME_TEST_ARN for live ACR coverage"
)


def runtime_client(read_timeout=120):
    return boto3.client(
        "bedrock-agentcore",
        region_name=os.environ["RUNTIME_TEST_ARN"].split(":")[3],
        config=Config(connect_timeout=10, read_timeout=read_timeout, retries={"total_max_attempts": 1}),
    )


@pytest.fixture
def client():
    with closing(runtime_client()) as client:
        yield client


@pytest.fixture
def session(client):
    session_id = str(uuid.uuid4())
    try:
        yield session_id
    finally:
        client.stop_runtime_session(agentRuntimeArn=os.environ["RUNTIME_TEST_ARN"], runtimeSessionId=session_id)


def invoke(client, session_id, **payload):
    response = client.invoke_agent_runtime(
        agentRuntimeArn=os.environ["RUNTIME_TEST_ARN"],
        runtimeSessionId=session_id,
        contentType="application/json",
        payload=json.dumps(payload).encode(),
    )
    try:
        return json.loads(response["response"].read())
    finally:
        response["response"].close()


def request(client, session_id, invocation_id, operation="start", **payload):
    background = payload.pop("background", None)
    envelope = {"version": 1, "operation": operation, "invocation_id": invocation_id}
    if background is not None:
        envelope["background"] = background
    return invoke(client, session_id, _agentcore_runtime=envelope, **payload)


def result(client, session_id, invocation_id, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = request(client, session_id, invocation_id, "get")
        if snapshot["status"] != "in_progress":
            assert snapshot["status"] == "completed", snapshot
            return snapshot
        time.sleep(1)
    pytest.fail(f"invocation {invocation_id} did not complete within {timeout}s")


def test_live_results_retries_and_session_isolation(client, session):
    ordinary = invoke(client, session, value="ordinary")
    assert ordinary["session_id"] == session
    assert ordinary["value"] == "ordinary"
    with ThreadPoolExecutor(max_workers=2) as pool:
        starts = list(
            pool.map(
                lambda _: request(client, session, "retry", background=True, delay=5, value="original"),
                range(2),
            )
        )
    assert all(item["status"] == "in_progress" for item in starts)
    saved = result(client, session, "retry")
    assert saved["result"] == {**ordinary, "value": "original", "call_count": 2}
    assert request(client, session, "retry", value="must not run") == saved

    failed = request(client, session, "failure", fail=True)
    assert failed["error"] == "intentional test failure"
    assert request(client, session, "failure", "get") == failed

    other_session = str(uuid.uuid4())
    try:
        assert request(client, other_session, "retry", "get")["status"] == "not_found"
        other = request(client, other_session, "retry", value="separate")["result"]
        assert other["session_id"] == other_session
        assert other["instance_id"] != ordinary["instance_id"]
        assert other["call_count"] == 1
        assert request(client, session, "retry", "get") == saved
    finally:
        client.stop_runtime_session(agentRuntimeArn=os.environ["RUNTIME_TEST_ARN"], runtimeSessionId=other_session)


def test_live_foreground_disconnect(client, session):
    ordinary = invoke(client, session)  # Warm the session before forcing a timeout.
    with closing(runtime_client(read_timeout=1)) as impatient:
        with pytest.raises(ReadTimeoutError):
            request(impatient, session, "disconnect", delay=5, value="survived")
    saved = result(client, session, "disconnect")
    assert saved["result"] == {**ordinary, "value": "survived", "call_count": 2}
    assert request(client, session, "disconnect", value="must not run") == saved


def test_live_background_survives_idle_timeout(client, session):
    arn = os.environ["RUNTIME_TEST_ARN"]
    with closing(boto3.client("bedrock-agentcore-control", region_name=arn.split(":")[3])) as control:
        runtime = control.get_agent_runtime(agentRuntimeId=arn.rsplit("/", 1)[1])
    assert runtime["lifecycleConfiguration"]["idleRuntimeSessionTimeout"] == 60, "deploy with idle timeout 60s"

    ordinary = invoke(client, session)
    assert request(client, session, "busy", background=True, delay=90)["status"] == "in_progress"
    # No invocations during this interval: polling would itself keep the session warm.
    time.sleep(80)
    saved = result(client, session, "busy")
    assert saved["result"] == {**ordinary, "call_count": 2}


@pytest.mark.asyncio
async def test_live_http_client_recovers_a_handle():
    session_id = str(uuid.uuid4())
    async with AgentCoreHttpClient(os.environ["RUNTIME_TEST_ARN"]) as client:
        try:
            handle = await client.invoke({"delay": 2, "value": "http-client"}, session_id=session_id, background=True)
            restored = client.get_invocation(session_id=session_id, invocation_id=handle.invocation_id)
            result = await restored.result(timeout=30)
            assert result["value"] == "http-client"
            assert result["session_id"] == session_id
            assert await handle.result(timeout=30) == result
        finally:
            await client.stop_session(session_id)
