"""HTTP contract tests: only AWS routing is absent."""

import asyncio
import multiprocessing
import os
import threading

import httpx
import pytest
import pytest_asyncio

from agentcore_rl_toolkit import AgentCoreRuntimeApp

SESSION_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"


@pytest.fixture
def app(tmp_path):
    return AgentCoreRuntimeApp(state_dir=tmp_path)


@pytest_asyncio.fixture
async def client(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://agent") as client:
        yield client


async def invoke(client, invocation_id="inv-1", *, operation="start", background=None, **payload):
    envelope = {"version": 1, "operation": operation, "invocation_id": invocation_id}
    if background is not None:
        envelope["background"] = background
    return await client.post(
        "/invocations",
        json={**payload, "_agentcore_runtime": envelope},
        headers={SESSION_HEADER: "session-a"},
    )


async def terminal(client, invocation_id="inv-1"):
    async with asyncio.timeout(5):
        while True:
            response = await invoke(client, invocation_id, operation="get")
            response.raise_for_status()
            if response.json()["status"] != "in_progress":
                return response.json()
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_ordinary_http_and_protocol_share_async_handler(app, client):
    @app.entrypoint
    async def handler(payload, context):
        if payload.get("stream"):
            return (chunk for chunk in [{"text": "ordinary"}])
        return {"payload": payload, "session": context.session_id}

    ordinary = await client.post("/invocations", json={"plain": True})
    assert ordinary.status_code == 200
    assert ordinary.json() == {"payload": {"plain": True}, "session": None}
    stream = await client.post("/invocations", json={"stream": True})
    assert stream.headers["content-type"].startswith("text/event-stream")
    assert "ordinary" in stream.text

    response = await invoke(client, _rollout={"model_id": "example"})
    assert response.json() == {
        "version": 1,
        "invocation_id": "inv-1",
        "status": "completed",
        "result": {"payload": {"_rollout": {"model_id": "example"}}, "session": "session-a"},
    }
    assert (await invoke(client, operation="get")).json() == response.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("async_handler", [False, True])
async def test_plain_background_requests_preserve_payload_and_start_separate_invocations(tmp_path, async_handler):
    app = AgentCoreRuntimeApp(background=True, state_dir=tmp_path)
    release = threading.Event()
    calls = []

    def run(payload, context):
        calls.append((payload, context.session_id))
        assert release.wait(5)
        return {"answer": 42}

    async def run_async(payload, context):
        return await asyncio.to_thread(run, payload, context)

    app.entrypoint(run_async if async_handler else run)
    payload = {"prompt": "same", "metadata": {"opaque": "unchanged"}}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://agent") as client:
        try:
            responses = [
                await client.post("/invocations", json=payload, headers={SESSION_HEADER: "session-a"}) for _ in range(2)
            ]
            for response in responses:
                assert response.status_code == 200
                assert response.json()["status"] == "in_progress"
            ids = [response.json()["invocation_id"] for response in responses]
            assert ids[0] != ids[1]
            assert (await client.get("/ping")).json()["status"] == "HealthyBusy"
        finally:
            release.set()
        for invocation_id in ids:
            assert (await terminal(client, invocation_id))["result"] == {"answer": 42}
        assert calls == [(payload, "session-a"), (payload, "session-a")]
        assert (await client.get("/ping")).json()["status"] == "Healthy"


@pytest.mark.asyncio
async def test_explicit_envelope_overrides_app_background(tmp_path):
    app = AgentCoreRuntimeApp(background=True, state_dir=tmp_path)
    calls = []

    @app.entrypoint
    def handler(payload):
        calls.append(payload)
        return payload

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://agent") as client:
        for invocation_id, background in [("default", None), ("foreground", False)]:
            response = await invoke(client, invocation_id, background=background, value=42)
            assert response.json()["status"] == "completed"
            assert response.json()["result"] == {"value": 42}
            assert (await invoke(client, invocation_id, operation="get")).json() == response.json()

        invalid = await client.post("/invocations", json={"_agentcore_runtime": None})
        assert invalid.status_code == 400
        assert calls == [{"value": 42}, {"value": 42}]


@pytest.mark.asyncio
async def test_background_deduplicates_by_id_and_keeps_busy_through_publication(app, client, monkeypatch):
    entered, release, publishing, publish = (threading.Event() for _ in range(4))
    calls = []
    finish = app._store.finish

    @app.entrypoint
    def handler(payload):
        calls.append(payload)
        entered.set()
        assert release.wait(5)
        return {"answer": 42}

    def delayed_finish(*args):
        publishing.set()
        assert publish.wait(5)
        finish(*args)

    monkeypatch.setattr(app._store, "finish", delayed_finish)
    try:
        first, retry = await asyncio.gather(
            invoke(client, background=True, prompt="same"),
            invoke(client, background=True, prompt="same"),
        )
        assert first.json()["status"] == retry.json()["status"] == "in_progress"
        assert await asyncio.to_thread(entered.wait, 5)
        assert (await invoke(client, operation="get")).json()["status"] == "in_progress"
        assert (await client.get("/ping")).json()["status"] == "HealthyBusy"

        release.set()
        assert await asyncio.to_thread(publishing.wait, 5)
        assert (await invoke(client, operation="get")).json()["status"] == "in_progress"
        assert (await client.get("/ping")).json()["status"] == "HealthyBusy"
        publish.set()
        result = await terminal(client)
        assert result["result"] == {"answer": 42}

        assert (await invoke(client, prompt="different")).json() == result
        assert len(calls) == 1
        assert (await invoke(client, "inv-2", prompt="same")).json()["result"] == {"answer": 42}
        assert len(calls) == 2
        assert (await client.get("/ping")).json()["status"] == "Healthy"
    finally:
        release.set()
        publish.set()


@pytest.mark.asyncio
async def test_get_during_completion_does_not_report_interrupted(app, client, monkeypatch):
    release_handler, read_started, release_read = (threading.Event() for _ in range(3))

    @app.entrypoint
    def handler(payload):
        assert release_handler.wait(5)
        return {"answer": 42}

    await invoke(client, background=True)
    execution = app._live["inv-1"]
    read = app._store.read

    def delayed_read(*args):
        snapshot = read(*args)
        read_started.set()
        assert release_read.wait(5)
        return snapshot

    monkeypatch.setattr(app._store, "read", delayed_read)
    poll = asyncio.create_task(invoke(client, operation="get"))
    try:
        assert await asyncio.to_thread(read_started.wait, 5)
        # Finish and remove the live task before get resumes with its old record.
        release_handler.set()
        await asyncio.wait_for(asyncio.shield(execution), 5)
        release_read.set()

        response = await poll
        response.raise_for_status()
        assert response.json()["status"] in {"in_progress", "completed"}
        assert (await invoke(client, operation="get")).json()["result"] == {"answer": 42}
    finally:
        release_handler.set()
        release_read.set()
        await asyncio.gather(poll, execution, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("disconnect_during", ["acceptance", "execution"])
async def test_disconnected_foreground_request_does_not_abandon_execution(app, client, monkeypatch, disconnect_during):
    entered, release = threading.Event(), threading.Event()
    start = app._store.start

    def pause():
        entered.set()
        assert release.wait(5)

    def delayed_start(*args):
        start(*args)
        pause()

    if disconnect_during == "acceptance":
        monkeypatch.setattr(app._store, "start", delayed_start)

    @app.entrypoint
    def handler(payload):
        if disconnect_during == "execution":
            pause()
        return "survived"

    request = asyncio.create_task(invoke(client))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        release.set()
        result = await terminal(client)
        assert result["status"] == "completed"
        assert result["result"] == "survived"
    finally:
        release.set()
        if not request.done():
            await request


def _run_app_until_process_exit(state_dir):
    """Publish one result, then exit abruptly during a second invocation."""
    app = AgentCoreRuntimeApp(state_dir=state_dir)

    @app.entrypoint
    def handler(payload):
        if payload["crash"]:
            os._exit(0)
        return {"saved": True}

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://agent") as client:
            for invocation_id, crash in [("finished", False), ("unfinished", True)]:
                response = await invoke(client, invocation_id, crash=crash, secret="input-must-not-be-persisted")
                response.raise_for_status()
        raise AssertionError("handler did not exit")

    asyncio.run(run())


@pytest.mark.asyncio
async def test_recovery_after_process_exit(app, client, tmp_path):
    child = multiprocessing.get_context("spawn").Process(target=_run_app_until_process_exit, args=(tmp_path,))
    child.start()
    try:
        await asyncio.to_thread(child.join, 30)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.kill()
            await asyncio.to_thread(child.join, 5)
        child.close()

    @app.entrypoint
    def handler(payload):
        pytest.fail("recovery re-entered the handler")

    assert (await invoke(client, "finished", operation="get")).json()["result"] == {"saved": True}
    assert (await invoke(client, "unfinished", operation="get")).json()["status"] == "interrupted"
    assert (await invoke(client, "unfinished")).json()["status"] == "interrupted"
    assert (await invoke(client, "unknown", operation="get")).json()["status"] == "not_found"
    assert all("input-must-not-be-persisted" not in path.read_text() for path in tmp_path.rglob("*.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure, error", [("handler", "bad input"), ("serialization", "not JSON serializable")])
async def test_failures_are_persisted_as_terminal_errors(app, client, failure, error):
    @app.entrypoint
    def handler(payload):
        if failure == "handler":
            raise ValueError("bad input")
        return object()

    response = await invoke(client)
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "completed"
    assert error in result["error"]
    assert "result" not in result
    assert (await invoke(client, operation="get")).json() == result


@pytest.mark.asyncio
async def test_storage_errors_never_report_success_or_not_found(app, client, monkeypatch):
    @app.entrypoint
    def handler(payload):
        return "cannot save"

    def unavailable(*args):
        raise PermissionError("storage unavailable")

    monkeypatch.setattr(app._store, "finish", unavailable)
    response = await invoke(client)
    assert response.status_code == 500
    assert response.json() == {"error": "storage unavailable"}
    assert (await invoke(client, operation="get")).json()["status"] == "interrupted"
    assert (await client.get("/ping")).json()["status"] == "Healthy"

    monkeypatch.setattr(app._store, "read", unavailable)
    assert (await invoke(client, operation="get")).status_code == 500


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [
        {"invocation_id": "../escape"},
        {"version": True},
        {"background": "false"},
        {"backgound": True},
        {"operation": "delete"},
    ],
)
async def test_invalid_protocol_requests_do_not_enter_handler(app, client, fields):
    @app.entrypoint
    def handler(payload):
        pytest.fail("invalid request entered handler")

    response = await client.post(
        "/invocations",
        json={"_agentcore_runtime": {"version": 1, "operation": "start", "invocation_id": "inv-1", **fields}},
    )
    assert response.status_code == 400
    assert isinstance(response.json()["error"], str)
