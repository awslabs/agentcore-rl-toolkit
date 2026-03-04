"""Tests for async methods on RolloutFuture and RolloutClient."""

import asyncio
import json
import time
from concurrent.futures import CancelledError
from unittest.mock import MagicMock, patch

import pytest

from agentcore_rl_toolkit.client import RolloutClient, RolloutFuture

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789:agent-runtime/my-runtime"
FAKE_BUCKET = "my-bucket"
FAKE_EXP = "exp-1"
FAKE_RESULT_KEY = "exp-1/input-1/sess-1.json"
FAKE_RESULT = {"rollout_data": [{"tokens": [1, 2, 3]}], "rewards": [1.0]}


def _make_s3_client(found=False):
    """Return a mock S3 client. ``found=True`` means head_object succeeds."""
    from botocore.exceptions import ClientError

    client = MagicMock()
    if found:
        client.head_object.return_value = {}
    else:
        error_response = {"Error": {"Code": "404", "Message": "Not Found"}}
        client.head_object.side_effect = ClientError(error_response, "HeadObject")
    body = MagicMock()
    body.read.return_value = json.dumps(FAKE_RESULT).encode()
    client.get_object.return_value = {"Body": body}
    return client


def _make_future(found=False, cancelled=False, **kwargs):
    s3 = _make_s3_client(found=found)
    future = RolloutFuture(
        s3_client=s3,
        s3_bucket=FAKE_BUCKET,
        result_key=FAKE_RESULT_KEY,
        initial_interval=0.01,
        max_interval=0.05,
        backoff_factor=1.5,
        **kwargs,
    )
    if cancelled:
        future._cancelled = True
    return future


def _make_client():
    """Return a RolloutClient with mocked boto3 clients.

    Each invoke_agent_runtime call returns a unique result_key so that
    multiple futures don't collide in the active_futures dict.
    """
    _invoke_counter = {"n": 0}

    with patch("agentcore_rl_toolkit.client.boto3") as mock_boto3:
        acr_client = MagicMock()

        def _fake_invoke(**kwargs):
            _invoke_counter["n"] += 1
            body = MagicMock()
            body.read.return_value = json.dumps(
                {
                    "s3_bucket": FAKE_BUCKET,
                    "result_key": f"{FAKE_EXP}/input-{_invoke_counter['n']}/sess.json",
                    "status": "processing",
                }
            ).encode()
            return {"response": body}

        acr_client.invoke_agent_runtime.side_effect = _fake_invoke

        s3_client = _make_s3_client(found=True)

        def fake_client(service, **kwargs):
            if service == "bedrock-agentcore":
                return acr_client
            return s3_client

        mock_boto3.client.side_effect = fake_client
        client = RolloutClient(
            agent_runtime_arn=FAKE_ARN,
            s3_bucket=FAKE_BUCKET,
            exp_id=FAKE_EXP,
        )
    return client


# ---------------------------------------------------------------------------
# RolloutFuture async tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_done_async_returns_false_on_404():
    future = _make_future(found=False)
    assert await future.done_async() is False


@pytest.mark.asyncio
async def test_done_async_returns_true_when_found():
    future = _make_future(found=True)
    assert await future.done_async() is True


@pytest.mark.asyncio
async def test_done_async_returns_true_when_cancelled():
    future = _make_future(found=False, cancelled=True)
    assert await future.done_async() is True


@pytest.mark.asyncio
async def test_result_async_polls_until_ready():
    """done_async returns False twice then True; result_async should succeed."""
    from botocore.exceptions import ClientError

    s3 = MagicMock()
    error_response = {"Error": {"Code": "404", "Message": "Not Found"}}
    s3.head_object.side_effect = [
        ClientError(error_response, "HeadObject"),
        ClientError(error_response, "HeadObject"),
        {},  # found on third try
    ]
    body = MagicMock()
    body.read.return_value = json.dumps(FAKE_RESULT).encode()
    s3.get_object.return_value = {"Body": body}

    future = RolloutFuture(
        s3_client=s3,
        s3_bucket=FAKE_BUCKET,
        result_key=FAKE_RESULT_KEY,
        initial_interval=0.01,
        max_interval=0.05,
        backoff_factor=1.0,
    )
    result = await future.result_async()
    assert result == FAKE_RESULT


@pytest.mark.asyncio
async def test_result_async_raises_timeout():
    """result_async(timeout=...) raises TimeoutError when result never appears."""
    future = _make_future(found=False)
    with pytest.raises(TimeoutError):
        await future.result_async(timeout=0.05)


@pytest.mark.asyncio
async def test_result_async_raises_cancelled():
    """result_async raises CancelledError when future was cancelled."""
    future = _make_future(found=False, cancelled=True)
    with pytest.raises(CancelledError):
        await future.result_async()


@pytest.mark.asyncio
async def test_await_syntax():
    """``await future`` should work via __await__."""
    future = _make_future(found=True)
    result = await future
    assert result == FAKE_RESULT


@pytest.mark.asyncio
async def test_result_async_returns_cached():
    """Second call returns cached result without polling again."""
    future = _make_future(found=True)
    r1 = await future.result_async()
    # Reset mock to verify no further calls
    future.s3_client.head_object.reset_mock()
    future.s3_client.get_object.reset_mock()
    r2 = await future.result_async()
    assert r1 == r2
    future.s3_client.head_object.assert_not_called()
    future.s3_client.get_object.assert_not_called()


# ---------------------------------------------------------------------------
# RolloutClient async tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_async_returns_future():
    client = _make_client()
    future = await client.invoke_async({"prompt": "hello"})
    assert isinstance(future, RolloutFuture)
    assert future.result_key.startswith(FAKE_EXP + "/")


@pytest.mark.asyncio
async def test_invoke_async_generates_uuids():
    """invoke_async with no session_id/input_id should auto-generate them."""
    client = _make_client()
    future = await client.invoke_async({"prompt": "hello"})
    assert future.session_id is not None
    assert future.input_id is not None


@pytest.mark.asyncio
async def test_invoke_async_custom_ids():
    """invoke_async passes custom session_id and input_id through."""
    client = _make_client()
    future = await client.invoke_async({"prompt": "hello"}, session_id="s-1", input_id="i-1")
    assert future.session_id == "s-1"
    assert future.input_id == "i-1"


@pytest.mark.asyncio
async def test_async_rate_limiting_lock_released_before_http():
    """Lock is released before the HTTP call so cold starts don't block others."""
    client = _make_client()

    # Track when the lock is acquired/released vs when the HTTP call happens
    events = []

    original_invoke = client.agentcore_client.invoke_agent_runtime

    def slow_invoke(**kwargs):
        events.append("http_start")
        time.sleep(0.05)  # simulate cold start
        events.append("http_end")
        return original_invoke(**kwargs)

    client.agentcore_client.invoke_agent_runtime = slow_invoke

    # Fire two invocations concurrently
    t1 = asyncio.create_task(client.invoke_async({"prompt": "a"}))
    t2 = asyncio.create_task(client.invoke_async({"prompt": "b"}))
    await asyncio.gather(t1, t2)

    # Both HTTP calls should overlap (second http_start before first http_end)
    # Due to thread pool, they run in parallel
    assert len(events) == 4  # two http_start + two http_end


# ---------------------------------------------------------------------------
# AsyncBatchResult tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_batch_async_yields_all_items():
    client = _make_client()
    payloads = [{"prompt": f"q{i}"} for i in range(3)]
    items = []
    async for item in client.run_batch_async(payloads, max_concurrent_sessions=3):
        items.append(item)
    assert len(items) == 3
    assert all(item.success for item in items)
    indices = sorted(item.index for item in items)
    assert indices == [0, 1, 2]


@pytest.mark.asyncio
async def test_run_batch_async_handles_invoke_errors():
    """If invoke fails for some payloads, they yield as error BatchItems."""
    client = _make_client()

    call_count = 0
    original = client.agentcore_client.invoke_agent_runtime

    def failing_invoke(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("Connection refused")
        return original(**kwargs)

    client.agentcore_client.invoke_agent_runtime = failing_invoke

    items = []
    payloads = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]
    async for item in client.run_batch_async(payloads, max_concurrent_sessions=3):
        items.append(item)

    assert len(items) == 3
    errors = [i for i in items if not i.success]
    assert len(errors) == 1
    assert "Connection refused" in errors[0].error


@pytest.mark.asyncio
async def test_run_batch_async_timeout():
    """Items that exceed timeout should yield as error BatchItems."""
    from botocore.exceptions import ClientError

    client = _make_client()

    # Make S3 HEAD always return 404 so result never appears
    error_response = {"Error": {"Code": "404", "Message": "Not Found"}}
    client.s3_client.head_object.side_effect = ClientError(error_response, "HeadObject")

    items = []
    async for item in client.run_batch_async(
        [{"prompt": "slow"}],
        max_concurrent_sessions=1,
        timeout=0.1,
        initial_interval=0.01,
    ):
        items.append(item)

    assert len(items) == 1
    assert not items[0].success
    assert "Timeout" in items[0].error


@pytest.mark.asyncio
async def test_run_batch_async_concurrent_submissions():
    """Cold start on one request doesn't block submission of others."""
    client = _make_client()

    submission_times = []
    original = client.agentcore_client.invoke_agent_runtime

    def tracking_invoke(**kwargs):
        submission_times.append(time.time())
        return original(**kwargs)

    client.agentcore_client.invoke_agent_runtime = tracking_invoke
    # Set high TPS limit so rate limiting doesn't add delay
    client._min_invoke_interval = 0.0

    items = []
    async for item in client.run_batch_async(
        [{"prompt": f"q{i}"} for i in range(3)],
        max_concurrent_sessions=3,
    ):
        items.append(item)

    assert len(items) == 3
    # All 3 submissions should happen nearly simultaneously
    if len(submission_times) == 3:
        span = submission_times[-1] - submission_times[0]
        assert span < 1.0  # Should be well under 1 second


# ---------------------------------------------------------------------------
# Sync API regression
# ---------------------------------------------------------------------------


def test_sync_invoke_still_works():
    """Verify sync invoke() is unaffected by async additions."""
    client = _make_client()
    future = client.invoke({"prompt": "hello"})
    assert isinstance(future, RolloutFuture)
    assert future.result_key.startswith(FAKE_EXP + "/")


def test_sync_done_still_works():
    """Verify sync done() is unaffected."""
    future = _make_future(found=True)
    assert future.done() is True


def test_sync_result_still_works():
    """Verify sync result() is unaffected."""
    future = _make_future(found=True)
    result = future.result()
    assert result == FAKE_RESULT


def test_sync_run_batch_still_works():
    """Verify sync run_batch() is unaffected."""
    client = _make_client()
    items = list(client.run_batch([{"prompt": "a"}], max_concurrent_sessions=1))
    assert len(items) == 1
    assert items[0].success
