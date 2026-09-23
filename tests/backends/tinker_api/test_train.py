"""Validate rollout failure handling and evaluation results."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agentcore_rl_toolkit.backends.tinker_api.data import Episode
from agentcore_rl_toolkit.backends.tinker_api.train import (
    agent_reward,
    collect_batch,
    drain_rollouts,
    evaluate,
    rollout,
)
from agentcore_rl_toolkit.rollout_gateway.trace import TraceRecord


@pytest.mark.asyncio
async def test_processing_error_drains_other_rollouts_before_raising():
    finished = []

    async def fake_rollout(config, gateway, client, payload, group_index, sample_index):
        if sample_index == 0:
            raise ValueError("broken processing code")
        await asyncio.sleep(0.01)
        finished.append(sample_index)
        return "episode"

    with patch("agentcore_rl_toolkit.backends.tinker_api.train.rollout", fake_rollout):
        with pytest.raises(ValueError, match="broken processing code"):
            await collect_batch(SimpleNamespace(group_size=2), None, None, [{}])
    assert finished == [1]


@pytest.mark.parametrize("reward", [None, True, "1", float("nan"), float("inf")])
def test_invalid_rewards_are_not_silently_zeroed(reward):
    with pytest.raises(ValueError, match="finite scalar"):
        agent_reward({"rewards": reward})


@pytest.mark.parametrize(
    "failure",
    [
        {"status_code": 500, "stop_reason": "Model stopped generating due to maximum token limit."},
        TimeoutError("result timed out"),
    ],
)
@pytest.mark.parametrize("has_trace", [True, False])
@pytest.mark.asyncio
async def test_failed_invocation_preserves_trace_or_returns_unscored_failure(failure, has_trace):
    record = TraceRecord(token_ids=[10, 11, 12], loss_mask=[1, 1], logprobs=[-0.2, -0.3])
    gateway = SimpleNamespace(
        create_session=Mock(),
        finish_session=AsyncMock(return_value=[record] if has_trace else []),
        drop_session=AsyncMock(),
    )
    result_async = AsyncMock(side_effect=failure) if isinstance(failure, Exception) else AsyncMock(return_value=failure)
    future = SimpleNamespace(result_async=result_async)
    client = SimpleNamespace(invoke_async=AsyncMock(return_value=future))
    config = SimpleNamespace(max_new_tokens=2, max_context_tokens=8, rollout_timeout=60)
    episode = await rollout(config, gateway, client, {}, 0, 0)
    if not has_trace:
        assert episode.reward is None
        assert episode.records == []
        assert episode.error
    else:
        assert episode.reward == record.reward == 0.0
        assert episode.error
        assert episode.records == [record]
        assert record.token_ids == [10, 11, 12]
        assert record.loss_mask == [1, 1]
        assert record.logprobs == [-0.2, -0.3]
    gateway.drop_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancellation_is_not_converted_to_a_failed_episode():
    gateway = SimpleNamespace(
        create_session=Mock(), finish_session=AsyncMock(return_value=[]), drop_session=AsyncMock()
    )
    future = SimpleNamespace(result_async=AsyncMock(side_effect=asyncio.CancelledError()))
    client = SimpleNamespace(invoke_async=AsyncMock(return_value=future))
    config = SimpleNamespace(max_new_tokens=2, max_context_tokens=8, rollout_timeout=60)
    with pytest.raises(asyncio.CancelledError):
        await drain_rollouts([rollout(config, gateway, client, {}, 0, 0)])
    gateway.drop_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_evaluation_includes_partial_batch_once_per_prompt(tmp_path):
    seen = []

    async def fake_rollout(config, gateway, client, payload, group_index, sample_index, *, temperature):
        seen.append((payload["index"], group_index, sample_index, temperature))
        record = TraceRecord(
            token_ids=[10, 11], loss_mask=[1], logprobs=[-0.3], metadata={"truncated": sample_index == 1}
        )
        return Episode(
            f"eval-{sample_index}", payload["reward"], [record], error="token limit" if sample_index == 1 else None
        )

    config = SimpleNamespace(output_dir=str(tmp_path), evaluation_batch_size=2, evaluation_temperature=0.6)
    payloads = [{"index": i, "reward": reward} for i, reward in enumerate([1.0, 0.0, 1.0])]
    with patch("agentcore_rl_toolkit.backends.tinker_api.train.rollout", fake_rollout):
        metrics = await evaluate(config, None, None, payloads, step=10)
    assert sorted(seen) == [(i, i, i, 0.6) for i in range(3)]
    assert metrics["eval/reward_mean"] == pytest.approx(2 / 3)
    assert metrics["eval/episodes"] == 3
    assert metrics["eval/failed_episodes"] == metrics["eval/truncated_episodes"] == 1
    assert metrics["eval/failure_rate"] == pytest.approx(1 / 3)
    rows = [json.loads(row) for row in (tmp_path / "evaluation-0010.jsonl").read_text().splitlines()]
    rows.sort(key=lambda row: row["input_index"])
    assert [row["input_index"] for row in rows] == [0, 1, 2]
    assert rows[1]["error"] == "token limit"
    assert rows[1]["truncated"] is True


@pytest.mark.asyncio
async def test_evaluation_persists_completed_results_while_other_requests_are_pending(tmp_path):
    first_finished = asyncio.Event()
    release_slow = asyncio.Event()
    path = tmp_path / "evaluation-0010.jsonl"

    async def fake_rollout(config, gateway, client, payload, group_index, sample_index, *, temperature):
        if sample_index == 0:
            await release_slow.wait()
            return Episode("missing", None, [], error="502")
        first_finished.set()
        return Episode("good", 1, [TraceRecord(token_ids=[10, 11], loss_mask=[1], logprobs=[-0.3])])

    config = SimpleNamespace(output_dir=str(tmp_path), evaluation_batch_size=2, evaluation_temperature=0.6)
    with patch("agentcore_rl_toolkit.backends.tinker_api.train.rollout", fake_rollout):
        task = asyncio.create_task(evaluate(config, None, None, [{}, {}], step=10))
        try:
            await asyncio.wait_for(first_finished.wait(), timeout=2)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert [row["input_index"] for row in rows] == [1]
            assert not task.done()
        finally:
            release_slow.set()
            metrics = await task
    assert metrics["eval/reward_mean"] == 1
    assert metrics["eval/episodes"] == 2
    assert metrics["eval/scored_episodes"] == 1
    assert metrics["eval/coverage"] == 0.5
    assert metrics["eval/empty_episodes"] == metrics["eval/failed_episodes"] == 1
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[1]["reward"] is None
    assert rows[1]["input_index"] == 0
