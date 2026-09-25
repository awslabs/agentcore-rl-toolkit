"""The next-token and episode boundaries must survive tools and trajectory forks."""

import math

import pytest

from agentcore_rl_toolkit.backends.tinker_api.data import Episode, trace_to_datum, training_data
from agentcore_rl_toolkit.rollout_gateway import TraceRecord


def record(sid, tokens=None, mask=None, logprobs=None):
    return TraceRecord(
        token_ids=tokens or [10, 11, 12, 13],
        loss_mask=mask if mask is not None else [1, 1],
        logprobs=logprobs if logprobs is not None else [-0.3, -0.4],
        rollout_id=sid,
    )


def test_token_shift_masks_prompt_and_tool_output():
    trace = record("a", [10, 11, 12, 13, 14, 15, 16], [1, 1, 0, 0, 1], [-0.1, -0.2, math.nan, 9, -0.3])
    datum = trace_to_datum(trace, 0.75)
    assert datum.model_input.to_ints() == [10, 11, 12, 13, 14, 15]
    assert list(datum.loss_fn_inputs["target_tokens"].data) == [11, 12, 13, 14, 15, 16]
    assert list(datum.loss_fn_inputs["advantages"].data) == [0, 0.75, 0.75, 0, 0, 0.75]
    assert list(datum.loss_fn_inputs["logprobs"].data) == pytest.approx([0, -0.1, -0.2, 0, 0, -0.3])


def test_forked_failure_is_counted_once_and_preserves_centering_and_masks():
    # Episode A's second row reuses an earlier generated token as masked context.
    a = Episode("a", 0, [record("a"), record("a", mask=[0, 1])], error="token limit")
    for trace in a.records:
        trace.metadata["truncated"] = True
    b = Episode("b", 4, [record("b")])
    datums, metrics = training_data([[a, b]])
    assert metrics["reward/mean"] == 2  # Two episodes, not three rows.
    assert metrics["rollout/episodes"] == 2
    assert metrics["train/generated_tokens"] == 5
    assert metrics["rollout/failed_episodes"] == metrics["rollout/truncated_episodes"] == 1
    assert metrics["rollout/failure_rate"] == 0.5
    assert [list(d.loss_fn_inputs["advantages"].data) for d in datums] == [[0, -2, -2], [0, 0, -2], [0, 2, 2]]


def test_constant_rewards_skip_training():
    datums, metrics = training_data([[Episode("a", 1, [record("a")]), Episode("b", 1, [record("b")])]])
    assert datums == []
    assert metrics["train/active_groups"] == 0
    assert metrics["reward/mean"] == 1


def test_empty_failure_is_excluded_from_group_mean_and_advantages():
    group = [Episode(sid, reward, [record(sid)]) for sid, reward in zip("abc", [1, 0, 1], strict=True)]
    group.append(Episode("missing", None, [], error="InvokeAgentRuntime 502"))
    datums, metrics = training_data([group])
    assert metrics["reward/mean"] == pytest.approx(2 / 3)
    assert metrics["rollout/episodes"] == 4
    assert metrics["rollout/scored_episodes"] == 3
    assert metrics["rollout/empty_episodes"] == metrics["rollout/failed_episodes"] == 1
    assert metrics["rollout/failure_rate"] == 0.25
    assert len(datums) == 3
    for datum, advantage in zip(datums, [1 / 3, -2 / 3, 1 / 3], strict=True):
        assert list(datum.loss_fn_inputs["advantages"].data) == pytest.approx([0, advantage, advantage])


@pytest.mark.parametrize("survivors", [0, 1])
def test_insufficient_group_does_not_remove_other_groups(survivors):
    partial = [Episode("a", 1, [record("a")])] if survivors else []
    partial.extend(Episode(f"missing-{i}", None, [], error="502") for i in range(4 - survivors))
    good = [Episode("b", 1, [record("b")]), Episode("c", 0, [record("c")])]
    datums, metrics = training_data([partial, good])
    assert len(datums) == 2
    assert metrics["train/insufficient_groups"] == 1
    assert metrics["train/active_groups"] == 1
    assert metrics["rollout/scored_episodes"] == survivors + 2
    assert list(datums[0].loss_fn_inputs["advantages"].data) == [0, 0.5, 0.5]


def test_all_empty_failures_have_no_reward_mean_or_training_data():
    datums, metrics = training_data([[Episode("a", None, [], error="502"), Episode("b", None, [], error="502")]])
    assert datums == []
    assert metrics["reward/mean"] is None
    assert metrics["rollout/scored_episodes"] == 0
    assert metrics["rollout/empty_episodes"] == 2
    assert metrics["rollout/failure_rate"] == 1


@pytest.mark.parametrize("episode", [Episode("a", 0, []), Episode("a", None, [record("a")], error="502")])
def test_empty_failure_cannot_be_scored_or_hide_generated_tokens(episode):
    with pytest.raises(ValueError, match="trainable tokens"):
        training_data([[episode]])


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_generated_logprob_is_rejected(value):
    with pytest.raises(ValueError, match="logprobs"):
        trace_to_datum(record("a", logprobs=[value, -0.1]), 1)


def test_mismatched_episode_identity_is_rejected():
    with pytest.raises(ValueError, match="rollout_id"):
        training_data([[Episode("a", 1, [record("b")]), Episode("b", 0, [record("b")])]])
