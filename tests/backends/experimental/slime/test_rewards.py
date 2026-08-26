"""Unit tests for normalize_episode_rewards.

No slime dependency: samples are SimpleNamespace objects (the function uses
duck typing via Any).
"""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest

from agentcore_rl_toolkit.backends.experimental.slime.integration.rewards import normalize_episode_rewards


def _args(estimator="grpo", normalization=True, std=False):
    return Namespace(
        advantage_estimator=estimator,
        rewards_normalization=normalization,
        grpo_std_normalization=std,
    )


def _sample(index, group_index, reward, rollout_id=None):
    return SimpleNamespace(
        index=index,
        group_index=group_index,
        rollout_id=rollout_id if rollout_id is not None else index,
        get_reward_value=lambda args: reward,
    )


# ---------------------------------------------------------------------------
# Identity paths
# ---------------------------------------------------------------------------


def test_identity_when_normalization_disabled():
    samples = [_sample(0, 0, 1.0), _sample(1, 0, 0.0)]
    raw, norm = normalize_episode_rewards(_args(normalization=False), samples)
    assert raw == norm == [1.0, 0.0]


@pytest.mark.parametrize("estimator", ["reinforce", "ppo", "gae"])
def test_identity_for_non_group_estimators(estimator):
    samples = [_sample(0, 0, 1.0), _sample(1, 0, -1.0)]
    raw, norm = normalize_episode_rewards(_args(estimator=estimator), samples)
    assert raw == norm


# ---------------------------------------------------------------------------
# Basic GRPO normalization
# ---------------------------------------------------------------------------


def test_grpo_centering_around_group_mean():
    # Two samples in one group with rewards 1.0 and -1.0 → mean 0 → centered: 1.0, -1.0
    samples = [_sample(0, 0, 1.0), _sample(1, 0, -1.0)]
    _, norm = normalize_episode_rewards(_args("grpo"), samples)
    assert norm[0] == pytest.approx(1.0)
    assert norm[1] == pytest.approx(-1.0)


def test_grpo_uniform_group_yields_zero():
    samples = [_sample(i, 0, 0.5) for i in range(4)]
    _, norm = normalize_episode_rewards(_args("grpo"), samples)
    assert all(n == pytest.approx(0.0) for n in norm)


def test_grpo_two_independent_groups():
    # Group 0: rewards [2, 0] → centered [1, -1]
    # Group 1: rewards [10, 6] → centered [2, -2]
    g0 = [_sample(0, 0, 2.0), _sample(1, 0, 0.0)]
    g1 = [_sample(2, 1, 10.0), _sample(3, 1, 6.0)]
    _, norm = normalize_episode_rewards(_args("grpo"), g0 + g1)
    assert norm[0] == pytest.approx(1.0)
    assert norm[1] == pytest.approx(-1.0)
    assert norm[2] == pytest.approx(2.0)
    assert norm[3] == pytest.approx(-2.0)


@pytest.mark.parametrize("estimator", ["grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"])
def test_all_group_norm_estimators_normalize(estimator):
    samples = [_sample(0, 0, 3.0), _sample(1, 0, 1.0)]
    _, norm = normalize_episode_rewards(_args(estimator), samples)
    assert norm[0] == pytest.approx(1.0)
    assert norm[1] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Std normalization
# ---------------------------------------------------------------------------


def test_grpo_std_normalization():
    import statistics

    rewards = [3.0, 1.0]
    samples = [_sample(0, 0, rewards[0]), _sample(1, 0, rewards[1])]
    _, norm = normalize_episode_rewards(_args("grpo", std=True), samples)
    mean = statistics.fmean(rewards)
    std = statistics.stdev(rewards)
    expected = [(r - mean) / (std + 1e-6) for r in rewards]
    assert norm[0] == pytest.approx(expected[0])
    assert norm[1] == pytest.approx(expected[1])


def test_std_normalization_single_sample_skips_std():
    # Only one unique rollout in the group → stdev undefined → no std division
    samples = [_sample(0, 0, 5.0)]
    _, norm = normalize_episode_rewards(_args("grpo", std=True), samples)
    assert norm[0] == pytest.approx(0.0)


def test_std_not_applied_for_reinforce_plus_plus_baseline():
    # reinforce_plus_plus_baseline is in _GROUP_NORM_ESTIMATORS but NOT _STD_ESTIMATORS
    rewards = [3.0, 1.0]
    samples = [_sample(0, 0, rewards[0]), _sample(1, 0, rewards[1])]
    _, norm = normalize_episode_rewards(_args("reinforce_plus_plus_baseline", std=True), samples)
    # No std division — just centering
    assert norm[0] == pytest.approx(1.0)
    assert norm[1] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Fork dedup: multiple rows from one rollout
# ---------------------------------------------------------------------------


def test_forked_rollout_counts_once_in_group_mean():
    # rollout 0 forked into two rows (same rollout_id=0); rollout 1 has one row.
    # Group rewards used for baseline: {0: 1.0, 1: -1.0} → mean=0 → centered: 1.0, 1.0, -1.0
    fork_a = _sample(index=0, group_index=0, reward=1.0, rollout_id=0)
    fork_b = _sample(index=1, group_index=0, reward=1.0, rollout_id=0)
    other = _sample(index=2, group_index=0, reward=-1.0, rollout_id=2)
    _, norm = normalize_episode_rewards(_args("grpo"), [fork_a, fork_b, other])
    assert norm[0] == pytest.approx(1.0)
    assert norm[1] == pytest.approx(1.0)
    assert norm[2] == pytest.approx(-1.0)


def test_rollout_key_falls_back_to_index_when_rollout_id_none():
    # SimpleNamespace with rollout_id=None → key should be index
    s = SimpleNamespace(index=5, group_index=0, rollout_id=None, get_reward_value=lambda a: 0.0)
    _, norm = normalize_episode_rewards(_args("grpo"), [s])
    assert norm[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Raw rewards are always returned unchanged
# ---------------------------------------------------------------------------


def test_raw_rewards_unchanged_in_grpo():
    samples = [_sample(0, 0, 2.0), _sample(1, 0, -2.0)]
    raw, _ = normalize_episode_rewards(_args("grpo"), samples)
    assert raw == [2.0, -2.0]
