"""GRPO normalization that stays correct when one rollout emits multiple samples.

    --custom-reward-post-process-path \
        agentcore_rl_toolkit.backends.experimental.slime.integration.rewards.normalize_episode_rewards

slime's built-in groups positionally (reshape to ``(-1, n_samples_per_prompt)``), which
breaks when a forked trajectory contributes more than one row. This groups by
``group_index`` and dedups by ``rollout_id``, so a rollout that forked into N rows still
counts once in the group baseline.
"""

from __future__ import annotations

import statistics
from argparse import Namespace
from collections import defaultdict
from typing import Any

_GROUP_NORM_ESTIMATORS = ("grpo", "gspo", "cispo", "reinforce_plus_plus_baseline")
_STD_ESTIMATORS = ("grpo", "gspo", "cispo")


def normalize_episode_rewards(args: Namespace, samples: list[Any]) -> tuple[list[float], list[float]]:
    """Return ``(raw_rewards, normalized_rewards)`` in ``samples`` order."""
    raw_rewards = [s.get_reward_value(args) for s in samples]
    if args.advantage_estimator not in _GROUP_NORM_ESTIMATORS or not args.rewards_normalization:
        return raw_rewards, raw_rewards

    use_std = args.advantage_estimator in _STD_ESTIMATORS and args.grpo_std_normalization

    # group_index -> {rollout_id: reward}; inner dict dedups forked rows whose reward is identical.
    groups: dict[Any, dict[Any, float]] = defaultdict(dict)
    for sample, reward in zip(samples, raw_rewards, strict=True):
        groups[sample.group_index][_rollout_key(sample)] = reward

    normalized: dict[Any, float] = {}
    for by_rollout in groups.values():
        rewards = list(by_rollout.values())
        mean = statistics.fmean(rewards)
        std = statistics.stdev(rewards) if use_std and len(rewards) > 1 else None
        for key, reward in by_rollout.items():
            centered = reward - mean
            normalized[key] = centered / (std + 1e-6) if std is not None else centered

    return raw_rewards, [normalized[_rollout_key(s)] for s in samples]


def _rollout_key(sample: Any) -> Any:
    return sample.rollout_id if sample.rollout_id is not None else sample.index
