"""Custom reward post-processing for multi-turn ACR episodes.

Replaces slime's default reshape-based normalization with episode-level
normalization that handles variable turn counts.

Provides pluggable normalization strategies. Default is GRPO (group-relative).

Usage:
    --custom-reward-post-process-path \
        agentcore_rl_toolkit.backends.slime.integration.rewards.normalize_episode_rewards
"""

from collections import defaultdict


def _grpo_normalize(episode_rewards: list[float], std_normalize: bool) -> list[float]:
    """GRPO: mean-center, optionally std-normalize across episodes in a task group."""
    n = len(episode_rewards)
    mean = sum(episode_rewards) / n
    centered = [r - mean for r in episode_rewards]

    if std_normalize:
        variance = sum(c * c for c in centered) / n
        std = variance**0.5
        if std > 1e-6:
            centered = [c / (std + 1e-6) for c in centered]
        else:
            centered = [0.0] * n

    return centered


def _identity_normalize(episode_rewards: list[float], std_normalize: bool) -> list[float]:
    """No normalization — pass raw rewards through."""
    return list(episode_rewards)


NORMALIZATION_STRATEGIES = {
    "grpo": _grpo_normalize,
    "identity": _identity_normalize,
}


def normalize_episode_rewards(args, samples):
    """Normalize rewards at the episode level within each task group.

    Two-step process:
    1. Aggregate: group turns by (group_index, session_id) to get one reward
       per episode. All turns in an episode share the same broadcast reward,
       so we take the first turn's reward as the episode reward.
    2. Normalize: within each task group (group_index), apply the chosen
       normalization strategy to the per-episode rewards.
    3. Assign: write the normalized episode reward back to all turns.

    This ensures episodes with different turn counts are weighted equally —
    a 3-turn success and a 2-turn failure each count as one data point.

    Normalization strategy is selected via reward_postprocessing
    (in config.yaml), default "grpo". Options: "grpo", "identity".

    Samples with group_index=-1 (dummy padding) are skipped.

    Args:
        args: Slime argument namespace.
        samples: Flat list of slime Samples.

    Returns:
        Tuple of (raw_rewards, normalized_rewards) as lists of floats.
    """
    strategy_name = getattr(args, "reward_postprocessing", "grpo")
    normalize_fn = NORMALIZATION_STRATEGIES.get(strategy_name, _grpo_normalize)
    std_normalize = getattr(args, "grpo_std_normalization", False)

    raw_rewards = [s.get_reward_value(args) for s in samples]

    # Step 1: Group samples by (group_index, session_id) to identify episodes
    episodes = defaultdict(list)
    for i, s in enumerate(samples):
        if s.group_index == -1:
            continue
        session_id = s.metadata.get("gateway_session_id", "") if s.metadata else ""
        episodes[(s.group_index, session_id)].append(i)

    # Step 2: Group episodes by task (group_index)
    task_groups = defaultdict(list)
    for (grp_idx, _session_id), sample_indices in episodes.items():
        episode_reward = raw_rewards[sample_indices[0]]
        task_groups[grp_idx].append((sample_indices, episode_reward))

    # Step 3: Normalize per-episode rewards within each task group
    rewards = list(raw_rewards)

    for episode_list in task_groups.values():
        episode_rewards = [r for _, r in episode_list]
        normalized = normalize_fn(episode_rewards, std_normalize)

        # Step 4: Assign normalized episode reward back to all turns
        for (sample_indices, _), norm_reward in zip(episode_list, normalized, strict=True):
            for idx in sample_indices:
                rewards[idx] = norm_reward

    return raw_rewards, rewards
