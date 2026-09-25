"""Convert captured token trajectories to Tinker importance-sampling data."""

import math
from dataclasses import dataclass

import tinker

from agentcore_rl_toolkit.rollout_gateway.trace import TraceRecord


@dataclass
class Episode:
    """One invocation; reward=None records a failure without trainable tokens."""

    rollout_id: str
    reward: float | None
    records: list[TraceRecord]
    error: str | None = None


def trace_to_datum(record: TraceRecord, advantage: float) -> tinker.Datum:
    """Shift tokens once, preserving generated-token masks across tool turns.

    Trace masks and logprobs cover the response suffix; Tinker tensors cover all
    next-token targets. Masked targets carry zero advantage and zero logprob.
    """
    prompt_length = len(record.token_ids) - len(record.loss_mask)
    if prompt_length < 1 or not any(record.loss_mask):
        raise ValueError("A training trace needs a prompt and at least one generated token")
    if any(mask not in (0, 1) for mask in record.loss_mask):
        raise ValueError("loss_mask must contain only 0 or 1")
    if not math.isfinite(advantage):
        raise ValueError("advantage must be finite")
    if any(mask and not math.isfinite(lp) for mask, lp in zip(record.loss_mask, record.logprobs, strict=True)):
        raise ValueError("Generated-token logprobs must be finite")

    prefix_length = prompt_length - 1
    logprobs = [0.0] * prefix_length + [
        lp if mask else 0.0 for mask, lp in zip(record.loss_mask, record.logprobs, strict=True)
    ]
    advantages = [0.0] * prefix_length + [advantage * mask for mask in record.loss_mask]

    def tensor(data: list, dtype: str) -> tinker.TensorData:
        return tinker.TensorData(data=data, dtype=dtype, shape=[len(data)])

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(record.token_ids[:-1]),
        loss_fn_inputs={
            "target_tokens": tensor(record.token_ids[1:], "int64"),
            "logprobs": tensor(logprobs, "float32"),
            "advantages": tensor(advantages, "float32"),
        },
    )


def training_data(groups: list[list[Episode]]) -> tuple[list[tinker.Datum], dict[str, float | None]]:
    """Center rewards over episodes, then sum losses over their emitted records.

    A fork changes the number of datums, not the number of rewards in its group.
    Shared generated prefixes are already masked by the trajectory manager.
    """
    datums = []
    rewards = []
    active_groups = 0
    generated_tokens = 0
    insufficient_groups = 0
    for group in groups:
        if len({episode.rollout_id for episode in group}) != len(group):
            raise ValueError("Duplicate rollout_id in a group")
        for episode in group:
            has_tokens = any(any(record.loss_mask) for record in episode.records)
            if episode.reward is None:
                if has_tokens or episode.error is None:
                    raise ValueError("An unscored episode must be a failure without trainable tokens")
            elif not has_tokens:
                raise ValueError("An episode without trainable tokens cannot have a reward")
        scored = [episode for episode in group if episode.reward is not None]
        if any(not math.isfinite(episode.reward) for episode in scored):
            raise ValueError("Episode rewards must be finite")
        rewards.extend(episode.reward for episode in scored)
        if len(scored) < 2:
            insufficient_groups += 1
            continue
        mean_reward = sum(episode.reward for episode in scored) / len(scored)
        if all(episode.reward == mean_reward for episode in scored):
            continue
        active_groups += 1
        for episode in scored:
            for record in episode.records:
                if record.rollout_id != episode.rollout_id:
                    raise ValueError("Trace rollout_id does not match its episode")
                if any(record.loss_mask):
                    datums.append(trace_to_datum(record, episode.reward - mean_reward))
                    generated_tokens += sum(record.loss_mask)
    attempts = sum(len(group) for group in groups)
    return datums, {
        "reward/mean": sum(rewards) / len(rewards) if rewards else None,
        "rollout/episodes": attempts,
        "rollout/scored_episodes": len(rewards),
        "rollout/empty_episodes": attempts - len(rewards),
        "rollout/truncated_episodes": sum(
            any(record.metadata.get("truncated", False) for record in episode.records)
            for group in groups
            for episode in group
        ),
        "rollout/failed_episodes": sum(episode.error is not None for group in groups for episode in group),
        "rollout/failure_rate": (
            sum(episode.error is not None for group in groups for episode in group) / attempts if attempts else 0.0
        ),
        "train/active_groups": active_groups,
        "train/insufficient_groups": insufficient_groups,
        "train/datums": len(datums),
        "train/generated_tokens": generated_tokens,
    }
