"""Convert rllm-model-gateway TraceRecords into slime Samples.

Each TraceRecord (one per LLM call) becomes a separate Sample.
All Samples from the same episode share group_index and gateway_session_id.
All Samples from the same task share group_index (= task_index).
"""

import logging

logger = logging.getLogger(__name__)


def _import_sample():
    try:
        from slime.utils.types import Sample

        return Sample
    except ImportError as err:
        raise ImportError(
            "slime is required for this module. Install with: pip install agentcore-rl-toolkit[slime]"
        ) from err


def trace_to_sample(
    trace,
    reward: float,
    group_index: int,
    sample_index: int,
    session_id: str,
    turn_index: int,
):
    """Convert a single gateway TraceRecord into a slime Sample.

    Args:
        trace: TraceRecord from rllm-model-gateway.
        reward: Reward for this turn (broadcast episode reward).
        group_index: Slime GRPO group (int). All turns from all episodes
            of the same task share this value.
        sample_index: Global sequential sample index.
        session_id: Gateway session ID for traceability.
        turn_index: Turn position within the episode (0-based).

    Returns:
        A slime Sample populated with token IDs, logprobs, reward, and metadata.
    """
    Sample = _import_sample()

    prompt_ids = trace.prompt_token_ids or []
    completion_ids = trace.completion_token_ids or []
    logprobs = trace.logprobs or []

    # Ensure at least 1 prompt token (Megatron requires prompt_length >= 1)
    if not prompt_ids:
        prompt_ids = [0]
    if not completion_ids:
        completion_ids = [0]
        logprobs = [0.0]

    # Pad/truncate logprobs to match completion length
    if len(logprobs) > len(completion_ids):
        logprobs = logprobs[: len(completion_ids)]
    elif len(logprobs) < len(completion_ids):
        logprobs = logprobs + [0.0] * (len(completion_ids) - len(logprobs))

    sample = Sample()
    sample.tokens = prompt_ids + completion_ids
    sample.response_length = len(completion_ids)
    sample.loss_mask = [1] * len(completion_ids)
    sample.rollout_log_probs = logprobs
    sample.reward = reward
    sample.group_index = group_index
    sample.index = sample_index
    sample.session_id = session_id
    sample.metadata = {
        "task_index": group_index,
        "gateway_session_id": session_id,
        "turn_index": turn_index,
    }

    finish = getattr(trace, "finish_reason", "stop")
    sample.status = Sample.Status.TRUNCATED if finish == "length" else Sample.Status.COMPLETED

    return sample


def episode_traces_to_samples(
    traces: list,
    episode_reward: float,
    group_index: int,
    session_id: str,
    sample_counter,
) -> list:
    """Convert all traces from one agent episode into correlated Samples.

    All returned Samples share the same group_index so GRPO groups them.
    Every turn receives the same episode reward (broadcast).

    Args:
        traces: TraceRecords from gateway, ordered chronologically.
        episode_reward: Scalar reward from agent's S3 result.
        group_index: Shared by all turns in this episode (= task_index).
        session_id: Gateway session ID for traceability.
        sample_counter: Iterator yielding global sample indices.

    Returns:
        List of slime Samples, one per trace.
    """
    samples = []
    for turn_index, trace in enumerate(traces):
        s = trace_to_sample(
            trace=trace,
            reward=episode_reward,
            group_index=group_index,
            sample_index=next(sample_counter),
            session_id=session_id,
            turn_index=turn_index,
        )
        samples.append(s)
    return samples


def extract_reward(acr_result: dict) -> float:
    """Extract scalar reward from an ACR S3 result dict."""
    rewards = acr_result.get("rewards", 0.0)
    if isinstance(rewards, list):
        return rewards[-1] if rewards else 0.0
    return float(rewards)
