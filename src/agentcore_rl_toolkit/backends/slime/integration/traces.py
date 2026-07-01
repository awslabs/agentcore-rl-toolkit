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


def _is_prefix_extension(prev_full: list[int], next_prompt: list[int]) -> bool:
    """True if ``next_prompt`` begins byte-for-byte with ``prev_full``.

    ``prev_full`` = previous turn's ``prompt_token_ids + completion_token_ids``.
    The gateway's cumulative token mode guarantees this for adjacent turns of a
    session (see rllm_model_gateway.token_accumulator). When it doesn't hold — the
    gateway re-rendered the message list (non-cumulative mode), the agent did
    context manipulation (truncation, summary injection), or the accumulator reset
    — the two turns are NOT mergeable into one contiguous sequence.
    """
    if len(next_prompt) < len(prev_full):
        return False
    # Compare only the prefix region; next_prompt may be longer (bridge tokens).
    return next_prompt[: len(prev_full)] == prev_full


def _segment_to_sample(
    segment: list,
    reward: float,
    group_index: int,
    sample_index: int,
    session_id: str,
    segment_index: int,
):
    """Merge one contiguous (prefix-extending) run of traces into a single Sample.

    Layout of the merged token stream, given traces ``t_0 .. t_{k-1}`` where each
    ``t_i.prompt`` starts with ``t_{i-1}.prompt + t_{i-1}.completion``::

        [ t_0.prompt | t_0.completion | bridge_1 | t_1.completion | bridge_2 | ... ]
          \\___prompt__/ \\____________________ response ____________________________/

    ``bridge_i`` = the new tokens turn ``i`` added before its own generation
    (end-of-prev-assistant + new user/tool messages + next gen header). These are
    NOT sampled by the policy, so they get ``loss_mask = 0`` and ``logprob = 0``;
    only the completion spans are trainable (``loss_mask = 1``). This yields ONE
    training sequence per trajectory of length ``≈ final_prompt + Σ completions``
    instead of re-encoding the growing prefix once per turn.
    """
    Sample = _import_sample()

    t0 = segment[0]
    prompt_ids = list(t0.prompt_token_ids or [])
    if not prompt_ids:
        prompt_ids = [0]

    # response stream = everything after the initial prompt, with per-token mask.
    response_ids: list[int] = []
    loss_mask: list[int] = []
    logprobs: list[float] = []

    prev_full = prompt_ids + list(t0.completion_token_ids or [])
    for i, trace in enumerate(segment):
        comp = list(trace.completion_token_ids or [])
        lp = list(trace.logprobs or [])
        if i > 0:
            # Bridge = new prompt tokens beyond the previous turn's full sequence.
            cur_prompt = list(trace.prompt_token_ids or [])
            bridge = cur_prompt[len(prev_full) :]
            response_ids.extend(bridge)
            loss_mask.extend([0] * len(bridge))
            logprobs.extend([0.0] * len(bridge))
            prev_full = cur_prompt + comp
        # Completion tokens are the trainable, policy-sampled span.
        if not comp:
            comp = [0]
            lp = [0.0]
        # Align logprobs to completion length.
        if len(lp) > len(comp):
            lp = lp[: len(comp)]
        elif len(lp) < len(comp):
            lp = lp + [0.0] * (len(comp) - len(lp))
        response_ids.extend(comp)
        loss_mask.extend([1] * len(comp))
        logprobs.extend(lp)

    if not response_ids:
        response_ids = [0]
        loss_mask = [0]
        logprobs = [0.0]

    sample = Sample()
    sample.tokens = prompt_ids + response_ids
    sample.response_length = len(response_ids)
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = logprobs
    sample.reward = reward
    sample.group_index = group_index
    sample.index = sample_index
    sample.session_id = session_id
    sample.metadata = {
        "task_index": group_index,
        "gateway_session_id": session_id,
        "segment_index": segment_index,
        "num_turns": len(segment),
    }

    # TRUNCATED if the final turn hit the length cap, else COMPLETED.
    finish = getattr(segment[-1], "finish_reason", "stop")
    sample.status = Sample.Status.TRUNCATED if finish == "length" else Sample.Status.COMPLETED
    return sample


def merge_traces_to_samples(
    traces: list,
    episode_reward: float,
    group_index: int,
    session_id: str,
    sample_counter,
) -> list:
    """Merge an episode's traces into per-trajectory Samples by prefix extension.

    Adjacent turns are merged into one Sample when they satisfy the
    prefix-extension invariant (each turn's prompt begins byte-for-byte with the
    previous turn's prompt+completion); see :func:`_segment_to_sample`.

    Whenever the invariant breaks between two turns (non-cumulative re-render,
    agent context manipulation, or a gateway accumulator reset), the trajectory is
    split at that point and each contiguous run becomes its own Sample. All Samples
    share the same ``session_id`` so :func:`rewards.normalize_episode_rewards`
    still treats the whole session as one episode in the GRPO group.

    Returns:
        List of slime Samples (one per contiguous prefix-extending segment).
    """
    if not traces:
        return []

    # Split traces into contiguous prefix-extending segments.
    segments: list[list] = [[traces[0]]]
    prev_full = list(traces[0].prompt_token_ids or []) + list(traces[0].completion_token_ids or [])
    for trace in traces[1:]:
        cur_prompt = list(trace.prompt_token_ids or [])
        if _is_prefix_extension(prev_full, cur_prompt):
            segments[-1].append(trace)
        else:
            logger.warning(
                "Session %s: prefix-extension broke at turn (prev_full=%d, cur_prompt=%d); "
                "splitting trajectory into a new segment. Merged token savings reduced.",
                session_id[:8],
                len(prev_full),
                len(cur_prompt),
            )
            segments.append([trace])
        prev_full = cur_prompt + list(trace.completion_token_ids or [])

    samples = []
    for segment_index, segment in enumerate(segments):
        samples.append(
            _segment_to_sample(
                segment=segment,
                reward=episode_reward,
                group_index=group_index,
                sample_index=next(sample_counter),
                session_id=session_id,
                segment_index=segment_index,
            )
        )
    return samples


def extract_reward(acr_result: dict) -> float:
    """Extract scalar reward from an ACR S3 result dict."""
    rewards = acr_result.get("rewards", 0.0)
    if isinstance(rewards, list):
        return rewards[-1] if rewards else 0.0
    return float(rewards)
