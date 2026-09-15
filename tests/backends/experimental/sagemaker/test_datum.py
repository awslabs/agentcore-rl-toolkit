"""``trace_record_to_datum`` unit tests: the TraceRecord -> SageMaker datum shift,
the prompt-span left-padding, and loss-mask propagation into every masked field."""

from agentcore_rl_toolkit.backends.experimental.sagemaker.datum import trace_record_to_datum
from agentcore_rl_toolkit.rollout_gateway import TraceRecord


def _inputs(datum: dict, key: str) -> list:
    return datum["lossFnInputs"][key]["data"]


def test_single_turn_datum_shape():
    # 2 prompt tokens + 3 response tokens; the datum is shifted by one.
    record = TraceRecord(
        token_ids=[10, 11, 20, 21, 22],
        loss_mask=[1, 1, 1],
        logprobs=[-0.1, -0.2, -0.3],
    )

    datum = trace_record_to_datum(record, advantage=0.5)

    assert datum["modelInput"]["chunks"] == [{"tokens": [10, 11, 20, 21], "type": "encoded_text"}]
    assert _inputs(datum, "target_tokens") == [11, 20, 21, 22]
    # prompt_length - 1 == 1 leading pad, then the response span
    assert _inputs(datum, "weights") == [0.0, 1.0, 1.0, 1.0]
    assert _inputs(datum, "logprobs") == [0.0, -0.1, -0.2, -0.3]
    assert _inputs(datum, "advantages") == [0.0, 0.5, 0.5, 0.5]

    seq_len = len(datum["modelInput"]["chunks"][0]["tokens"])
    for key in ("target_tokens", "logprobs", "advantages", "weights"):
        assert datum["lossFnInputs"][key]["shape"] == [seq_len]
        assert len(_inputs(datum, key)) == seq_len


def test_masked_positions_get_zero_advantage():
    """Tool results / user turns land inside the response span as loss_mask=0 and
    carry a stored logprob of 0.0. They must not receive the group advantage: a
    surrogate that ignored ``weights`` would otherwise score them against a
    fabricated old policy (logprob 0.0 == p 1.0)."""
    record = TraceRecord(
        token_ids=[10, 20, 21, 30, 31, 40],
        # assistant turn, then a masked tool result, then another assistant turn
        loss_mask=[1, 1, 0, 0, 1],
        logprobs=[-0.1, -0.2, 0.0, 0.0, -0.5],
    )

    datum = trace_record_to_datum(record, advantage=2.0)

    assert _inputs(datum, "weights") == [1.0, 1.0, 0.0, 0.0, 1.0]
    assert _inputs(datum, "advantages") == [2.0, 2.0, 0.0, 0.0, 2.0]

    # every masked position is zeroed in both fields, whatever the mask pattern
    weights = _inputs(datum, "weights")
    advantages = _inputs(datum, "advantages")
    assert all(a == 0.0 for w, a in zip(weights, advantages, strict=True) if w == 0.0)


def test_negative_advantage_survives_masking():
    record = TraceRecord(token_ids=[10, 20, 30], loss_mask=[1, 0], logprobs=[-0.1, 0.0])

    datum = trace_record_to_datum(record, advantage=-1.5)

    assert _inputs(datum, "advantages") == [-1.5, 0.0]


def test_empty_or_fully_masked_records_are_dropped():
    empty = TraceRecord(token_ids=[10, 11], loss_mask=[], logprobs=[])
    assert trace_record_to_datum(empty, advantage=1.0) is None

    all_zero = TraceRecord(token_ids=[10, 20, 21], loss_mask=[0, 0], logprobs=[0.0, 0.0])
    assert trace_record_to_datum(all_zero, advantage=1.0) is None


def test_record_without_prompt_is_dropped():
    # loss_mask covers every token, so there is no prompt token to shift against.
    no_prompt = TraceRecord(token_ids=[20, 21], loss_mask=[1, 1], logprobs=[-0.1, -0.2])
    assert trace_record_to_datum(no_prompt, advantage=1.0) is None
