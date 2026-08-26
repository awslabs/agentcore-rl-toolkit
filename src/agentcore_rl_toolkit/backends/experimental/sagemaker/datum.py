"""SageMaker Training Sessions datum construction from gateway TraceRecords."""

from agentcore_rl_toolkit.rollout_gateway import TraceRecord


def trace_record_to_datum(record: TraceRecord, advantage: float) -> dict | None:
    """Convert a gateway TraceRecord + advantage into a SageMaker training datum."""
    response_length = len(record.loss_mask)
    if response_length == 0 or not any(record.loss_mask):
        return None

    token_ids = list(record.token_ids)
    prompt_length = len(token_ids) - response_length
    if prompt_length < 1:
        return None

    input_tokens = token_ids[:-1]
    target_tokens = token_ids[1:]
    seq_len = len(input_tokens)

    weights = [0.0] * (prompt_length - 1) + [float(m) for m in record.loss_mask]
    logprobs = [0.0] * (prompt_length - 1) + list(record.logprobs)
    advantages = [0.0] * (prompt_length - 1) + [float(advantage)] * response_length

    return {
        "modelInput": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
        "lossFnInputs": {
            "target_tokens": {"data": target_tokens, "dtype": "int64", "shape": [seq_len]},
            "logprobs": {"data": logprobs, "dtype": "float32", "shape": [seq_len]},
            "advantages": {"data": advantages, "dtype": "float32", "shape": [seq_len]},
            "weights": {"data": weights, "dtype": "float32", "shape": [seq_len]},
        },
    }
