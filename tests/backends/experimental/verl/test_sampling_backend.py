"""VerlSamplingBackend unit tests.

verl is not imported: LLMServerClient is duck-typed (only ``generate`` is used)
and TokenOutput is mimicked by a plain dataclass with the same fields.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from agentcore_rl_toolkit.backends.experimental.verl.sampling_backend import (
    VerlSamplingBackend,
    _verl_sampling_params,
)


@dataclass
class FakeTokenOutput:
    """Shape-compatible stand-in for verl's TokenOutput pydantic model."""

    token_ids: list[int]
    log_probs: Optional[list[float]] = None
    stop_reason: Optional[str] = None
    extra_fields: dict[str, Any] = field(default_factory=dict)


class FakeLLMServerClient:
    def __init__(self, outputs: list[FakeTokenOutput]):
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        self.calls.append(
            {"request_id": request_id, "prompt_ids": prompt_ids, "sampling_params": sampling_params, **kwargs}
        )
        return self.outputs.pop(0)


def test_sampling_params_whitelist():
    mapped = _verl_sampling_params(
        {
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "stop": ["</s>"],
            "stop_token_ids": [2],
            # SGLang-only keys that would blow up vLLM's SamplingParams(**...):
            "no_stop_trim": True,
            "spaces_between_special_tokens": False,
            "skip_special_tokens": False,
        }
    )
    assert mapped == {
        "max_new_tokens": 128,
        "logprobs": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "stop": ["</s>"],
        "stop_token_ids": [2],
    }


def test_sampling_params_top_k_filtering():
    assert "top_k" not in _verl_sampling_params({"top_k": 0})
    assert "top_k" not in _verl_sampling_params({"top_k": None})
    assert "top_k" not in _verl_sampling_params({"top_k": 1.5})
    assert _verl_sampling_params({"top_k": -1})["top_k"] == -1


def test_sampling_params_defaults():
    mapped = _verl_sampling_params({})
    assert mapped == {"max_new_tokens": 4096, "logprobs": True}


@pytest.mark.asyncio
async def test_generate_maps_token_output():
    client = FakeLLMServerClient(
        [FakeTokenOutput(token_ids=[5, 6, 7], log_probs=[-0.1, -0.2, -0.3], stop_reason="completed")]
    )
    backend = VerlSamplingBackend(client)

    turn = await backend.generate(prompt_ids=[1, 2, 3], sampling_params={"max_new_tokens": 64}, session_id="sid-1")

    assert turn.prompt_ids == [1, 2, 3]
    assert turn.output_ids == [5, 6, 7]
    assert turn.output_log_probs == [-0.1, -0.2, -0.3]
    assert turn.finish_reason == "stop"
    # sid rides as the sticky request_id
    assert client.calls[0]["request_id"] == "sid-1"
    assert client.calls[0]["sampling_params"]["logprobs"] is True


@pytest.mark.asyncio
async def test_generate_missing_logprobs_padded():
    client = FakeLLMServerClient([FakeTokenOutput(token_ids=[5, 6], log_probs=None, stop_reason="completed")])
    backend = VerlSamplingBackend(client)
    turn = await backend.generate(prompt_ids=[1], sampling_params={}, session_id="s")
    assert turn.output_log_probs == [0.0, 0.0]


@pytest.mark.asyncio
async def test_generate_length_finish_reason():
    # explicit 'length' from the server (SGLang passes raw finish type through)
    client = FakeLLMServerClient([FakeTokenOutput(token_ids=[5], stop_reason="length")])
    turn = await VerlSamplingBackend(client).generate(
        prompt_ids=[1], sampling_params={"max_new_tokens": 64}, session_id="s"
    )
    assert turn.finish_reason == "length"

    # vLLM collapses 'length' into 'completed'; infer from the token budget
    client = FakeLLMServerClient([FakeTokenOutput(token_ids=[5, 6, 7, 8], stop_reason="completed")])
    turn = await VerlSamplingBackend(client).generate(
        prompt_ids=[1], sampling_params={"max_new_tokens": 4}, session_id="s"
    )
    assert turn.finish_reason == "length"


@pytest.mark.asyncio
async def test_generate_abort_raises():
    client = FakeLLMServerClient([FakeTokenOutput(token_ids=[], stop_reason="aborted")])
    backend = VerlSamplingBackend(client)
    with pytest.raises(RuntimeError, match="aborted"):
        await backend.generate(prompt_ids=[1], sampling_params={}, session_id="s")


@pytest.mark.asyncio
async def test_extra_fields_folding():
    client = FakeLLMServerClient(
        [
            FakeTokenOutput(
                token_ids=[5],
                stop_reason="completed",
                extra_fields={"global_steps": 10, "min_global_steps": 10, "max_global_steps": 10},
            ),
            FakeTokenOutput(
                token_ids=[6],
                stop_reason="completed",
                extra_fields={"global_steps": 12, "min_global_steps": 9, "max_global_steps": 12},
            ),
        ]
    )
    backend = VerlSamplingBackend(client)
    await backend.generate(prompt_ids=[1], sampling_params={"max_new_tokens": 64}, session_id="s")
    await backend.generate(prompt_ids=[1, 5], sampling_params={"max_new_tokens": 64}, session_id="s")

    folded = backend.pop_extra_fields("s")
    assert folded == {"global_steps": 12, "min_global_steps": 9, "max_global_steps": 12}
    # pop clears
    assert backend.pop_extra_fields("s") == {}
    # unknown sid is empty, not an error
    assert backend.pop_extra_fields("nope") == {}
