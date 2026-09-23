"""Check the sampling contract sent to the real Tinker SDK types."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agentcore_rl_toolkit.rollout_gateway.sampling_backends.tinker_sdk import TinkerSdkBackend


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("params", "expected_stop"),
    [({}, None), ({"stop": None}, None), ({"stop": []}, []), ({"stop": [151645]}, [151645])],
)
async def test_sampling_params_preserve_stop(params, expected_stop):
    client = SimpleNamespace(
        sample_async=AsyncMock(
            return_value=SimpleNamespace(sequences=[SimpleNamespace(tokens=[17], logprobs=[-0.2], stop_reason="stop")])
        )
    )
    await TinkerSdkBackend(client).generate(prompt_ids=[11, 12], sampling_params={"max_new_tokens": 2048, **params})
    wire = client.sample_async.call_args.kwargs["sampling_params"].model_dump(exclude_none=True)
    if expected_stop is None:
        assert "stop" not in wire
    else:
        assert wire["stop"] == expected_stop
    assert wire["max_tokens"] == 2048
