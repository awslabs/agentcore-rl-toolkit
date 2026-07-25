"""gateway_host tests: threaded serving, singleton semantics, and one
end-to-end OpenAI-wire turn captured through the verl sampling backend.

verl is not imported — the LLMServerClient is a duck-typed fake.
"""

import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from agentcore_rl_toolkit.backends.experimental.verl import gateway_host
from agentcore_rl_toolkit.backends.experimental.verl.gateway_host import get_or_start_gateway


@dataclass
class FakeTokenOutput:
    token_ids: list[int]
    log_probs: Optional[list[float]] = None
    stop_reason: Optional[str] = None
    extra_fields: dict[str, Any] = field(default_factory=dict)


class FakeLLMServerClient:
    """Echoes a fixed response for any prompt."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        self.calls.append({"request_id": request_id, "prompt_ids": list(prompt_ids)})
        return FakeTokenOutput(token_ids=[101, 102], log_probs=[-0.5, -0.6], stop_reason="completed")


class FakeTokenizer:
    """Minimal HF-tokenizer stand-in for HfTemplateRenderer + adapters.

    apply_chat_template maps each message to two ids; decode returns a fixed
    assistant string.
    """

    eos_token = "</s>"
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tools=None, tokenize=True, add_generation_prompt=True, **kw):
        ids = []
        for i, _ in enumerate(messages):
            ids += [10 + i, 20 + i]
        if add_generation_prompt:
            ids.append(99)
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "hello world"

    def convert_tokens_to_ids(self, token):
        return None


@pytest.fixture(autouse=True)
def reset_gateway():
    yield
    gateway_host._reset_for_tests()


def _start(client=None):
    return get_or_start_gateway(
        server_manager=client or FakeLLMServerClient(),
        tokenizer=FakeTokenizer(),
        host="127.0.0.1",
        public_host="127.0.0.1",
    )


def test_starts_and_serves_healthz():
    handle = _start()
    assert handle.base_url.startswith("http://127.0.0.1:")
    with urllib.request.urlopen(f"{handle.base_url}/healthz", timeout=5) as r:
        assert r.status == 200


def test_singleton_returns_same_handle():
    h1 = _start()
    h2 = _start()
    assert h1 is h2


def test_reset_allows_restart():
    h1 = _start()
    port1 = h1.base_url.rsplit(":", 1)[1]
    gateway_host._reset_for_tests()
    h2 = _start()
    assert h2 is not h1
    # auto-port: new server is live even if the port differs
    with urllib.request.urlopen(f"{h2.base_url}/healthz", timeout=5) as r:
        assert r.status == 200
    assert port1  # silence unused warning; ports may or may not collide


@pytest.mark.asyncio
async def test_end_to_end_turn_capture():
    """One OpenAI-wire chat turn with the sid in the Bearer slot, then
    finish_session yields a TraceRecord trained on the fake backend's tokens."""
    import aiohttp

    client = FakeLLMServerClient()
    handle = _start(client)
    sid = "test-session-0000-0000-000000000000000"

    handle.gateway.create_session(sid, sampling_defaults={"max_new_tokens": 32})

    async with aiohttp.ClientSession() as http:
        resp = await http.post(
            f"{handle.base_url}/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {sid}"},
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["choices"][0]["message"]["content"] == "hello world"

    # backend saw the sid as the sticky request_id
    assert client.calls and client.calls[0]["request_id"] == sid

    records = await handle.gateway.finish_session(sid, reward=1.0)
    assert len(records) == 1
    r = records[0]
    assert r.reward == 1.0
    # response region == the fake backend's two generated tokens, all trained
    assert r.token_ids[-2:] == [101, 102]
    assert r.loss_mask == [1, 1]
    assert r.logprobs == [-0.5, -0.6]
