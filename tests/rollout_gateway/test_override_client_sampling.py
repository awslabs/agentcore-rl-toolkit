"""Tests for the opt-in ``override_client_sampling`` gateway flag (see ``RolloutGateway``)."""

import pytest
from aiohttp.test_utils import TestClient, TestServer

from agentcore_rl_toolkit.rollout_gateway import RolloutGateway
from agentcore_rl_toolkit.rollout_gateway.adapters import AnthropicAdapter, OpenAIAdapter
from agentcore_rl_toolkit.rollout_gateway.render import ParsedOutput
from agentcore_rl_toolkit.rollout_gateway.trajectory import TurnRecord


class FakeRenderer:
    """One id per whitespace token."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _encode(self, text: str) -> list[int]:
        return [self.vocab.setdefault(t, len(self.vocab) + 1) for t in text.split()]

    async def render(self, messages, *, tools=None, add_generation_prompt=True):
        ids: list[int] = []
        for m in messages:
            content = m.get("content") or ""
            if isinstance(content, list):
                content = " ".join(block.get("text", "") for block in content)
            ids += self._encode(f"{m['role']}: {content}")
        return ids + (self._encode("assistant:") if add_generation_prompt else [])

    def get_stop_sequences(self):
        return []

    def parse(self, output_ids, *, finish_reason, tools_schema=None):
        inv = {v: k for k, v in self.vocab.items()}
        text = " ".join(inv.get(i, "?") for i in output_ids)
        return ParsedOutput(reasoning="", text=text, tool_uses=[], ill_formed=False)


class FakeBackend:
    """Replies "ok" and records the sampling params of every turn."""

    def __init__(self, renderer: FakeRenderer):
        self.renderer = renderer
        self.sampling: list[dict] = []

    async def generate(self, *, prompt_ids, sampling_params, session_id=None, image_data=None, video_data=None):
        self.sampling.append(dict(sampling_params))
        out_ids = self.renderer._encode("ok")
        return TurnRecord(
            prompt_ids=list(prompt_ids),
            output_ids=out_ids,
            finish_reason="stop",
            output_log_probs=[-0.5] * len(out_ids),
        )


TRAINER = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": 512}
CLIENT = {"temperature": 0.2, "top_p": 0.9, "top_k": 40}


async def _post(app, path, body, headers):
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.post(path, json=body, headers=headers)
        assert resp.status == 200, await resp.text()
    finally:
        await client.close()


def _openai(**extra):
    return {"model": "x", "messages": [{"role": "user", "content": "hi"}], **extra}


def _anthropic(**extra):
    return {"model": "x", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}], **extra}


@pytest.mark.asyncio
async def test_default_client_sampling_wins():
    renderer = FakeRenderer()
    backend = FakeBackend(renderer)
    adapter = OpenAIAdapter(backend=backend, renderer=renderer, tokenizer=None)
    assert adapter.override_client_sampling is False
    adapter.open_session("s", sampling_defaults=TRAINER)
    await _post(adapter.app, "/v1/chat/completions", _openai(**CLIENT), {"Authorization": "Bearer s"})
    assert {k: backend.sampling[0][k] for k in CLIENT} == CLIENT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cls", "path", "body", "headers"),
    [
        (OpenAIAdapter, "/v1/chat/completions", _openai(**CLIENT, max_tokens=64), {"Authorization": "Bearer s"}),
        (AnthropicAdapter, "/v1/messages", _anthropic(**CLIENT), {"x-api-key": "s"}),
    ],
)
async def test_flag_on_trainer_sampling_wins(cls, path, body, headers):
    renderer = FakeRenderer()
    backend = FakeBackend(renderer)
    adapter = cls(backend=backend, renderer=renderer, tokenizer=None, override_client_sampling=True)
    adapter.open_session("s", sampling_defaults=TRAINER)
    await _post(adapter.app, path, body, headers)
    sp = backend.sampling[0]
    assert (sp["temperature"], sp["top_p"], sp["top_k"]) == (1.0, 1.0, -1)
    assert sp["max_new_tokens"] == 64  # still min(trainer 512, client 64)


@pytest.mark.asyncio
async def test_flag_on_client_cannot_add_keys_the_trainer_left_unset():
    renderer = FakeRenderer()
    backend = FakeBackend(renderer)
    adapter = OpenAIAdapter(backend=backend, renderer=renderer, tokenizer=None, override_client_sampling=True)
    adapter.open_session("s", sampling_defaults={"temperature": 1.0})
    await _post(adapter.app, "/v1/chat/completions", _openai(top_p=0.5, top_k=5), {"Authorization": "Bearer s"})
    assert "top_p" not in backend.sampling[0] and "top_k" not in backend.sampling[0]


def test_gateway_forwards_flag_to_all_adapters():
    renderer = FakeRenderer()
    gateway = RolloutGateway(backend=FakeBackend(renderer), renderer=renderer, override_client_sampling=True)
    assert gateway.adapters and all(a.override_client_sampling for a in gateway.adapters)
