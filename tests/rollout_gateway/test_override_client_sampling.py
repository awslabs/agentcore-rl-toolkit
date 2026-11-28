"""Tests for the opt-in ``override_client_sampling`` gateway flag.

Harnesses run unmodified and may send their own ``temperature`` / ``top_p`` /
``top_k``. When the flag is on, the trainer's per-session sampling defaults win,
so the behaviour policy matches the one the trainer recomputes logprobs under.
``max_tokens`` keeps its existing min(trainer, client) behaviour either way.
"""

import pytest
from aiohttp.test_utils import TestClient, TestServer

from agentcore_rl_toolkit.rollout_gateway import RolloutGateway
from agentcore_rl_toolkit.rollout_gateway.adapters import AnthropicAdapter, OpenAIAdapter

from .test_require_registered_sessions import FakeBackend, FakeRenderer

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
    backend = FakeBackend(renderer, replies=["ok"])
    captured = []
    orig = backend.generate

    async def generate(**kw):
        captured.append(kw["sampling_params"])
        return await orig(**kw)

    backend.generate = generate
    adapter = OpenAIAdapter(backend=backend, renderer=renderer, tokenizer=None)
    assert adapter.override_client_sampling is False
    adapter.open_session("s", sampling_defaults=TRAINER)
    await _post(adapter.app, "/v1/chat/completions", _openai(**CLIENT), {"Authorization": "Bearer s"})
    assert {k: captured[0][k] for k in CLIENT} == CLIENT


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
    backend = FakeBackend(renderer, replies=["ok"])
    captured = []
    orig = backend.generate

    async def generate(**kw):
        captured.append(kw["sampling_params"])
        return await orig(**kw)

    backend.generate = generate
    adapter = cls(backend=backend, renderer=renderer, tokenizer=None, override_client_sampling=True)
    adapter.open_session("s", sampling_defaults=TRAINER)
    await _post(adapter.app, path, body, headers)
    sp = captured[0]
    assert (sp["temperature"], sp["top_p"], sp["top_k"]) == (1.0, 1.0, -1)
    assert sp["max_new_tokens"] == 64  # still min(trainer 512, client 64)


@pytest.mark.asyncio
async def test_flag_on_client_cannot_add_keys_the_trainer_left_unset():
    renderer = FakeRenderer()
    backend = FakeBackend(renderer, replies=["ok"])
    captured = []
    orig = backend.generate

    async def generate(**kw):
        captured.append(kw["sampling_params"])
        return await orig(**kw)

    backend.generate = generate
    adapter = OpenAIAdapter(backend=backend, renderer=renderer, tokenizer=None, override_client_sampling=True)
    adapter.open_session("s", sampling_defaults={"temperature": 1.0})
    await _post(adapter.app, "/v1/chat/completions", _openai(top_p=0.5, top_k=5), {"Authorization": "Bearer s"})
    assert "top_p" not in captured[0] and "top_k" not in captured[0]


def test_gateway_forwards_flag_to_all_adapters():
    renderer = FakeRenderer()
    gateway = RolloutGateway(backend=FakeBackend(renderer, []), renderer=renderer, override_client_sampling=True)
    assert gateway.adapters and all(a.override_client_sampling for a in gateway.adapters)
