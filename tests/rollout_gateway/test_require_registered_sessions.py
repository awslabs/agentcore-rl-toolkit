"""Tests for the opt-in ``require_registered_sessions`` gateway guard.

Drives the adapters over the aiohttp test client (same fixture style as
test_adapters.py) to assert:

  (a) default mode: an unknown sid still implicitly opens a session (unchanged);
  (b) flag on: an unregistered Bearer is refused 401 and leaves no session state;
  (c) flag on: after open_session(sid) the same request succeeds;
  (d) flag on: unauthenticated endpoints (/healthz, /v1/models) are unaffected;
  (e) the guard lives in the shared turn handler, so the Anthropic route enforces it;
  (f) RolloutGateway forwards the flag to every co-mounted adapter.
"""

import pytest
from aiohttp.test_utils import TestClient, TestServer

from agentcore_rl_toolkit.rollout_gateway import RolloutGateway
from agentcore_rl_toolkit.rollout_gateway.adapters import AnthropicAdapter, OpenAIAdapter
from agentcore_rl_toolkit.rollout_gateway.render import ParsedOutput
from agentcore_rl_toolkit.rollout_gateway.trajectory import TurnRecord


class FakeRenderer:
    """Deterministic word-level 'tokenizer': one id per whitespace token."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _id(self, tok: str) -> int:
        return self.vocab.setdefault(tok, len(self.vocab) + 1)

    def _encode(self, text: str) -> list[int]:
        return [self._id(t) for t in text.split()]

    async def render(self, messages, *, tools=None, add_generation_prompt=True):
        ids: list[int] = []
        for m in messages:
            ids += self._encode(f"{m['role']}:")
            content = m.get("content") or ""
            if isinstance(content, list):
                content = " ".join(block.get("text", "") for block in content)
            ids += self._encode(content)
        if add_generation_prompt:
            ids += self._encode("assistant:")
        return ids

    def get_stop_sequences(self):
        return []

    def parse(self, output_ids, *, finish_reason, tools_schema=None):
        inv = {v: k for k, v in self.vocab.items()}
        text = " ".join(inv.get(i, "?") for i in output_ids)
        return ParsedOutput(reasoning="", text=text, tool_uses=[], ill_formed=False)


class FakeBackend:
    """Returns a scripted response per turn; echoes it as new vocab ids."""

    def __init__(self, renderer: FakeRenderer, replies: list[str]):
        self.renderer = renderer
        self.replies = list(replies)
        self.calls: list[list[int]] = []

    async def generate(self, *, prompt_ids, sampling_params, session_id=None, image_data=None, video_data=None):
        self.calls.append(list(prompt_ids))
        reply = self.replies.pop(0)
        out_ids = self.renderer._encode(reply)
        return TurnRecord(
            prompt_ids=list(prompt_ids),
            output_ids=out_ids,
            finish_reason="stop",
            output_log_probs=[-0.5] * len(out_ids),
        )


def _openai_body(content: str = "two plus two") -> dict:
    return {"model": "x", "messages": [{"role": "user", "content": content}]}


async def _serve(app):
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


@pytest.mark.asyncio
async def test_default_mode_unknown_sid_opens_session():
    renderer = FakeRenderer()
    backend = FakeBackend(renderer, replies=["four"])
    adapter = OpenAIAdapter(backend=backend, renderer=renderer, tokenizer=None)
    assert adapter.require_registered_sessions is False

    client = await _serve(adapter.app)
    try:
        resp = await client.post(
            "/v1/chat/completions", json=_openai_body(), headers={"Authorization": "Bearer stranger"}
        )
        assert resp.status == 200
        assert "stranger" in adapter.store
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_flag_on_unregistered_bearer_rejected():
    renderer = FakeRenderer()
    backend = FakeBackend(renderer, replies=["four"])
    adapter = OpenAIAdapter(backend=backend, renderer=renderer, tokenizer=None, require_registered_sessions=True)

    client = await _serve(adapter.app)
    try:
        resp = await client.post(
            "/v1/chat/completions", json=_openai_body(), headers={"Authorization": "Bearer stranger"}
        )
        assert resp.status == 401
        assert (await resp.text()) == "unknown session"
        assert "stranger" not in adapter.store
        assert adapter.manager.turn_count("stranger") == 0
        assert backend.calls == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_flag_on_registered_sid_succeeds():
    renderer = FakeRenderer()
    backend = FakeBackend(renderer, replies=["four"])
    adapter = OpenAIAdapter(backend=backend, renderer=renderer, tokenizer=None, require_registered_sessions=True)

    client = await _serve(adapter.app)
    try:
        sid = "ep1:solver"
        adapter.open_session(sid)  # what RolloutGateway.create_session calls
        resp = await client.post(
            "/v1/chat/completions", json=_openai_body(), headers={"Authorization": f"Bearer {sid}"}
        )
        assert resp.status == 200
        assert (await resp.json())["choices"][0]["message"]["content"] == "four"
        assert adapter.manager.turn_count(sid) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_flag_on_unauthenticated_endpoints_unaffected():
    renderer = FakeRenderer()
    backend = FakeBackend(renderer, replies=[])
    adapter = OpenAIAdapter(backend=backend, renderer=renderer, tokenizer=None, require_registered_sessions=True)

    client = await _serve(adapter.app)
    try:
        for path in ("/healthz", "/v1/models"):
            resp = await client.get(path)
            assert resp.status == 200
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_flag_on_anthropic_adapter_rejects_unregistered():
    renderer = FakeRenderer()
    backend = FakeBackend(renderer, replies=["four"])
    adapter = AnthropicAdapter(backend=backend, renderer=renderer, tokenizer=None, require_registered_sessions=True)

    client = await _serve(adapter.app)
    try:
        body = {"model": "x", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
        resp = await client.post("/v1/messages", json=body, headers={"x-api-key": "stranger"})
        assert resp.status == 401
        assert backend.calls == []
        adapter.open_session("known")
        ok = await client.post("/v1/messages", json=body, headers={"x-api-key": "known"})
        assert ok.status == 200
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_gateway_forwards_flag_to_all_adapters():
    renderer = FakeRenderer()
    backend = FakeBackend(renderer, replies=["four"])
    gateway = RolloutGateway(backend=backend, renderer=renderer, require_registered_sessions=True)
    assert gateway.adapters and all(a.require_registered_sessions for a in gateway.adapters)

    client = await _serve(gateway.app)
    try:
        resp = await client.post(
            "/v1/chat/completions", json=_openai_body(), headers={"Authorization": "Bearer stranger"}
        )
        assert resp.status == 401
        gateway.create_session("known")
        ok = await client.post("/v1/chat/completions", json=_openai_body(), headers={"Authorization": "Bearer known"})
        assert ok.status == 200
    finally:
        await client.close()
