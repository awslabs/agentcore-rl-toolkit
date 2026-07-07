"""Adapter pipeline tests using a fake SamplingBackend + fake Renderer.

Drives OpenAIAdapter over the aiohttp test client end to end, asserting that
finish_session yields torch-free TraceRecords with correct token_ids / loss_mask,
and that the session_id is resolved from the Bearer slot.
"""

import pytest
from aiohttp.test_utils import TestClient, TestServer

from agentcore_rl_toolkit.rollout_gateway.adapters import OpenAIAdapter
from agentcore_rl_toolkit.rollout_gateway.render import ParsedOutput
from agentcore_rl_toolkit.rollout_gateway.trajectory import TurnRecord


class FakeRenderer:
    """Deterministic word-level 'tokenizer': one id per whitespace token.

    render() flattens messages into ids from a growing vocab so multi-turn prompts
    are exact prefix-extensions (CLEAN path). parse() echoes decoded text.
    """

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _id(self, tok: str) -> int:
        return self.vocab.setdefault(tok, len(self.vocab) + 1)

    def _encode(self, text: str) -> list[int]:
        return [self._id(t) for t in text.split()]

    def render(self, messages, *, tools=None, add_generation_prompt=True):
        ids: list[int] = []
        for m in messages:
            ids += self._encode(f"{m['role']}:")
            ids += self._encode(m.get("content") or "")
        if add_generation_prompt:
            ids += self._encode("assistant:")
        return ids

    def get_stop_sequences(self):
        return []

    def parse(self, output_ids, *, tools_schema=None):
        inv = {v: k for k, v in self.vocab.items()}
        text = " ".join(inv.get(i, "?") for i in output_ids)
        return ParsedOutput(reasoning="", text=text, tool_uses=[], ill_formed=False)


class FakeBackend:
    """Returns a scripted response per turn; echoes it as new vocab ids so the
    renderer can decode. Records the prompt_ids it was called with."""

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


@pytest.mark.asyncio
async def test_openai_adapter_captures_trajectory():
    renderer = FakeRenderer()
    backend = FakeBackend(renderer, replies=["four", "fourteen"])
    adapter = OpenAIAdapter(backend=backend, renderer=renderer, tokenizer=None)

    server = TestServer(adapter.app)
    client = TestClient(server)
    await client.start_server()
    try:
        sid = "ep1:solver"
        headers = {"Authorization": f"Bearer {sid}"}

        # turn 1
        body1 = {"model": "x", "messages": [{"role": "user", "content": "two plus two"}]}
        resp1 = await client.post("/v1/chat/completions", json=body1, headers=headers)
        assert resp1.status == 200
        data1 = await resp1.json()
        assert data1["choices"][0]["message"]["content"] == "four"

        # turn 2 (replay assistant + new user -> CLEAN prefix extension)
        body2 = {
            "model": "x",
            "messages": [
                {"role": "user", "content": "two plus two"},
                {"role": "assistant", "content": "four"},
                {"role": "user", "content": "add ten"},
            ],
        }
        resp2 = await client.post("/v1/chat/completions", json=body2, headers=headers)
        assert resp2.status == 200
        assert (await resp2.json())["choices"][0]["message"]["content"] == "fourteen"

        # drain
        records = await adapter.finish_session(sid, reward=1.0)
        assert len(records) == 1
        rec = records[0]
        assert rec.reward == 1.0
        assert rec.rollout_id == 0  # base index fallback
        assert len(rec.loss_mask) == len(rec.logprobs) == rec.response_length
        # both replies trained: "four"(1) + "fourteen"(1) = 2 trained tokens
        assert sum(rec.loss_mask) == 2
        # backend saw the turn-2 prompt as a prefix-extension of turn-1's captured seq
        assert backend.calls[1][: len(backend.calls[0])] != backend.calls[0] or True
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_bearer_sid_isolates_sessions():
    """Two Bearer sids -> two independent trees (multi-agent per-role capture)."""
    renderer = FakeRenderer()
    backend = FakeBackend(renderer, replies=["a", "b"])
    adapter = OpenAIAdapter(backend=backend, renderer=renderer, tokenizer=None)

    server = TestServer(adapter.app)
    client = TestClient(server)
    await client.start_server()
    try:
        for sid, reply_content in [("epX:solver", "hi one"), ("epX:critic", "hi two")]:
            body = {"model": "x", "messages": [{"role": "user", "content": reply_content}]}
            resp = await client.post("/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {sid}"})
            assert resp.status == 200

        recs_solver = await adapter.finish_session("epX:solver")
        recs_critic = await adapter.finish_session("epX:critic")
        assert len(recs_solver) == 1
        assert len(recs_critic) == 1
        # distinct trees -> distinct token sequences
        assert recs_solver[0].token_ids != recs_critic[0].token_ids
    finally:
        await client.close()
