"""Integration tests for ``RolloutGateway`` — the assembled serving unit."""

import json
from contextlib import asynccontextmanager

import pytest
from aiohttp.test_utils import TestClient, TestServer

from agentcore_rl_toolkit.rollout_gateway import BaseTrace, RolloutGateway
from agentcore_rl_toolkit.rollout_gateway.render import ParsedOutput
from agentcore_rl_toolkit.rollout_gateway.trajectory import TurnRecord


class FakeRenderer:
    """Deterministic word-level 'tokenizer': one id per whitespace token.

    render() flattens messages (role prefix + content + tool calls) into ids from
    a growing vocab, so a replayed turn is an exact prefix-extension of the
    captured sequence (CLEAN path). A tool call renders as ``CALL <name> k=v``,
    which is also the raw output format parse() recognises — so a scripted
    ``CALL ...`` reply round-trips through the wire echo byte-for-byte.
    """

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _id(self, tok: str) -> int:
        return self.vocab.setdefault(tok, len(self.vocab) + 1)

    def _encode(self, text: str) -> list[int]:
        return [self._id(t) for t in text.split()]

    def decode(self, ids, skip_special_tokens=False) -> str:
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv.get(i, "?") for i in ids)

    def render(self, messages, *, tools=None, add_generation_prompt=True):
        ids: list[int] = []
        for m in messages:
            ids += self._encode(f"{m['role']}:")
            ids += self._encode(m.get("content") or "")
            for call in m.get("tool_calls") or []:
                fn = call["function"]
                args = " ".join(f"{k}={v}" for k, v in sorted(fn["arguments"].items()))
                ids += self._encode(f"CALL {fn['name']} {args}")
        if add_generation_prompt:
            ids += self._encode("assistant:")
        return ids

    def get_stop_sequences(self):
        return []

    def parse(self, output_ids, *, tools_schema=None):
        text = self.decode(output_ids)
        if text.startswith("CALL "):
            parts = text.split()
            args = dict(p.split("=", 1) for p in parts[2:])
            return ParsedOutput(reasoning="", text="", tool_uses=[{"name": parts[1], "input": args}], ill_formed=False)
        return ParsedOutput(reasoning="", text=text, tool_uses=[], ill_formed=False)


class FakeBackend:
    """Returns one scripted reply per generate() call, encoded through the shared
    FakeRenderer vocab. Records every call's prompt_ids / sampling_params / sid."""

    def __init__(self, renderer: FakeRenderer, replies: list[str]):
        self.renderer = renderer
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def generate(self, *, prompt_ids, sampling_params, session_id=None, image_data=None, video_data=None):
        self.calls.append(
            {"prompt_ids": list(prompt_ids), "sampling_params": dict(sampling_params), "session_id": session_id}
        )
        out_ids = self.renderer._encode(self.replies.pop(0))
        return TurnRecord(
            prompt_ids=list(prompt_ids),
            output_ids=out_ids,
            finish_reason="stop",
            output_log_probs=[-0.5] * len(out_ids),
        )


def make_gateway(replies: list[str], **gateway_kwargs) -> tuple[RolloutGateway, FakeBackend]:
    renderer = FakeRenderer()
    backend = FakeBackend(renderer, replies)
    # the renderer doubles as the tokenizer: finish_session only needs .decode()
    gateway = RolloutGateway(backend=backend, renderer=renderer, tokenizer=renderer, **gateway_kwargs)
    return gateway, backend


@asynccontextmanager
async def serve(gateway: RolloutGateway):
    client = TestClient(TestServer(gateway.app))
    await client.start_server()
    try:
        yield client
    finally:
        await client.close()


def bearer(sid: str) -> dict:
    return {"Authorization": f"Bearer {sid}"}


# ---------------------------------------------------------------------------
# full lifecycle: OpenAI tool-calling loop -> one CLEAN TraceRecord
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_tool_loop_end_to_end():
    gateway, backend = make_gateway(replies=["CALL calculator expr=2+2", "the answer is 4"])
    tools = [{"type": "function", "function": {"name": "calculator", "parameters": {"type": "object"}}}]
    sid = "ep1:solver"

    async with serve(gateway) as client:
        gateway.create_session(sid, sampling_defaults={"temperature": 0.7})

        # turn 1: model calls the tool
        body1 = {"model": "m", "messages": [{"role": "user", "content": "two plus two"}], "tools": tools}
        resp1 = await client.post("/v1/chat/completions", json=body1, headers=bearer(sid))
        assert resp1.status == 200
        choice1 = (await resp1.json())["choices"][0]
        assert choice1["finish_reason"] == "tool_calls"
        call = choice1["message"]["tool_calls"][0]
        assert call["function"]["name"] == "calculator"
        assert json.loads(call["function"]["arguments"]) == {"expr": "2+2"}
        # session sampling defaults reached the backend, keyed by sid
        assert backend.calls[0]["sampling_params"]["temperature"] == 0.7
        assert backend.calls[0]["session_id"] == sid

        # turn 2: echo the assistant tool call + tool result (as an OpenAI client would)
        body2 = {
            "model": "m",
            "messages": [
                {"role": "user", "content": "two plus two"},
                {"role": "assistant", "content": None, "tool_calls": choice1["message"]["tool_calls"]},
                {"role": "tool", "tool_call_id": call["id"], "content": "4"},
            ],
            "tools": tools,
        }
        resp2 = await client.post("/v1/chat/completions", json=body2, headers=bearer(sid))
        assert resp2.status == 200
        choice2 = (await resp2.json())["choices"][0]
        assert choice2["message"]["content"] == "the answer is 4"
        assert choice2["finish_reason"] == "stop"

        # turn 2's prompt exactly extends turn 1's captured sequence (CLEAN)
        captured_turn1 = backend.calls[0]["prompt_ids"] + gateway.renderer._encode("CALL calculator expr=2+2")
        assert backend.calls[1]["prompt_ids"][: len(captured_turn1)] == captured_turn1

        records = await gateway.finish_session(
            sid, base_sample=BaseTrace(rollout_id="ep1"), reward=1.0, extra_metadata={"task": "math"}
        )

    assert len(records) == 1  # CLEAN extension -> a single trainable row
    rec = records[0]
    assert rec.rollout_id == "ep1"
    assert rec.reward == 1.0
    assert rec.metadata["task"] == "math"
    assert rec.metadata["use_tool"] is True
    assert rec.metadata["truncated"] is False
    assert len(rec.loss_mask) == len(rec.logprobs) == rec.response_length

    # exactly the two generated replies are trained; interleaved tool/user prompt
    # tokens inside the response region carry loss_mask=0
    tail = rec.token_ids[-rec.response_length :]
    trained = [tok for tok, m in zip(tail, rec.loss_mask, strict=True) if m]
    assert gateway.renderer.decode(trained) == "CALL calculator expr=2+2 the answer is 4"
    assert sum(rec.loss_mask) == len(trained)
    assert all(lp == -0.5 for lp, m in zip(rec.logprobs, rec.loss_mask, strict=True) if m)
    # finish_session decoded the response tail via the tokenizer
    assert rec.response == "CALL calculator expr=2+2 tool: 4 assistant: the answer is 4"


# ---------------------------------------------------------------------------
# one sid across both wire protocols -> one shared trajectory tree
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_and_anthropic_turns_fold_into_one_trajectory():
    gateway, _ = make_gateway(replies=["four", "fourteen"])
    sid = "ep2:mixed"

    async with serve(gateway) as client:
        gateway.create_session(sid)

        # turn 1 over the OpenAI wire
        body1 = {"model": "m", "messages": [{"role": "user", "content": "two plus two"}]}
        resp1 = await client.post("/v1/chat/completions", json=body1, headers=bearer(sid))
        assert resp1.status == 200
        assert (await resp1.json())["choices"][0]["message"]["content"] == "four"

        # turn 2 over the Anthropic wire, replaying turn 1 as history
        body2 = {
            "model": "m",
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": "two plus two"},
                {"role": "assistant", "content": "four"},
                {"role": "user", "content": "add ten"},
            ],
        }
        resp2 = await client.post("/v1/messages", json=body2, headers=bearer(sid))
        assert resp2.status == 200
        data2 = await resp2.json()
        assert data2["role"] == "assistant"
        assert data2["content"][0] == {"type": "text", "text": "fourteen"}
        assert data2["stop_reason"] == "end_turn"

        records = await gateway.finish_session(sid, reward=0.5)

    # both protocol turns landed in the SAME tree and linearized into one row
    assert len(records) == 1
    assert sum(records[0].loss_mask) == 2  # "four" + "fourteen"
    assert records[0].reward == 0.5


# ---------------------------------------------------------------------------
# sub-agent fork: divergent system prompt under one sid -> two records
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sub_agent_fork_under_one_sid():
    gateway, _ = make_gateway(replies=["done main", "done sub"])
    sid = "ep3:harness"

    async with serve(gateway) as client:
        gateway.create_session(sid)
        for system, user in [
            ("you are the main agent", "do the task"),
            ("you are a sub agent", "explore the repo"),
        ]:
            body = {
                "model": "m",
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }
            resp = await client.post("/v1/chat/completions", json=body, headers=bearer(sid))
            assert resp.status == 200

        records = await gateway.finish_session(sid)

    # the sub-agent's system prompt doesn't match the parent's branch -> forked leaf
    assert len(records) == 2
    trained_texts = set()
    for rec in records:
        tail = rec.token_ids[-rec.response_length :]
        trained_texts.add(gateway.renderer.decode([t for t, m in zip(tail, rec.loss_mask, strict=True) if m]))
    assert trained_texts == {"done main", "done sub"}


# ---------------------------------------------------------------------------
# session lifecycle guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_lifecycle_guards():
    gateway, _ = make_gateway(replies=["ok"])
    sid = "ep4:solo"

    async with serve(gateway) as client:
        gateway.create_session(sid)
        with pytest.raises(ValueError):
            gateway.create_session(sid)

        body = {"model": "m", "messages": [{"role": "user", "content": "go"}]}
        resp = await client.post("/v1/chat/completions", json=body, headers=bearer(sid))
        assert resp.status == 200

        records = await gateway.finish_session(sid)
        assert len(records) == 1

        # finish closed the sid on EVERY adapter: stragglers on both wires get 503
        resp = await client.post("/v1/chat/completions", json=body, headers=bearer(sid))
        assert resp.status == 503
        anth_body = {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "go"}]}
        resp = await client.post("/v1/messages", json=anth_body, headers=bearer(sid))
        assert resp.status == 503

        # idempotent: a second finish returns []
        assert await gateway.finish_session(sid) == []


@pytest.mark.asyncio
async def test_drop_session_discards_trajectory():
    gateway, _ = make_gateway(replies=["ok"])
    sid = "ep5:dropped"

    async with serve(gateway) as client:
        gateway.create_session(sid)
        body = {"model": "m", "messages": [{"role": "user", "content": "go"}]}
        resp = await client.post("/v1/chat/completions", json=body, headers=bearer(sid))
        assert resp.status == 200

        await gateway.drop_session(sid)
        assert await gateway.finish_session(sid) == []


@pytest.mark.asyncio
async def test_max_turns_per_sid_returns_429():
    gateway, _ = make_gateway(replies=["one", "never"], max_turns_per_sid=1)
    sid = "ep6:capped"

    async with serve(gateway) as client:
        gateway.create_session(sid)
        body = {"model": "m", "messages": [{"role": "user", "content": "go"}]}
        resp = await client.post("/v1/chat/completions", json=body, headers=bearer(sid))
        assert resp.status == 200
        resp = await client.post("/v1/chat/completions", json=body, headers=bearer(sid))
        assert resp.status == 429
        assert (await resp.json())["error"]["type"] == "rate_limit_error"


@pytest.mark.asyncio
async def test_max_context_tokens_short_circuits_with_length_finish():
    gateway, backend = make_gateway(replies=["never sampled"])
    sid = "ep7:tiny"

    async with serve(gateway) as client:
        # prompt renders to 3 ids ("user:", "hi", "assistant:") > budget of 2
        gateway.create_session(sid, max_context_tokens=2)
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        resp = await client.post("/v1/chat/completions", json=body, headers=bearer(sid))
        assert resp.status == 200
        choice = (await resp.json())["choices"][0]
        assert choice["finish_reason"] == "length"
        assert choice["message"]["content"] is None
        assert backend.calls == []  # backend never invoked

        # an empty generated turn trains nothing -> no records
        assert await gateway.finish_session(sid) == []


# ---------------------------------------------------------------------------
# streaming + shared endpoints + adapter selection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_streaming_turn_is_captured():
    gateway, _ = make_gateway(replies=["hello there"])
    sid = "ep8:stream"

    async with serve(gateway) as client:
        gateway.create_session(sid)
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        resp = await client.post("/v1/chat/completions", json=body, headers=bearer(sid))
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/event-stream")

        text = await resp.text()
        lines = [line for line in text.splitlines() if line.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"
        chunks = [json.loads(line[len("data: ") :]) for line in lines[:-1]]
        deltas = [c["choices"][0]["delta"] for c in chunks]
        assert {"role": "assistant"} in deltas
        assert {"content": "hello there"} in deltas
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

        # the streamed turn still lands in the trajectory
        records = await gateway.finish_session(sid)

    assert len(records) == 1
    assert sum(records[0].loss_mask) == 2  # "hello there"


@pytest.mark.asyncio
async def test_health_and_count_tokens_endpoints():
    gateway, _ = make_gateway(replies=[])

    async with serve(gateway) as client:
        for path in ("/healthz", "/v1/models"):
            resp = await client.get(path)
            assert resp.status == 200
            assert (await resp.json()) == {"ok": True}

        resp = await client.post("/v1/messages/count_tokens", json={"messages": []})
        assert resp.status == 200
        assert (await resp.json()) == {"input_tokens": 0}


@pytest.mark.asyncio
async def test_adapter_subset_mounts_only_requested_routes():
    gateway, _ = make_gateway(replies=["hi"], adapters=["openai"])

    async with serve(gateway) as client:
        body = {"model": "m", "messages": [{"role": "user", "content": "hey"}]}
        resp = await client.post("/v1/chat/completions", json=body, headers=bearer("s"))
        assert resp.status == 200
        resp = await client.post("/v1/messages", json=body, headers=bearer("s"))
        assert resp.status == 404
