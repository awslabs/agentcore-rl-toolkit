"""Gateway wire-contract test using the real OpenAI SDK.

An OpenAI-compatible agent pointed at the gateway with ``api_key=<session id>``
sends it as ``Authorization: Bearer <sid>``, which the gateway resolves as the
capture-session key. This exercises that contract with the actual SDK over real
HTTP — no agent framework required.
"""

import asyncio
import uuid

import openai
import pytest

from agentcore_rl_toolkit.backends.experimental.verl.gateway_host import get_or_start_gateway

from .conftest import FakeLLMServerClient, FakeTokenizer

pytestmark = pytest.mark.asyncio


async def test_openai_sdk_session_capture():
    llm = FakeLLMServerClient()
    handle = get_or_start_gateway(
        server_manager=llm,
        tokenizer=FakeTokenizer(),
        host="127.0.0.1",
        public_host="127.0.0.1",
    )
    sid = str(uuid.uuid4())
    handle.gateway.create_session(sid, sampling_defaults={"max_new_tokens": 32})

    def agent_turns():
        client = openai.OpenAI(base_url=f"{handle.base_url}/v1", api_key=sid)
        messages = [{"role": "user", "content": "What is 2+2?"}]
        for _ in range(2):  # multi-turn: replayed history must prefix-extend
            resp = client.chat.completions.create(model="test/model", messages=messages)
            reply = resp.choices[0].message
            assert reply.content == "hello world"
            messages += [
                {"role": "assistant", "content": reply.content},
                {"role": "user", "content": "and 3+3?"},
            ]

    # The OpenAI SDK is sync; run it off the event loop like a real remote client.
    await asyncio.to_thread(agent_turns)

    # sid rode the Bearer slot into the sticky engine routing key
    assert all(c["request_id"] == sid for c in llm.calls)
    assert len(llm.calls) == 2

    records = await handle.gateway.finish_session(sid, reward=1.0)
    # CLEAN prefix extension: both turns merge into ONE loss-masked record
    assert len(records) == 1
    r = records[0]
    assert r.reward == 1.0
    assert sum(r.loss_mask) == 4  # 2 turns x 2 generated tokens trained
    assert len(r.loss_mask) == len(r.logprobs)


async def test_openai_sdk_finished_session_refused():
    """The adapter is permissive about unseen api keys (an unregistered sid
    implicitly opens a session — how local ``"EMPTY"`` runs work), but a
    *finished* session is closed: stragglers get 503 instead of silently
    starting a new capture under a consumed sid."""
    handle = get_or_start_gateway(
        server_manager=FakeLLMServerClient(),
        tokenizer=FakeTokenizer(),
        host="127.0.0.1",
        public_host="127.0.0.1",
    )
    sid = str(uuid.uuid4())

    def call(expect_closed: bool):
        client = openai.OpenAI(base_url=f"{handle.base_url}/v1", api_key=sid, max_retries=0)
        if expect_closed:
            with pytest.raises(openai.APIStatusError) as exc_info:
                client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
            assert exc_info.value.status_code == 503
        else:
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])

    # implicit open on first use, then drain the session...
    await asyncio.to_thread(call, False)
    records = await handle.gateway.finish_session(sid)
    assert len(records) == 1
    # ...after which the sid is closed to stragglers
    await asyncio.to_thread(call, True)
