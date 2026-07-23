"""Real end-to-end integration test: RolloutGateway against a live SGLang server."""

import importlib.util
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager

import pytest

from agentcore_rl_toolkit.rollout_gateway import BaseTrace, HfTemplateRenderer, RolloutGateway

SGLANG_URL = os.environ.get("E2E_SGLANG_URL", "http://localhost:30000")
MODEL = os.environ.get("E2E_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS = 512
SERVER_LOG = "/tmp/e2e_sglang_server.log"

HAVE_DEPS = all(importlib.util.find_spec(m) for m in ("openai", "sglang"))

pytestmark = [
    pytest.mark.skipif(not HAVE_DEPS, reason="needs openai + sglang (real-model E2E)"),
    pytest.mark.asyncio,
]


# ---------------------------------------------------------------------------
# server lifecycle (session fixture owns the SGLang process)
# ---------------------------------------------------------------------------


def server_up() -> bool:
    try:
        with urllib.request.urlopen(f"{SGLANG_URL}/v1/models", timeout=5):
            return True
    except (urllib.error.URLError, OSError):
        return False


@pytest.fixture(scope="session")
def server():
    """Launch sglang unless one is already running at E2E_SGLANG_URL."""
    if server_up():
        yield  # external server; don't own its lifecycle
        return
    with open(SERVER_LOG, "w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "sglang.launch_server", "--model-path", MODEL]
            + ["--port", SGLANG_URL.rsplit(":", 1)[-1], "--mem-fraction-static", "0.5"],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group -> clean teardown
        )
    while not server_up():
        assert proc.poll() is None, f"sglang server died; see {SERVER_LOG}"
        time.sleep(3)
    yield
    os.killpg(proc.pid, signal.SIGTERM)
    proc.wait(timeout=30)


# ---------------------------------------------------------------------------
# gateway assembly: real renderer + real SGLang backend (with call recording)
# ---------------------------------------------------------------------------


class RecordingBackend:
    """Delegates to the real SglangHttpBackend, recording each turn so the test
    can make token-exact assertions against what SGLang actually returned."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[dict] = []

    async def generate(self, **kwargs):
        turn = await self.inner.generate(**kwargs)
        self.calls.append(
            {
                "prompt_ids": list(turn.prompt_ids),
                "output_ids": list(turn.output_ids),
                "logprobs": list(turn.output_log_probs),
                "session_id": kwargs.get("session_id"),
            }
        )
        return turn


_TOKENIZER = None


def make_gateway():
    global _TOKENIZER
    from agentcore_rl_toolkit.rollout_gateway.sampling_backends.sglang_http import SglangHttpBackend

    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL)
    backend = RecordingBackend(SglangHttpBackend(SGLANG_URL))
    renderer = HfTemplateRenderer(_TOKENIZER)
    gateway = RolloutGateway(backend=backend, renderer=renderer, tokenizer=_TOKENIZER, adapters=["openai"])
    return gateway, backend, renderer, _TOKENIZER


@asynccontextmanager
async def serve(gateway: RolloutGateway):
    from aiohttp import web

    runner = web.AppRunner(gateway.app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def _trained(rec) -> list[int]:
    """The mask=1 token ids from a record's response region."""
    tail = rec.token_ids[-rec.response_length :]
    return [tok for tok, m in zip(tail, rec.loss_mask, strict=True) if m]


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


async def test_single_turn_token_exact(server):
    """One real generation; the TraceRecord must be token-for-token what SGLang saw."""
    gateway, backend, renderer, tokenizer = make_gateway()
    sid = "e2e:single"
    gateway.create_session(sid, sampling_defaults={"temperature": 0.0})
    messages = [{"role": "user", "content": "What is 2+2? Reply with just the number. /no_think"}]

    async with serve(gateway) as base_url:
        import openai

        client = openai.AsyncOpenAI(base_url=f"{base_url}/v1", api_key=sid)
        resp = await client.chat.completions.create(model=MODEL, messages=messages, max_tokens=MAX_NEW_TOKENS)
        content = resp.choices[0].message.content
        assert content and "4" in content, f"model answered: {content!r}"

        records = await gateway.finish_session(sid, base_sample=BaseTrace(rollout_id="e2e-1"), reward=1.0)

    assert len(backend.calls) == 1
    prompt_ids, output_ids = backend.calls[0]["prompt_ids"], backend.calls[0]["output_ids"]
    # the gateway rendered the prompt with the REAL Qwen chat template
    assert prompt_ids == renderer.render(messages, tools=None, add_generation_prompt=True)
    assert backend.calls[0]["session_id"] == sid

    assert len(records) == 1
    rec = records[0]
    # token-exact: the record is exactly prompt + generation, nothing re-tokenized
    assert rec.token_ids == prompt_ids + output_ids
    assert rec.response_length == len(output_ids)
    assert rec.loss_mask == [1] * len(output_ids)
    # logprobs are SGLang's real sampling logprobs, passed through untouched
    assert rec.logprobs == backend.calls[0]["logprobs"]
    assert all(lp <= 0.0 for lp in rec.logprobs)
    # decoded artifacts round-trip through the real tokenizer
    assert rec.response == tokenizer.decode(output_ids, skip_special_tokens=False)
    assert renderer.parse(output_ids).text == content
    assert rec.reward == 1.0 and rec.rollout_id == "e2e-1"
    assert rec.metadata["truncated"] is False and rec.metadata["ill_formed"] is False


async def test_multi_turn_drift_healed_into_one_record(server):
    """Two turns with the echoed history. Qwen3's template strips the <think> block
    from the replayed assistant turn, so turn 2's prompt does NOT prefix-extend the
    captured tokens — the manager must heal that real drift (REALIGN) rather than
    fork, yielding ONE record whose trained tail is exactly turn 2's generation."""
    gateway, backend, renderer, tokenizer = make_gateway()
    sid = "e2e:multi"
    gateway.create_session(sid, sampling_defaults={"temperature": 0.0})

    async with serve(gateway) as base_url:
        import openai

        client = openai.AsyncOpenAI(base_url=f"{base_url}/v1", api_key=sid)
        history = [{"role": "user", "content": "What is 2+2? Reply with just the number. /no_think"}]
        resp1 = await client.chat.completions.create(model=MODEL, messages=history, max_tokens=MAX_NEW_TOKENS)
        content1 = resp1.choices[0].message.content

        history += [
            {"role": "assistant", "content": content1},
            {"role": "user", "content": "Now add 10 to that. Reply with just the number. /no_think"},
        ]
        resp2 = await client.chat.completions.create(model=MODEL, messages=history, max_tokens=MAX_NEW_TOKENS)
        assert resp2.choices[0].message.content

        records = await gateway.finish_session(sid, base_sample=BaseTrace(rollout_id="e2e-2"), reward=0.5)

    assert len(backend.calls) == 2
    out1, out2 = backend.calls[0]["output_ids"], backend.calls[1]["output_ids"]

    # ONE record: the manager absorbed the drift instead of splitting the episode
    assert len(records) == 1
    rec = records[0]
    assert rec.reward == 0.5 and rec.rollout_id == "e2e-2"
    assert len(rec.loss_mask) == len(rec.logprobs) == rec.response_length

    # turn 2's generation is trained, token-exact, at the end of the sequence
    assert rec.token_ids[-len(out2) :] == out2
    assert rec.loss_mask[-len(out2) :] == [1] * len(out2)
    assert rec.logprobs[-len(out2) :] == backend.calls[1]["logprobs"]

    trained = _trained(rec)
    if trained == out2:
        # REALIGN path: turn 1's drifted response span was overwritten as context
        assert sum(rec.loss_mask) == len(out2)
    else:
        # CLEAN path (no drift for this template/content): both turns trained
        assert trained == out1 + out2
    # mask=0 positions carry no logprob signal
    assert all(lp == 0.0 for lp, m in zip(rec.logprobs, rec.loss_mask, strict=True) if not m)


async def test_rewritten_echo_trains_only_final_turn(server):
    """Non-cumulative capture, rewrite flavor: the client replays history but
    EDITS the echoed assistant turn (as harnesses do — compaction, whitespace,
    annotations). The edited echo no longer dict-matches the generated leaf, so
    the manager merges the rewrite (demotes turn 1 to routing-only) instead of
    stranding it as a trained dead-end: ONE record, only turn 2's generation
    trained, turn 1 present as context only."""
    gateway, backend, renderer, tokenizer = make_gateway()
    sid = "e2e:rewrite"
    gateway.create_session(sid, sampling_defaults={"temperature": 0.0})

    async with serve(gateway) as base_url:
        import openai

        client = openai.AsyncOpenAI(base_url=f"{base_url}/v1", api_key=sid)
        history = [{"role": "user", "content": "What is 2+2? Reply with just the number. /no_think"}]
        resp1 = await client.chat.completions.create(model=MODEL, messages=history, max_tokens=MAX_NEW_TOKENS)
        content1 = resp1.choices[0].message.content

        history += [
            # rewritten echo: content differs from what the model generated
            {"role": "assistant", "content": f"{content1} (verified)"},
            {"role": "user", "content": "Now add 10 to that. Reply with just the number. /no_think"},
        ]
        resp2 = await client.chat.completions.create(model=MODEL, messages=history, max_tokens=MAX_NEW_TOKENS)
        assert resp2.choices[0].message.content

        records = await gateway.finish_session(sid, base_sample=BaseTrace(rollout_id="e2e-3"), reward=0.5)

    assert len(backend.calls) == 2
    out2 = backend.calls[1]["output_ids"]

    # one record; the abandoned turn-1 generation is NOT trained
    assert len(records) == 1
    rec = records[0]
    assert _trained(rec) == out2
    assert sum(rec.loss_mask) == len(out2)
    # turn 2's full prompt (incl. the rewritten turn-1 text) became leading context
    assert rec.response_length == len(out2)
    assert rec.token_ids == backend.calls[1]["prompt_ids"] + out2
    assert rec.logprobs == backend.calls[1]["logprobs"]


async def test_stateless_turns_fork_into_separate_records(server):
    """Non-cumulative context mode: the client sends each request WITHOUT the
    prior history (stateless single-shot calls under one sid). Each turn's
    prompt shares no prefix with the tree, so every turn forks into its own
    branch — one independently-trained record per turn, all under one
    rollout_id."""
    gateway, backend, renderer, tokenizer = make_gateway()
    sid = "e2e:stateless"
    gateway.create_session(sid, sampling_defaults={"temperature": 0.0})
    questions = [
        "What is 2+2? Reply with just the number. /no_think",
        "What is 3*3? Reply with just the number. /no_think",
    ]

    async with serve(gateway) as base_url:
        import openai

        client = openai.AsyncOpenAI(base_url=f"{base_url}/v1", api_key=sid)
        for q in questions:
            resp = await client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": q}], max_tokens=MAX_NEW_TOKENS
            )
            assert resp.choices[0].message.content

        records = await gateway.finish_session(sid, base_sample=BaseTrace(rollout_id="e2e-4"), reward=1.0)

    assert len(backend.calls) == 2
    assert len(records) == 2
    # records come out in leaf order = request order; each is exactly its own turn
    for rec, call in zip(records, backend.calls, strict=True):
        assert rec.token_ids == call["prompt_ids"] + call["output_ids"]
        assert rec.response_length == len(call["output_ids"])
        assert rec.loss_mask == [1] * len(call["output_ids"])
        assert rec.logprobs == call["logprobs"]
        assert rec.rollout_id == "e2e-4"
        assert rec.reward == 1.0
