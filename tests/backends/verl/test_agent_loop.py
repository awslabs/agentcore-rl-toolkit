"""AgentCoreAgentLoop tests: conversion math, invoke wiring, rewards, and failure
paths. The loop is constructed through verl's own ``AgentLoopBase`` against a live
(threaded) gateway; only the LLM server client and the AgentCoreHttpClient (no AWS) are
faked."""

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from agentcore_rl_toolkit.backends.verl import agent_loop as al
from agentcore_rl_toolkit.backends.verl.agent_loop import AgentCoreAgentLoop
from agentcore_rl_toolkit.concurrency.priority_semaphore import LocalPrioritySemaphore
from agentcore_rl_toolkit.rollout_gateway import TraceRecord
from agentcore_rl_toolkit.rollout_session import agentcore_http_session as http_session
from agentcore_rl_toolkit.runtime import InvocationError

from .conftest import FakeLLMServerClient, FakeTokenizer, make_data_config, make_trainer_config

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def mock_http_client(monkeypatch):
    client = MagicMock(stop_session=AsyncMock())
    monkeypatch.setattr(http_session, "AgentCoreHttpClient", MagicMock(return_value=client))


def _make_loop(llm_client=None, *, use_v1=True, trainer_config=None, **loop_kwargs):
    loop_kwargs.setdefault("max_tokens_per_turn", 8)
    with patch.object(al.ray, "get_actor") as get_actor:
        get_actor.return_value.wait_async.remote = AsyncMock()
        get_actor.return_value.acquire.remote = AsyncMock()
        get_actor.return_value.release.remote = AsyncMock()
        loop = AgentCoreAgentLoop(
            trainer_config or make_trainer_config(use_v1=use_v1),
            llm_client or FakeLLMServerClient(),
            FakeTokenizer(),
            None,
            None,
            make_data_config(),
            rollout_session_backend={
                "backend": "agentcore_rl_toolkit.rollout_session.agentcore_http_session.AgentCoreHttpSession",
                "agentcore_runtime_arn": "arn:aws:bedrock-agentcore:us-west-2:123:runtime/test",
            },
            gateway_bind_host="127.0.0.1",
            gateway_public_host="127.0.0.1",
            name="agentcore_agent",  # hydra passes the YAML entry's name through
            **loop_kwargs,
        )
    loop.test_client = http_session.AgentCoreHttpClient.return_value
    return loop


def _wire_result(loop, result: dict[str, Any], *, drive_turns: int = 1, error: Exception | None = None):
    """Make invoke return a future whose result first drives
    ``drive_turns`` chat turns against the live gateway (simulating the ACR
    agent calling back in), then returns ``result``."""

    async def fake_invoke(payload, session_id=None, invocation_id=None, background=False):
        future = MagicMock()

        async def get_result():
            async with aiohttp.ClientSession() as http:
                messages = [{"role": "user", "content": "hi"}]
                for _ in range(drive_turns):
                    resp = await http.post(
                        f"{loop._gateway.base_url}/v1/chat/completions",
                        json={"model": "m", "messages": messages},
                        headers={"Authorization": f"Bearer {session_id}"},
                    )
                    assert resp.status == 200
                    body = await resp.json()
                    messages = messages + [body["choices"][0]["message"], {"role": "user", "content": "more"}]
            if error:
                raise error
            return result

        future.result = get_result
        call = {"payload": payload, "session_id": session_id, "invocation_id": invocation_id, "background": background}
        fake_invoke.calls.append(call)
        return future

    fake_invoke.calls = []
    loop.test_client.invoke = fake_invoke
    return fake_invoke


async def test_run_end_to_end_inline_reward():
    llm = FakeLLMServerClient()
    loop = _make_loop(llm)
    invoke = _wire_result(loop, {"reward": 0.75})

    outputs = await loop.run(
        {"temperature": 0.7},
        raw_prompt=[{"role": "user", "content": "hi"}],
        payload={"question": "2+2?"},
        uid="u1",
    )

    assert len(outputs) == 1
    out = outputs[0]
    # response region = the two generated tokens from FakeLLMServerClient
    assert out.response_ids == [101, 102]
    assert out.response_mask == [1, 1]
    assert out.response_logprobs == [-0.5, -0.6]
    assert len(out.prompt_ids) > 0
    assert out.reward_score == 0.75
    assert out.extra_fields["acr_result"]["reward"] == 0.75
    assert out.extra_fields["reward_extra_info"] == {"agent_reward": 0.75}
    assert out.extra_fields["metrics"] == {
        "acr_failed": 0.0,
        "num_trace_records": 1.0,
    }
    assert "num_trace_records" not in out.extra_fields

    # invoke wiring: sid is a 36-char uuid used as both ACR session id and Bearer sid
    call = invoke.calls[0]
    assert len(call["session_id"]) == 36
    assert out.extra_fields["acr_session_id"] == call["session_id"]
    # this loop hands the agent its gateway session key via _config.api_key
    assert call["payload"]["_config"]["api_key"] == call["session_id"]
    # OpenAI-SDK convention: the advertised base_url carries the /v1 prefix
    assert call["payload"]["_config"]["base_url"] == f"{loop._gateway.base_url}/v1"
    assert call["payload"]["_config"]["model_id"] == "test/model"
    assert call["background"] is True
    assert call["invocation_id"]
    loop.test_client.stop_session.assert_awaited_once_with(call["session_id"])
    # the payload column is forwarded verbatim; row plumbing never leaks in
    assert {k: v for k, v in call["payload"].items() if k != "_config"} == {"question": "2+2?"}
    # LLM client got the sid as sticky request_id
    assert llm.calls[0]["request_id"] == call["session_id"]


@pytest.mark.parametrize("reward", [None, "invalid", "1.0", True, [], [1.0], float("nan"), float("inf")])
async def test_invalid_reward_raises_and_stops_session(reward):
    loop = _make_loop()
    invoke = _wire_result(loop, {"reward": reward})
    with pytest.raises(ValueError):
        await loop.run({}, payload={"prompt": "hi"})
    sid = invoke.calls[0]["session_id"]
    loop.test_client.stop_session.assert_awaited_once_with(sid)


async def test_missing_reward_raises():
    loop = _make_loop()
    _wire_result(loop, {"artifacts": {"x": 1}})
    with pytest.raises(ValueError, match="finite scalar agent reward"):
        await loop.run({}, payload={"prompt": "hi"})


async def test_separate_reward_mode_rejected():
    """Trainer-side scoring needs data_source/reward_model.ground_truth columns
    the payload-first dataset contract doesn't provide; rejected at construction
    rather than KeyError-ing inside verl's reward manager."""
    with pytest.raises(ValueError, match="reward_mode='separate' is not supported"):
        _make_loop(reward_mode="separate")


async def test_invalid_reward_mode_rejected():
    with pytest.raises(ValueError, match="reward_mode"):
        _make_loop(reward_mode="nope")


async def test_multi_turn_merges_to_one_record():
    loop = _make_loop()
    _wire_result(loop, {"reward": 1.0}, drive_turns=3)
    outputs = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")
    # CLEAN prefix extensions merge into ONE record with 3 trained turns
    assert len(outputs) == 1
    assert sum(outputs[0].response_mask) == 6  # 3 turns x 2 generated tokens
    assert outputs[0].num_turns == 4  # 3 LLM turns + 1


async def test_timeout_without_trace_raises():
    loop = _make_loop()

    async def fake_invoke(payload, session_id=None, invocation_id=None, background=False):
        future = MagicMock()
        future.result = AsyncMock(side_effect=TimeoutError())
        return future

    loop.test_client.invoke = fake_invoke
    with pytest.raises(RuntimeError, match="timed out"):
        await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")


async def test_invoke_error_without_trace_raises():
    loop = _make_loop()
    loop.test_client.invoke = AsyncMock(side_effect=RuntimeError("throttled"))
    with pytest.raises(RuntimeError, match="throttled"):
        await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")


async def test_agent_error_with_trace_scores_zero():
    """A persisted handler error still leaves a real partial trajectory."""
    loop = _make_loop()
    error = InvocationError("session", {"version": 1, "invocation_id": "inv", "status": "completed", "error": "boom"})
    invoke = _wire_result(loop, {}, error=error)
    outputs = await loop.run({}, payload={"prompt": "hi"})
    assert outputs[0].reward_score == 0.0
    assert "boom" in outputs[0].extra_fields["acr_error"]
    assert outputs[0].extra_fields["metrics"] == {"acr_failed": 1.0, "num_trace_records": 1.0}
    assert outputs[0].extra_fields["reward_extra_info"] == {"agent_reward": 0.0}
    assert sum(outputs[0].response_mask) == 2
    loop.test_client.stop_session.assert_awaited_once_with(invoke.calls[0]["session_id"])


async def test_application_status_code_is_opaque():
    loop = _make_loop()
    _wire_result(loop, {"status_code": 500, "reward": 0.9})
    outputs = await loop.run({}, payload={"prompt": "hi"})
    assert outputs[0].reward_score == 0.9


async def test_use_v1_required():
    with pytest.raises(ValueError, match="use_v1"):
        _make_loop(use_v1=False)


async def test_static_session_warning(caplog):
    """Stale agent image: the agent calls in with api_key='EMPTY' instead of its
    ACR session id, so the real sid drains empty and raises after warning about
    the static session."""
    loop = _make_loop()

    async def fake_invoke(payload, session_id=None, invocation_id=None, background=False):
        future = MagicMock()

        async def get_result():
            async with aiohttp.ClientSession() as http:
                resp = await http.post(
                    f"{loop._gateway.base_url}/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer EMPTY"},
                )
                assert resp.status == 200
            return {"reward": 1.0}

        future.result = get_result
        return future

    loop.test_client.invoke = fake_invoke
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="no trainable trajectory"):
            await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")

    assert any("static session 'EMPTY'" in r.message for r in caplog.records)


async def test_cancellation_stops_session_before_releasing_concurrency_slot():
    loop = _make_loop()
    loop._rollout_semaphore = LocalPrioritySemaphore(1)
    started = asyncio.Event()

    async def invoke(*args, **kwargs):
        started.set()
        await asyncio.Future()

    async def stop(session_id):
        assert loop._rollout_semaphore.locked()

    loop.test_client.invoke = invoke
    loop.test_client.stop_session.side_effect = stop
    task = asyncio.create_task(loop.run({}, payload={"prompt": "hi"}))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    loop.test_client.stop_session.assert_awaited_once()
    assert loop._rollout_semaphore.value == 1


async def test_no_payload_column_raises():
    """A missing `payload` column is a config error, raised loudly (not degraded
    per-rollout): the payload column is the single dataset contract — plain row
    fields are never forwarded (the row namespace is shared with verl plumbing,
    and verl-shaped column values are not agent-shaped)."""
    loop = _make_loop()
    with pytest.raises(ValueError, match="payload"):
        await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], question="2+2?", uid="u1")


async def test_payload_column_forwarded_verbatim():
    """The `payload` dataset column is the agent's exact invoke payload —
    forwarded as-is; sibling row fields never leak in."""
    loop = _make_loop()
    invoke = _wire_result(loop, {"reward": 1.0})
    await loop.run(
        {},
        raw_prompt=[{"role": "user", "content": "hi"}],
        payload={"prompt": "What is 2+2?", "answer": "4"},
        question="ignored",
        uid="u1",
    )
    task = invoke.calls[0]["payload"]
    assert {k: v for k, v in task.items() if k != "_config"} == {"prompt": "What is 2+2?", "answer": "4"}


async def test_output_conversion_preserves_prompt_overflow():
    loop = _make_loop(
        trainer_config=make_trainer_config(prompt_length=2, response_length=6, max_model_len=6),
        max_tokens_per_turn=4,
    )
    record = TraceRecord(
        token_ids=[1, 2, 3, 4, 5, 6],
        loss_mask=[1, 1],
        logprobs=[-0.5, -0.6],
    )

    out = loop._record_to_output(record, 0, 1.0, 1, {}, 0.1)

    assert out.prompt_ids == [1, 2]
    assert out.response_ids == [3, 4, 5, 6]
    assert out.response_mask == [0, 0, 1, 1]
    assert out.response_logprobs == [0.0, 0.0, -0.5, -0.6]


async def test_session_budget_matches_response_storage():
    llm = FakeLLMServerClient()
    loop = _make_loop(
        llm,
        trainer_config=make_trainer_config(prompt_length=8, response_length=24, max_model_len=32),
    )
    _wire_result(loop, {"reward": 1.0})
    gateway = loop._gateway.gateway
    with patch.object(gateway, "create_session", wraps=gateway.create_session) as create_session:
        await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"})

    max_context_tokens = create_session.call_args.kwargs["max_context_tokens"]
    assert max_context_tokens == 24
    assert max_context_tokens == loop.response_length
    assert max_context_tokens != loop.prompt_length + loop.response_length


async def test_sampling_defaults_passed_to_gateway():
    llm = FakeLLMServerClient()
    loop = _make_loop(llm)
    _wire_result(loop, {"reward": 1.0})
    await loop.run(
        {"temperature": 0.3, "top_p": 0.8, "top_k": 20},
        raw_prompt=[{"role": "user", "content": "hi"}],
        payload={"prompt": "hi"},
    )
    sp = llm.calls[0]["sampling_params"]
    assert sp["temperature"] == 0.3
    assert sp["top_p"] == 0.8
    assert sp["top_k"] == 20
    assert sp["max_new_tokens"] == 8


async def test_explicit_max_model_len_is_required():
    with pytest.raises(ValueError, match="explicit positive.*max_model_len"):
        _make_loop(trainer_config=make_trainer_config(max_model_len=None))


@pytest.mark.parametrize("field", ["prompt_length", "response_length"])
async def test_padded_region_cannot_exceed_max_model_len(field):
    lengths = {field: 129}
    with pytest.raises(ValueError, match=rf"{field} \(129\) cannot exceed .*max_model_len \(128\)"):
        _make_loop(trainer_config=make_trainer_config(max_model_len=128, **lengths))


@pytest.mark.parametrize("value", [0, -1, True, 129])
async def test_invalid_max_tokens_per_turn_rejected(value):
    with pytest.raises(ValueError, match="max_tokens_per_turn"):
        _make_loop(max_tokens_per_turn=value)


async def test_reward_extra_info_carries_scalar_metrics_over_defaults():
    loop = _make_loop(
        FakeLLMServerClient(), reward_extra_info_defaults={"f_beta": 0.0, "num_turns": 0.0, "reward": 0.0}
    )
    _wire_result(
        loop,
        {
            "reward": 0.75,
            "metrics": {"f_beta": 0.6, "num_turns": 3, "undeclared": 9},
        },
    )
    outs = await loop.run(
        {"temperature": 1.0}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"q": 1}, uid="u"
    )
    rei = outs[-1].extra_fields["reward_extra_info"]
    assert rei["f_beta"] == 0.6 and rei["num_turns"] == 3.0
    assert "acr_failed" not in rei and "num_trace_records" not in rei
    assert "reward" not in rei  # verl derives its reward metric from rm_scores
    # ...but the agent's own scalar is carried under a non-colliding key, so it does not
    # append to the list verl fills from rm_scores.
    assert rei["agent_reward"] == 0.75
    assert "undeclared" not in rei and "note" not in rei


async def test_failed_rollout_carries_reward_extra_info_defaults():
    loop = _make_loop(FakeLLMServerClient(), reward_extra_info_defaults={"f_beta": 0.0})
    rei = loop._reward_extra_info(None, 0.0)
    assert rei == {"f_beta": 0.0, "agent_reward": 0.0}


async def test_reward_extra_info_keys_are_stable_across_loop_instances():
    defaults = {"submitted": 0.0, "num_turns": 0.0, "ok": 0.0}
    successful_loop = _make_loop(FakeLLMServerClient(), reward_extra_info_defaults=defaults)
    failed_loop = _make_loop(FakeLLMServerClient(), reward_extra_info_defaults=defaults)

    successful = successful_loop._reward_extra_info({"metrics": {"submitted": 1.0, "num_turns": 4, "ok": True}}, 0.75)
    failed = failed_loop._reward_extra_info(None, 0.0)

    assert successful == {
        "submitted": 1.0,
        "num_turns": 4.0,
        "ok": 1.0,
        "agent_reward": 0.75,
    }
    assert failed == {
        "submitted": 0.0,
        "num_turns": 0.0,
        "ok": 0.0,
        "agent_reward": 0.0,
    }
    assert set(successful) == set(failed)
