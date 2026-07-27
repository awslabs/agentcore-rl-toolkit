"""AgentCoreAgentLoop tests: conversion math, invoke wiring, rewards, and failure
paths. The loop is constructed through verl's own ``AgentLoopBase`` against a live
(threaded) gateway; only the LLM server client and the RolloutClient (no AWS) are
faked."""

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from .conftest import FakeLLMServerClient, FakeTokenizer, make_data_config, make_trainer_config

pytestmark = pytest.mark.asyncio


def _make_loop(llm_client=None, *, use_v1=True, **loop_kwargs):
    from agentcore_rl_toolkit.backends.experimental.verl.agent_loop import AgentCoreAgentLoop

    with patch("agentcore_rl_toolkit.backends.experimental.verl.agent_loop.RolloutClient") as client_cls:
        client_cls.return_value = MagicMock()
        loop = AgentCoreAgentLoop(
            make_trainer_config(use_v1=use_v1),
            llm_client or FakeLLMServerClient(),
            FakeTokenizer(),
            None,
            None,
            make_data_config(),
            agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/test",
            s3_bucket="test-bucket",
            gateway_bind_host="127.0.0.1",
            gateway_public_host="127.0.0.1",
            name="agentcore_agent",  # hydra passes the YAML entry's name through
            **loop_kwargs,
        )
    return loop


def _wire_result(loop, result: dict[str, Any], *, drive_turns: int = 1):
    """Make invoke_async return a future whose result_async first drives
    ``drive_turns`` chat turns against the live gateway (simulating the ACR
    agent calling back in), then returns ``result``."""

    async def fake_invoke_async(payload, session_id=None, input_id=None, **overrides):
        future = MagicMock()

        async def result_async(timeout=None):
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
            return result

        future.result_async = result_async
        call = {"payload": payload, "session_id": session_id, "input_id": input_id, **overrides}
        fake_invoke_async.calls.append(call)
        return future

    fake_invoke_async.calls = []
    loop._client.invoke_async = fake_invoke_async
    return fake_invoke_async


async def test_run_end_to_end_inline_reward():
    llm = FakeLLMServerClient()
    loop = _make_loop(llm)
    invoke = _wire_result(loop, {"status_code": 200, "rewards": 0.75})

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
    assert out.extra_fields["acr_result"]["rewards"] == 0.75
    assert out.extra_fields["reward_extra_info"] == {}

    # invoke wiring: sid is a 36-char uuid used as both ACR session id and Bearer sid
    call = invoke.calls[0]
    assert len(call["session_id"]) == 36
    # OpenAI-SDK convention: the advertised base_url carries the /v1 prefix
    assert call["base_url"] == f"{loop._gateway.base_url}/v1"
    assert call["model_id"] == "test/model"
    assert call["input_id"] == "u1"
    # the payload column is forwarded verbatim; row plumbing never leaks in
    assert call["payload"] == {"question": "2+2?"}
    # LLM client got the sid as sticky request_id
    assert llm.calls[0]["request_id"] == call["session_id"]


async def test_built_in_mode_missing_reward_warns_and_scores_zero(caplog):
    loop = _make_loop()  # default reward_mode="built_in"
    _wire_result(loop, {"status_code": 200, "artifacts": {"x": 1}})
    with caplog.at_level(logging.WARNING):
        outputs = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")
    assert outputs[0].reward_score == 0.0
    assert any("returned no {'rewards'" in r.message for r in caplog.records)


async def test_malformed_reward_raises():
    """A non-numeric reward means broken agent-side reward code, which would be
    broken on every rollout — raising keeps the unscored row out of the batch
    instead of silently zeroing it and flattening the GRPO group."""
    loop = _make_loop()
    _wire_result(loop, {"status_code": 200, "rewards": "invalid"})
    with pytest.raises(ValueError, match="non-numeric built-in reward"):
        await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")


async def test_malformed_reward_on_failed_rollout_still_scores_zero():
    """Failures score 0.0 before the reward is even parsed, so a garbage reward
    on an already-failed rollout stays an inert row (no raise)."""
    loop = _make_loop()
    _wire_result(loop, {"status_code": 500, "stop_reason": "boom", "rewards": "invalid"})
    outputs = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")
    assert outputs[0].reward_score == 0.0


async def test_empty_reward_list_treated_as_missing(caplog):
    loop = _make_loop()
    _wire_result(loop, {"status_code": 200, "rewards": []})
    with caplog.at_level(logging.WARNING):
        outputs = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")
    assert outputs[0].reward_score == 0.0
    assert any("returned no {'rewards'" in r.message for r in caplog.records)


async def test_separate_reward_mode_rejected():
    """Trainer-side scoring needs data_source/reward_model.ground_truth columns
    the payload-first dataset contract doesn't provide; rejected at construction
    rather than KeyError-ing inside verl's reward manager."""
    with pytest.raises(ValueError, match="reward_mode='separate' is not supported"):
        _make_loop(reward_mode="separate")


async def test_invalid_reward_mode_rejected():
    from agentcore_rl_toolkit.backends.experimental.verl.agent_loop import AgentCoreAgentLoop

    with pytest.raises(ValueError, match="reward_mode"):
        with patch("agentcore_rl_toolkit.backends.experimental.verl.agent_loop.RolloutClient"):
            AgentCoreAgentLoop(
                make_trainer_config(),
                FakeLLMServerClient(),
                FakeTokenizer(),
                None,
                None,
                make_data_config(),
                agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/test",
                s3_bucket="test-bucket",
                gateway_bind_host="127.0.0.1",
                gateway_public_host="127.0.0.1",
                reward_mode="nope",
            )


async def test_multi_turn_merges_to_one_record():
    loop = _make_loop()
    _wire_result(loop, {"status_code": 200, "rewards": 1.0}, drive_turns=3)
    outputs = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")
    # CLEAN prefix extensions merge into ONE record with 3 trained turns
    assert len(outputs) == 1
    assert sum(outputs[0].response_mask) == 6  # 3 turns x 2 generated tokens
    assert outputs[0].num_turns == 4  # 3 LLM turns + 1


async def test_timeout_returns_degenerate_output():
    loop = _make_loop()

    async def fake_invoke_async(payload, session_id=None, input_id=None, **overrides):
        future = MagicMock()
        future.result_async = AsyncMock(side_effect=TimeoutError())
        return future

    loop._client.invoke_async = fake_invoke_async
    outputs = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")

    assert len(outputs) == 1
    out = outputs[0]
    assert out.extra_fields["acr_failed"] is True
    # [1], not [0]: verl's rollout-correction helper requires >=1 valid response
    # token per row; logprob 0.0 + reward 0 keeps the row inert.
    assert out.response_mask == [1]
    assert out.reward_score == 0.0


async def test_invoke_error_returns_degenerate_output():
    loop = _make_loop()
    loop._client.invoke_async = AsyncMock(side_effect=RuntimeError("throttled"))
    outputs = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")
    assert outputs[0].extra_fields["acr_failed"] is True
    assert "throttled" in outputs[0].extra_fields["acr_error"]


async def test_agent_error_status_with_trace_scores_zero():
    """Agent errored after some LLM turns: trace trains, reward forced to 0.0."""
    loop = _make_loop()
    _wire_result(loop, {"status_code": 500, "stop_reason": "boom", "rewards": 0.9})
    outputs = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")
    assert len(outputs) == 1
    assert outputs[0].reward_score == 0.0  # inline reward ignored on failure
    assert "status_code=500" in outputs[0].extra_fields["acr_error"]
    assert sum(outputs[0].response_mask) == 2  # partial trace still trains


async def test_use_v1_required():
    with pytest.raises(ValueError, match="use_v1"):
        _make_loop(use_v1=False)


async def test_static_session_warning(caplog):
    """Stale agent image: the agent calls in with api_key='EMPTY' instead of its
    ACR session id, so the real sid drains empty -> degenerate output plus a
    warning naming the static session."""
    loop = _make_loop()

    async def fake_invoke_async(payload, session_id=None, input_id=None, **overrides):
        future = MagicMock()

        async def result_async(timeout=None):
            async with aiohttp.ClientSession() as http:
                resp = await http.post(
                    f"{loop._gateway.base_url}/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer EMPTY"},
                )
                assert resp.status == 200
            return {"status_code": 200, "rewards": 1.0}

        future.result_async = result_async
        return future

    loop._client.invoke_async = fake_invoke_async
    with caplog.at_level(logging.WARNING):
        outputs = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")

    assert outputs[0].extra_fields["acr_failed"] is True
    assert any("static session 'EMPTY'" in r.message for r in caplog.records)


async def test_exp_id_derived_from_trainer_config():
    loop = _make_loop()
    # conftest's make_trainer_config has no project/experiment name -> defaults
    assert loop._exp_id == "verl-run"


async def test_client_cached_by_config():
    from agentcore_rl_toolkit.backends.experimental.verl import agent_loop as al

    loop1 = _make_loop()
    n_after_first = len(al._CLIENTS)
    loop2 = _make_loop()  # same config -> same cached client
    assert len(al._CLIENTS) == n_after_first
    assert loop1._client is loop2._client


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
    invoke = _wire_result(loop, {"status_code": 200, "rewards": 1.0})
    await loop.run(
        {},
        raw_prompt=[{"role": "user", "content": "hi"}],
        payload={"prompt": "What is 2+2?", "answer": "4"},
        question="ignored",
        uid="u1",
    )
    assert invoke.calls[0]["payload"] == {"prompt": "What is 2+2?", "answer": "4"}


async def test_truncation_caps():
    llm = FakeLLMServerClient()
    loop = _make_loop(llm)
    loop.response_length = 1  # force response truncation
    _wire_result(loop, {"status_code": 200, "rewards": 1.0})
    outputs = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")
    out = outputs[0]
    assert len(out.response_ids) == 1
    assert len(out.response_mask) == 1
    assert len(out.response_logprobs) == 1


async def test_sampling_defaults_passed_to_gateway():
    llm = FakeLLMServerClient()
    loop = _make_loop(llm)
    _wire_result(loop, {"status_code": 200, "rewards": 1.0})
    await loop.run(
        {"temperature": 0.3, "top_p": 0.8, "top_k": 20},
        raw_prompt=[{"role": "user", "content": "hi"}],
        payload={"prompt": "hi"},
    )
    sp = llm.calls[0]["sampling_params"]
    assert sp["temperature"] == 0.3
    assert sp["top_p"] == 0.8
    assert sp["top_k"] == 20
    # max_new_tokens defaults to response_length
    assert sp["max_new_tokens"] == 32
