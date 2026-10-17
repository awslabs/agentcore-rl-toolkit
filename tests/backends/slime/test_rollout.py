"""Unit tests for the slime rollout helpers.

Covers AgentCoreRLConfig.from_args, _build_payload, _agent_reward,
_run_agent, _aborted, and _to_sample.  Slime is replaced by a fake module (see
conftest.py), so these tests run without the full training stack.
"""

from __future__ import annotations

import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest

from agentcore_rl_toolkit.backends.slime.integration import rollout
from agentcore_rl_toolkit.rollout_gateway.trace import Status as TraceStatus
from agentcore_rl_toolkit.rollout_gateway.trace import TraceRecord
from agentcore_rl_toolkit.rollout_session.agentcore_http_session import AgentCoreHttpSession
from agentcore_rl_toolkit.rollout_session.errors import RolloutContractError
from agentcore_rl_toolkit.runtime import InvocationError

# ---------------------------------------------------------------------------
# AgentCoreRLConfig.from_args
# ---------------------------------------------------------------------------


_ARN = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/test"
_BACKEND = {
    "backend": "agentcore_rl_toolkit.rollout_session.agentcore_http_session.AgentCoreHttpSession",
    "agentcore_runtime_arn": _ARN,
}


def _args(**kwargs):
    defaults = dict(
        rollout_session_backend=_BACKEND,
        gateway_host=None,
        model_id=None,
        max_rollout_time=1800.0,
        tps_limit=25,
        gateway_port=0,
        hf_checkpoint="Qwen/Qwen2.5-0.5B",
        wandb_group=None,
        wandb_project=None,
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


def test_config_from_args_required_fields():
    cfg = rollout.AgentCoreRLConfig.from_args(_args())
    assert cfg.rollout_session_backend == _BACKEND


def test_config_from_args_defaults():
    cfg = rollout.AgentCoreRLConfig.from_args(_args())
    assert cfg.max_rollout_time == 1800.0
    assert cfg.tps_limit == 25
    assert cfg.gateway_port == 0


def test_config_from_args_overrides():
    cfg = rollout.AgentCoreRLConfig.from_args(_args(tps_limit=10, gateway_port=9090))
    assert cfg.tps_limit == 10
    assert cfg.gateway_port == 9090


def test_config_from_args_missing_backend_raises():
    args = _args()
    del args.rollout_session_backend
    with pytest.raises(ValueError, match="rollout_session_backend"):
        rollout.AgentCoreRLConfig.from_args(args)


# ---------------------------------------------------------------------------
# _build_payload
# ---------------------------------------------------------------------------


def test_build_payload_returns_metadata_payload(make_sample):
    sample = make_sample(metadata={"payload": {"prompt": "hi", "answer": "42"}})
    assert rollout._build_payload(sample) == {"prompt": "hi", "answer": "42"}


def test_build_payload_missing_payload_key_raises(make_sample):
    sample = make_sample(metadata={"other": "stuff"})
    with pytest.raises(ValueError, match="payload"):
        rollout._build_payload(sample)


def test_build_payload_non_dict_payload_raises(make_sample):
    sample = make_sample(metadata={"payload": "not-a-dict"})
    with pytest.raises(ValueError, match="payload"):
        rollout._build_payload(sample)


def test_build_payload_reserved_config_raises(make_sample):
    sample = make_sample(metadata={"payload": {"prompt": "hi", "_config": {}}})
    with pytest.raises(ValueError, match="_config"):
        rollout._build_payload(sample)


def test_build_payload_none_metadata_raises():
    sample = SimpleNamespace(metadata=None)
    with pytest.raises(ValueError, match="payload"):
        rollout._build_payload(sample)


# ---------------------------------------------------------------------------
# _agent_reward
# ---------------------------------------------------------------------------


def test_agent_reward_float():
    assert rollout._agent_reward({"reward": 1.0}) == 1.0


def test_agent_reward_int():
    assert rollout._agent_reward({"reward": 1}) == 1.0


def test_agent_reward_absent_returns_none():
    assert rollout._agent_reward({}) is None


def test_agent_reward_none_returns_none():
    assert rollout._agent_reward({"reward": None}) is None


def test_agent_reward_bool_raises():
    with pytest.raises(ValueError, match="Non-numeric"):
        rollout._agent_reward({"reward": True})


def test_agent_reward_string_raises():
    with pytest.raises(ValueError, match="Non-numeric"):
        rollout._agent_reward({"reward": "good"})


# ---------------------------------------------------------------------------
# _run_agent
# ---------------------------------------------------------------------------


class _Handle:
    def __init__(self, outcome):
        self._outcome = outcome

    async def result(self):
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        if callable(self._outcome):
            return await self._outcome()
        return self._outcome


class _FakeClient:
    """Stands in for AgentCoreHttpClient: records invocations and stopped sessions."""

    runtime_arn = _ARN

    def __init__(self, outcome):
        self._outcome = outcome
        self.invocations = []
        self.stopped = []

    async def invoke(self, payload, *, session_id, invocation_id=None, background=False):
        self.invocations.append((payload, session_id, background))
        return _Handle(self._outcome)

    async def stop_session(self, session_id):
        self.stopped.append(session_id)


class _NoWait:
    async def wait_async(self):
        pass


@pytest.fixture(autouse=True)
def _sessions(monkeypatch):
    """Route make_session to an AgentCoreHttpSession over the test's fake client."""
    made = []

    def make_session(sid, cfg, state):
        made.append((sid, cfg, state["session_id"]))
        return AgentCoreHttpSession.with_client(sid, cfg["client"], state)

    monkeypatch.setattr(rollout, "make_session", make_session)
    return made


def _service(client, max_rollout_time=5.0):
    return SimpleNamespace(
        session_rate_limiter=_NoWait(),
        config=SimpleNamespace(max_rollout_time=max_rollout_time, rollout_session_backend={"client": client}),
    )


def test_real_factory_builds_http_session():
    from agentcore_rl_toolkit.rollout_session.factory import make_session

    state = rollout.PersistentDict({"session_id": "sid-1"}, persister=rollout.NullPersister())
    session = make_session("sid-1", _BACKEND, state)
    assert isinstance(session, AgentCoreHttpSession)
    assert session._client.runtime_arn == _ARN


@pytest.mark.asyncio
async def test_run_agent_bad_backend_config_raises(monkeypatch):
    from agentcore_rl_toolkit.rollout_session import factory

    monkeypatch.setattr(rollout, "make_session", factory.make_session)
    service = SimpleNamespace(
        session_rate_limiter=_NoWait(),
        config=SimpleNamespace(max_rollout_time=5.0, rollout_session_backend={**_BACKEND, "bogus": 1}),
    )
    with pytest.raises(TypeError, match="bogus"):
        await rollout._run_agent(service, "sid-1", {})


@pytest.mark.asyncio
async def test_run_agent_returns_result_and_stops_session(_sessions):
    client = _FakeClient({"reward": 1.0})
    task = {"prompt": "hi", "_config": {"api_key": "sid-1"}}
    result, error = await rollout._run_agent(_service(client), "sid-1", task)
    assert (result, error) == ({"reward": 1.0}, None)
    assert _sessions == [("sid-1", {"client": client}, "sid-1")]
    assert client.invocations == [(task, "sid-1", True)]
    assert client.stopped == ["sid-1"]


@pytest.mark.asyncio
async def test_run_agent_invocation_error_is_reported():
    snapshot = {"invocation_id": "inv", "status": "failed", "error": "boom"}
    client = _FakeClient(InvocationError("sid-1", snapshot))
    result, error = await rollout._run_agent(_service(client), "sid-1", {})
    assert result == {}
    assert "boom" in error
    assert client.stopped == ["sid-1"]


@pytest.mark.asyncio
async def test_run_agent_timeout_stops_session():
    async def never():
        await asyncio.sleep(3600)

    client = _FakeClient(never)
    result, error = await rollout._run_agent(_service(client, max_rollout_time=0.01), "sid-1", {})
    assert result == {}
    assert "timed out" in error
    assert client.stopped == ["sid-1"]


@pytest.mark.asyncio
async def test_run_agent_transport_error_is_reported():
    client = _FakeClient(ConnectionError("reset"))
    _, error = await rollout._run_agent(_service(client), "sid-1", {})
    assert error == "ConnectionError: reset"


@pytest.mark.asyncio
async def test_run_agent_non_dict_result_raises():
    client = _FakeClient(["not", "a", "dict"])
    with pytest.raises(RolloutContractError):
        await rollout._run_agent(_service(client), "sid-1", {})
    assert client.stopped == ["sid-1"]


# ---------------------------------------------------------------------------
# _aborted
# ---------------------------------------------------------------------------


def test_aborted_sets_remove_sample(make_sample):
    sample = make_sample(index=5)
    out = rollout._aborted(sample, "timeout")
    assert out.remove_sample is True


def test_aborted_sets_status_aborted(make_sample):
    from .conftest import FakeStatus

    sample = make_sample(index=3)
    out = rollout._aborted(sample, "some reason")
    assert out.status == FakeStatus.ABORTED


def test_aborted_injects_abort_reason(make_sample):
    sample = make_sample(index=1, metadata={"payload": {}})
    out = rollout._aborted(sample, "network error")
    assert out.metadata["abort_reason"] == "network error"


def test_aborted_preserves_existing_metadata(make_sample):
    sample = make_sample(index=2, metadata={"payload": {"prompt": "hi"}})
    out = rollout._aborted(sample, "oops")
    assert out.metadata["payload"] == {"prompt": "hi"}
    assert "abort_reason" in out.metadata


def test_aborted_sets_inert_tokens(make_sample):
    sample = make_sample(index=0)
    out = rollout._aborted(sample, "reason")
    assert out.tokens == [0, 0]
    assert out.loss_mask == [0]
    assert out.response_length == 1
    assert out.reward == 0.0


# ---------------------------------------------------------------------------
# _to_sample
# ---------------------------------------------------------------------------


def test_to_sample_maps_record_fields(make_sample):
    base = make_sample(
        index=7, group_index=2, prompt="question", label="answer", session_id="sid-1", metadata={"k": "v"}
    )
    record = TraceRecord(
        token_ids=(10, 20, 30),
        loss_mask=(1, 1, 0),
        logprobs=(-0.1, -0.2, -0.3),
        response="hello",
        response_length=2,
        status=TraceStatus.COMPLETED,
    )
    out = rollout._to_sample(base, record, reward=0.8)

    assert out.index == 7
    assert out.group_index == 2
    assert out.rollout_id == 7
    assert out.tokens == [10, 20, 30]
    assert out.loss_mask == [1, 1, 0]
    assert out.rollout_log_probs == pytest.approx([-0.1, -0.2, -0.3])
    assert out.reward == pytest.approx(0.8)
    assert out.response == "hello"
    assert out.metadata["acr_session_id"] == "sid-1"
    assert out.metadata["k"] == "v"
