"""``run_one_rollout`` unit tests: the ``_config`` payload sent to an AgentCoreRuntimeApp
agent, scalar reward extraction, and zero reward on agent failure or timeout."""

import asyncio

import pytest

from agentcore_rl_toolkit.backends.sagemaker import rollout
from agentcore_rl_toolkit.rollout_gateway import TraceRecord
from agentcore_rl_toolkit.rollout_session.wire import RolloutDumpResponse


class FakeGateway:
    def __init__(self):
        self.created = []
        self.finished = {}

    def create_session(self, sid, **kwargs):
        self.created.append(sid)

    async def finish_session(self, sid, *, base_sample, reward):
        self.finished[sid] = reward
        return [
            TraceRecord(token_ids=[1, 2], loss_mask=[1], logprobs=[-0.1]),
            TraceRecord(token_ids=[], loss_mask=[], logprobs=[]),
        ]


def fake_session(behavior):
    """An AgentCoreHttpSession stand-in whose ``run`` returns or raises per ``behavior``."""

    class FakeSession:
        tasks = []

        def __init__(self, session_id):
            self.session_id = session_id

        @classmethod
        def with_client(cls, session_id, client):
            return cls(session_id)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def run(self, task):
            FakeSession.tasks.append(task)
            return await behavior(task)

    return FakeSession


def run(monkeypatch, behavior, max_rollout_time=5.0):
    session = fake_session(behavior)
    monkeypatch.setattr(rollout, "AgentCoreHttpSession", session)
    gateway = FakeGateway()
    records, reward, result = asyncio.run(
        rollout.run_one_rollout(
            client=object(),
            gateway=gateway,
            payload={"prompt": "1 + 1?", "answer": "2"},
            base_url="http://gateway:9000/v1",
            model_id="model",
            max_rollout_time=max_rollout_time,
            sampling_defaults={},
            max_context_tokens=1024,
        )
    )
    return records, reward, result, gateway, session.tasks


def test_sends_config_and_returns_scalar_reward(monkeypatch):
    async def ok(task):
        return RolloutDumpResponse(task_output={"reward": 1.0}, reward=1.0, exception=None)

    records, reward, result, gateway, tasks = run(monkeypatch, ok)

    (sid,) = gateway.created
    assert tasks == [
        {
            "prompt": "1 + 1?",
            "answer": "2",
            "_config": {"base_url": "http://gateway:9000/v1", "model_id": "model", "api_key": sid},
        }
    ]
    assert reward == 1.0 and result == {"reward": 1.0}
    assert gateway.finished == {sid: 1.0}
    assert len(records) == 1  # the empty record is dropped


@pytest.mark.parametrize("output", [{}, {"reward": "bad"}, {"rewards": [1.0]}])
def test_missing_or_non_numeric_reward_scores_zero(monkeypatch, output):
    async def ok(task):
        return RolloutDumpResponse(task_output=output, reward=None, exception=None)

    _, reward, _, _, _ = run(monkeypatch, ok)

    assert reward == 0.0


def test_agent_exception_scores_zero(monkeypatch):
    async def failed(task):
        return RolloutDumpResponse(task_output=None, reward=None, exception="Traceback: boom")

    _, reward, result, gateway, _ = run(monkeypatch, failed)

    assert reward == 0.0 and result == {}
    assert list(gateway.finished.values()) == [0.0]


def test_timeout_scores_zero(monkeypatch):
    async def slow(task):
        await asyncio.sleep(10)

    _, reward, _, _, _ = run(monkeypatch, slow, max_rollout_time=0.05)

    assert reward == 0.0
