"""Unit tests for the experimental slime rollout helpers.

Covers AgentCoreRLConfig.from_args, _build_payload, _agent_reward,
_aborted, and _to_sample.  Slime is replaced by a fake module (see
conftest.py), so these tests run without the full training stack.
"""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest

from agentcore_rl_toolkit.backends.experimental.slime.integration import rollout
from agentcore_rl_toolkit.rollout_gateway.trace import Status as TraceStatus
from agentcore_rl_toolkit.rollout_gateway.trace import TraceRecord

# ---------------------------------------------------------------------------
# AgentCoreRLConfig.from_args
# ---------------------------------------------------------------------------


def _args(**kwargs):
    defaults = dict(
        agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/test",
        s3_bucket="my-bucket",
        gateway_host=None,
        exp_id=None,
        model_id=None,
        max_rollout_time=1800.0,
        tps_limit=25,
        max_pool_connections=100,
        gateway_port=0,
        hf_checkpoint="Qwen/Qwen2.5-0.5B",
        wandb_group=None,
        wandb_project=None,
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


def test_config_from_args_required_fields():
    cfg = rollout.AgentCoreRLConfig.from_args(_args())
    assert cfg.agent_runtime_arn == "arn:aws:bedrock-agentcore:us-west-2:123:runtime/test"
    assert cfg.s3_bucket == "my-bucket"


def test_config_from_args_defaults():
    cfg = rollout.AgentCoreRLConfig.from_args(_args())
    assert cfg.max_rollout_time == 1800.0
    assert cfg.tps_limit == 25
    assert cfg.gateway_port == 0


def test_config_from_args_overrides():
    cfg = rollout.AgentCoreRLConfig.from_args(_args(tps_limit=10, gateway_port=9090))
    assert cfg.tps_limit == 10
    assert cfg.gateway_port == 9090


def test_config_from_args_missing_arn_raises():
    args = _args(agent_runtime_arn=None)
    del args.agent_runtime_arn
    with pytest.raises(ValueError, match="agent_runtime_arn"):
        rollout.AgentCoreRLConfig.from_args(args)


def test_config_from_args_missing_bucket_raises():
    args = _args(s3_bucket=None)
    del args.s3_bucket
    with pytest.raises(ValueError, match="s3_bucket"):
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


def test_build_payload_none_metadata_raises():
    sample = SimpleNamespace(metadata=None)
    with pytest.raises(ValueError, match="payload"):
        rollout._build_payload(sample)


# ---------------------------------------------------------------------------
# _agent_reward
# ---------------------------------------------------------------------------


def test_agent_reward_scalar():
    assert rollout._agent_reward({"rewards": 1.0}) == 1.0


def test_agent_reward_int():
    assert rollout._agent_reward({"rewards": 1}) == 1.0


def test_agent_reward_list_returns_last():
    assert rollout._agent_reward({"rewards": [0.2, 0.5, 0.9]}) == pytest.approx(0.9)


def test_agent_reward_absent_returns_none():
    assert rollout._agent_reward({}) is None


def test_agent_reward_none_returns_none():
    assert rollout._agent_reward({"rewards": None}) is None


def test_agent_reward_empty_list_returns_none():
    assert rollout._agent_reward({"rewards": []}) is None


def test_agent_reward_bool_raises():
    with pytest.raises(ValueError, match="Non-numeric"):
        rollout._agent_reward({"rewards": True})


def test_agent_reward_string_raises():
    with pytest.raises(ValueError, match="Non-numeric"):
        rollout._agent_reward({"rewards": "good"})


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
