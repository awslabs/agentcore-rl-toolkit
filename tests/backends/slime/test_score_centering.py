"""slime's score centering over gateway rollouts.

The rollout hook is tested against fakes (see conftest.py). The Sample conversion runs
against slime's real ``Sample`` and validators, so those tests skip without slime.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from agentcore_rl_toolkit.backends.slime.integration import rollout
from agentcore_rl_toolkit.rollout_gateway import TopLogprobs, TraceRecord

K = 3


def _args(**kwargs):
    defaults = dict(
        use_score_centering=True,
        score_centering_top_k=K,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_temperature=1.0,
    )
    return Namespace(**{**defaults, **kwargs})


def _head(*tokens):
    """A distinct, descending K-wide head per position: ids [t, t+1, t+2]."""
    ids = np.asarray([[t, t + 1, t + 2] for t in tokens], dtype=np.int32)
    logprobs = np.tile(np.log(np.asarray([0.5, 0.3, 0.1], dtype=np.float32)), (len(tokens), 1))
    return TopLogprobs(ids, logprobs)


def _record(heads, *, loss_mask, response=(20, 21, 22, 23, 24)):
    """Prompt [1, 2] plus ``response``; trained tokens carry logprob -0.5."""
    return TraceRecord(
        token_ids=[1, 2, *response],
        loss_mask=list(loss_mask),
        logprobs=[-0.5 if m else 0.0 for m in loss_mask],
        response_length=len(response),
        top_logprobs=heads,
    )


# ---------------------------------------------------------------------------
# Rollout hook
# ---------------------------------------------------------------------------


def test_top_k_is_zero_without_score_centering():
    assert rollout._score_centering_top_k(Namespace()) == 0
    assert rollout._score_centering_top_k(_args(use_score_centering=False)) == 0
    assert rollout._score_centering_top_k(_args()) == K


def test_exact_top_p_mode_is_rejected():
    with pytest.raises(ValueError, match="--rollout-top-p 1"):
        rollout._score_centering_top_k(_args(rollout_top_p=0.9))


class _FakeGateway:
    def __init__(self, records):
        self.records = records
        self.sessions = {}

    def create_session(self, sid, **kwargs):
        self.sessions[sid] = kwargs

    async def finish_session(self, sid, base_sample):
        return self.records

    async def drop_session(self, sid):
        pass


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.asyncio
async def test_generate_requests_sampler_heads_with_score_centering(monkeypatch, make_sample, enabled):
    service = SimpleNamespace(gateway=_FakeGateway([_record([], loss_mask=[0] * 5)]), base_url="u", model_id="m")
    monkeypatch.setattr(rollout, "AgentCoreRLService", lambda args: service)

    async def run_agent(service, sid, task):
        return {"reward": 1.0}, None

    monkeypatch.setattr(rollout, "_run_agent", run_agent)
    checked = []
    score_centering = ModuleType("slime.utils.score_centering")
    score_centering.score_centering_request = lambda args, params: checked.append((args, params))
    monkeypatch.setitem(sys.modules, "slime.utils.score_centering", score_centering)
    appended = []
    monkeypatch.setattr(rollout, "_append_sampler_heads", lambda *a: appended.append(a))

    args = _args(use_score_centering=enabled)
    await rollout.generate(args, make_sample(index=0, metadata={"payload": {"q": 1}}), {"temperature": 1.0})

    (session,) = service.gateway.sessions.values()
    assert session["num_top_logprobs"] == (K if enabled else 0)
    assert len(appended) == int(enabled)
    if enabled:
        session["validate_sampling_params"]({"temperature": 1.0})
        assert checked == [(args, {"temperature": 1.0})]
    else:
        assert session["validate_sampling_params"] is None


# ---------------------------------------------------------------------------
# Sample conversion, against slime's real Sample
# ---------------------------------------------------------------------------


@pytest.fixture
def slime_sample(monkeypatch):
    """slime's real ``Sample`` in place of the conftest fake, for this test only."""
    try:
        importlib.metadata.distribution("slime")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("slime is not installed")
    for name in [name for name in sys.modules if name == "slime" or name.startswith("slime.")]:
        monkeypatch.delitem(sys.modules, name)
    types = importlib.import_module("slime.utils.types")
    monkeypatch.setattr(rollout, "Sample", types.Sample)
    monkeypatch.setattr(rollout, "_STATUS", {s.value: s for s in types.Sample.Status})
    return types.Sample


def _base(Sample):
    return Sample(index=0, group_index=0, prompt="q", metadata={})


def test_trained_segments_carry_their_heads(slime_sample):
    from slime.utils.score_centering import validate_sampler_topk

    # response [20, 21, 22, 23, 24]: model 20, 21, tool output 22, model 23, 24
    record = _record([(0, _head(20, 21)), (3, _head(23, 24))], loss_mask=[1, 1, 0, 1, 1])
    out = rollout._to_sample(_base(slime_sample), record, 1.0, _args(), K)

    assert out.tokens == record.token_ids
    assert out.response_length == 5
    assert out.loss_mask == [1, 1, 0, 1, 1]
    assert out.rollout_log_probs == [-0.5, -0.5, 0.0, -0.5, -0.5]
    ids = np.asarray(out.rollout_topk_token_ids)
    assert ids[[0, 1, 3, 4], 0].tolist() == [20, 21, 23, 24]
    assert ids[2].tolist() == [0, 1, 2]  # slime's placeholder for the masked tool token
    assert np.asarray(out.rollout_topk_log_probs).shape == (5, K)
    validate_sampler_topk(out, K)


def test_trailing_masked_tokens_get_placeholders(slime_sample):
    record = _record([(1, _head(21))], loss_mask=[0, 1, 0, 0, 0])
    out = rollout._to_sample(_base(slime_sample), record, 1.0, _args(), K)
    assert out.loss_mask == [0, 1, 0, 0, 0]
    assert np.asarray(out.rollout_topk_token_ids).shape == (5, K)


def test_trained_tokens_without_a_head_are_rejected(slime_sample):
    record = _record([(0, _head(20, 21))], loss_mask=[1, 1, 0, 1, 0])
    with pytest.raises(RuntimeError, match="without a sampler head"):
        rollout._to_sample(_base(slime_sample), record, 1.0, _args(), K)


def test_heads_of_the_wrong_width_are_rejected(slime_sample):
    record = _record([(0, _head(20))], loss_mask=[1, 0, 0, 0, 0])
    with pytest.raises(RuntimeError, match="expected"):
        rollout._to_sample(_base(slime_sample), record, 1.0, _args(score_centering_top_k=4), 4)


def test_aborted_sample_has_placeholder_heads(slime_sample):
    from slime.utils.score_centering import validate_sampler_topk

    out = rollout._aborted(_base(slime_sample), "reason", _args(), K)
    assert out.remove_sample is True
    assert out.loss_mask == [0]
    validate_sampler_topk(out, K)


def test_agent_overrides_that_change_the_distribution_are_rejected(slime_sample):
    validate = rollout._score_centering_validator(_args())
    validate({"temperature": 1.0, "top_p": 1.0, "top_k": -1})
    for override in ({"temperature": 0.7}, {"top_p": 0.9}, {"top_k": 20}):
        with pytest.raises(ValueError, match="Score centering"):
            validate({"temperature": 1.0, "top_p": 1.0, "top_k": -1, **override})
