"""Unit tests for score function centering on slime.

slime and Megatron are replaced by fakes (see conftest.py, and the cp=1 stand-ins for
slime's loss helpers below), so these tests run without the training stack.
"""

from __future__ import annotations

import argparse
import math
import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from agentcore_rl_toolkit.backends.slime.integration import rollout, score_centering
from agentcore_rl_toolkit.rollout_gateway import AuxiliaryTokenSamples, TraceRecord
from agentcore_rl_toolkit.rollout_gateway.sampling_backends.sglang_token_samples import SAMPLES_ENV

# ---------------------------------------------------------------------------
# Settings and validation
# ---------------------------------------------------------------------------


def _args(**kwargs):
    defaults = dict(
        sfc_num_auxiliary_token_samples=4,
        sfc_kl_beta=0.3,
        sfc_loss_agg_mode="token-mean",
        loss_type="custom_loss",
        custom_loss_function_path=score_centering.LOSS_PATH,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_temperature=1.0,
        allgather_cp=False,
        use_tis=False,
        get_mismatch_metrics=False,
        use_opsm=False,
        entropy_coef=0.0,
        use_rollout_logprobs=False,
        use_kl_loss=False,
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


def test_sfc_settings_default_off():
    assert score_centering.sfc_settings(Namespace()) == (0, 0.0)


def test_sfc_settings_reads_the_flags():
    assert score_centering.sfc_settings(_args()) == (4, 0.3)


def test_flags_parse_into_settings():
    parser = score_centering.add_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--sfc-num-auxiliary-token-samples", "8", "--sfc-kl-beta", "0.3"])
    assert score_centering.sfc_settings(args) == (8, 0.3)
    assert args.sfc_loss_agg_mode == "token-mean"
    with pytest.raises(SystemExit):
        parser.parse_args(["--sfc-loss-agg-mode", "seq-mean-token-sum"])


def test_sfc_settings_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        score_centering.sfc_settings(_args(sfc_kl_beta=-0.1))


def test_validate_accepts_a_configured_run(monkeypatch):
    monkeypatch.setenv(SAMPLES_ENV, "1")
    score_centering.validate(_args(), rollout=True)


def test_validate_lists_every_unmet_requirement(monkeypatch):
    monkeypatch.delenv(SAMPLES_ENV, raising=False)
    with pytest.raises(ValueError) as excinfo:
        score_centering.validate(_args(rollout_top_p=0.9, loss_type="policy_loss"), rollout=True)
    for needle in (SAMPLES_ENV, "--rollout-top-p", "--loss-type"):
        assert needle in str(excinfo.value)


# ---------------------------------------------------------------------------
# Rollout side: draws into Sample.multimodal_train_inputs
# ---------------------------------------------------------------------------


def _record(loss_mask, samples):
    n = len(loss_mask)
    return TraceRecord(
        token_ids=[1] * (n + 2),
        loss_mask=list(loss_mask),
        logprobs=[0.0] * n,
        response_length=n,
        auxiliary_token_samples=samples,
    )


def test_train_inputs_places_draws_at_their_response_offsets():
    record = _record(
        [1, 1, 0, 1],
        [(0, AuxiliaryTokenSamples([[4, 5], [6, 7]], [2, 2])), (3, AuxiliaryTokenSamples([[8, 9]], [3]))],
    )
    inputs = score_centering.train_inputs(record, 2, require_versions=True)
    assert inputs[score_centering.SAMPLE_IDS_KEY].tolist() == [[4, 5], [6, 7], [0, 0], [8, 9]]
    assert inputs[score_centering.SAMPLE_VERSIONS_KEY].tolist() == [2, 2, -1, 3]


def test_train_inputs_rejects_trained_tokens_without_draws():
    record = _record([1, 1], [(0, AuxiliaryTokenSamples([[4, 5]]))])
    with pytest.raises(RuntimeError, match="without sampler draws"):
        score_centering.train_inputs(record, 2, require_versions=False)


def test_train_inputs_needs_versions_only_for_the_kl():
    record = _record([1], [(0, AuxiliaryTokenSamples([[4, 5]]))])
    assert score_centering.train_inputs(record, 2, require_versions=False)[
        score_centering.SAMPLE_VERSIONS_KEY
    ].tolist() == [-1]
    with pytest.raises(RuntimeError, match="weight version"):
        score_centering.train_inputs(record, 2, require_versions=True)


def test_aborted_sample_carries_inert_draws(make_sample):
    out = rollout._aborted(make_sample(index=0), "reason", num_auxiliary_token_samples=3)
    assert out.multimodal_train_inputs[score_centering.SAMPLE_IDS_KEY].shape == (1, 3)
    assert out.multimodal_train_inputs[score_centering.SAMPLE_VERSIONS_KEY].tolist() == [-1]


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


@pytest.mark.parametrize("evaluation", [False, True])
@pytest.mark.asyncio
async def test_generate_requests_draws_for_training_rollouts_only(monkeypatch, make_sample, evaluation):
    record = _record([1], [(0, AuxiliaryTokenSamples([[4, 5]], [1]))])
    service = SimpleNamespace(
        gateway=_FakeGateway([record]),
        base_url="http://gw/v1",
        model_id="m",
        num_auxiliary_token_samples=2,
        require_sample_versions=True,
    )
    monkeypatch.setattr(rollout, "AgentCoreRLService", lambda args: service)

    async def run_agent(service, sid, task):
        return {"reward": 1.0}, None

    monkeypatch.setattr(rollout, "_run_agent", run_agent)
    sample = make_sample(index=0, metadata={"payload": {"q": 1}})
    (out,) = await rollout.generate(Namespace(), sample, {}, evaluation=evaluation)

    (session,) = service.gateway.sessions.values()
    assert session["num_auxiliary_token_samples"] == (0 if evaluation else 2)
    if evaluation:
        assert getattr(out, "multimodal_train_inputs", None) is None
    else:
        assert out.multimodal_train_inputs[score_centering.SAMPLE_IDS_KEY].tolist() == [[4, 5]]


# ---------------------------------------------------------------------------
# Train actor patches
# ---------------------------------------------------------------------------


@pytest.fixture
def train_state(monkeypatch):
    state = score_centering._TrainState()
    monkeypatch.setattr(score_centering, "_STATE", state)
    return state


def test_get_batch_moves_the_draws_out_of_the_forward_kwargs(monkeypatch):
    torch = pytest.importorskip("torch")
    ids = torch.arange(10, dtype=torch.int32).reshape(5, 2)
    versions = torch.tensor([1, 1, -1, 2, 2])

    def get_batch(data_iterator, keys, *args, **kwargs):
        return {
            "response_lengths": [3, 2],
            "multimodal_train_inputs": {
                score_centering.SAMPLE_IDS_KEY: ids,
                score_centering.SAMPLE_VERSIONS_KEY: versions,
            },
        }

    monkeypatch.setitem(score_centering._ORIGINAL, "get_batch", get_batch)
    batch = score_centering._get_batch(None, ["tokens"])
    assert batch["multimodal_train_inputs"] is None
    assert [t.tolist() for t in batch[score_centering.SAMPLE_IDS_KEY]] == [ids[:3].tolist(), ids[3:].tolist()]
    assert [t.tolist() for t in batch[score_centering.SAMPLE_VERSIONS_KEY]] == [[1, 1, -1], [2, 2]]


def test_step_counter_and_push_steps(monkeypatch, train_state):
    def train_one_step(*args, **kwargs):
        if kwargs.get("fail"):
            raise RuntimeError("boom")

    monkeypatch.setitem(score_centering._ORIGINAL, "train_one_step", train_one_step)

    class Updater:
        weight_version = 0

        def update_weights(self):
            self.weight_version += 1

    push = score_centering._record_push(Updater.update_weights)
    updater = Updater()
    push(updater)  # initial sync before any training
    score_centering._train_one_step()
    with pytest.raises(RuntimeError):
        score_centering._train_one_step(fail=True)
    push(updater)
    assert train_state.steps == 2
    assert train_state.push_steps == {1: 0, 2: 2}


# ---------------------------------------------------------------------------
# Loss, against cp=1 stand-ins for Megatron and slime's loss helpers
# ---------------------------------------------------------------------------


def _build_shifted_tokens(T, device, unconcat_tokens, total_lengths, response_lengths, allgather_cp):
    import torch

    full = torch.zeros(T, dtype=torch.long, device=device)
    offset = 0
    for tokens, total in zip(unconcat_tokens, total_lengths, strict=True):
        full[offset : offset + total - 1] = tokens[1:total]
        offset += total
    return full


def _extract_per_sample(full, entropy, total_lengths, response_lengths, allgather_cp):
    out, offset = [], 0
    for total, response in zip(total_lengths, response_lengths, strict=True):
        end = offset + total
        out.append(full[end - response - 1 : end - 1])
        offset = end
    return out, []


def _log_probs_and_entropy(logits, *, args, unconcat_tokens, total_lengths, response_lengths, with_entropy):
    import torch

    log_probs = torch.log_softmax(logits.squeeze(0) / args.rollout_temperature, dim=-1)
    targets = _build_shifted_tokens(log_probs.shape[0], None, unconcat_tokens, total_lengths, response_lengths, False)
    per_row = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)
    return None, {
        "log_probs": _extract_per_sample(per_row, None, total_lengths, response_lengths, False)[0],
        "entropy": _extract_per_sample(entropy, None, total_lengths, response_lengths, False)[0],
    }


@pytest.fixture
def fake_trainer(monkeypatch):
    pytest.importorskip("torch")
    tp_rank = SimpleNamespace(value=0)
    modules = {
        name: ModuleType(name)
        for name in (
            "megatron",
            "megatron.core",
            "megatron.core.tensor_parallel",
            "megatron.core.tensor_parallel.mappings",
            "slime.backends",
            "slime.backends.megatron_utils",
            "slime.backends.megatron_utils.loss",
            "slime.backends.megatron_utils.cp_utils",
            "slime.utils.ppo_utils",
        )
    }
    modules["megatron.core"].mpu = SimpleNamespace(get_tensor_model_parallel_rank=lambda: tp_rank.value)
    modules["megatron.core.tensor_parallel.mappings"].reduce_from_tensor_model_parallel_region = lambda x: x
    loss = modules["slime.backends.megatron_utils.loss"]
    loss._build_shifted_tokens = _build_shifted_tokens
    loss._extract_per_sample = _extract_per_sample
    loss.get_log_probs_and_entropy = _log_probs_and_entropy
    cp_utils = modules["slime.backends.megatron_utils.cp_utils"]
    cp_utils.slice_log_prob_with_cp = lambda x, total, response: x

    def get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, calculate_per_token_loss):
        assert calculate_per_token_loss
        return lambda x: sum((x_i * m).sum() for x_i, m in zip(x.split(response_lengths), loss_masks, strict=True))

    cp_utils.get_sum_of_sample_mean = get_sum_of_sample_mean
    modules["slime.utils.ppo_utils"].compute_approx_kl = lambda log_probs, ref, kl_loss_type: log_probs - ref
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return tp_rank


def _batch(torch, vocab=6, k=3, seed=0):
    """Two samples: prompt lengths 2 and 3, response lengths 3 and 2."""
    generator = torch.Generator().manual_seed(seed)
    prompts, responses = [2, 3], [3, 2]
    tokens = [torch.randint(vocab, (p + r,), generator=generator) for p, r in zip(prompts, responses, strict=True)]
    return {
        "unconcat_tokens": tokens,
        "total_lengths": [p + r for p, r in zip(prompts, responses, strict=True)],
        "response_lengths": responses,
        "loss_masks": [torch.tensor([1, 0, 1]), torch.tensor([1, 1])],
        "advantages": [torch.full((3,), 0.5), torch.full((2,), -1.0)],
        score_centering.SAMPLE_IDS_KEY: [torch.randint(vocab, (r, k), generator=generator) for r in responses],
        score_centering.SAMPLE_VERSIONS_KEY: [torch.tensor([1, -1, 1]), torch.tensor([2, 2])],
        "rollout_mask_sums": None,
    }


def _sum_of_sample_mean(batch):
    def reduce(x):
        return sum(
            (x_i * m).sum() / m.sum()
            for x_i, m in zip(x.split(batch["response_lengths"]), batch["loss_masks"], strict=True)
        )

    return reduce


def _expected_scores(torch, logits, batch, temperature):
    """c = log p(a) − mean_k log p(v_k), row by row."""
    log_probs = torch.log_softmax(logits.squeeze(0) / temperature, dim=-1)
    scores, offset = [], 0
    for tokens, total, response, draws in zip(
        batch["unconcat_tokens"],
        batch["total_lengths"],
        batch["response_lengths"],
        batch[score_centering.SAMPLE_IDS_KEY],
        strict=True,
    ):
        for r in range(response):
            row = log_probs[offset + total - response - 1 + r]
            scores.append(row[tokens[total - response + r]] - row[draws[r]].mean())
        offset += total
    return torch.stack(scores)


def test_centered_score_matches_log_prob_differences(fake_trainer):
    import torch

    batch = _batch(torch)
    logits = torch.randn(1, sum(batch["total_lengths"]), 6)
    args = _args(rollout_temperature=0.7)
    scores = torch.cat(score_centering._centered_scores(args, batch, logits))
    assert torch.allclose(scores, _expected_scores(torch, logits, batch, 0.7), atol=1e-5)


def test_centered_score_sums_over_vocab_shards(fake_trainer):
    import torch

    batch = _batch(torch)
    logits = torch.randn(1, sum(batch["total_lengths"]), 6)
    args = _args()
    full = torch.cat(score_centering._centered_scores(args, batch, logits))
    partials = []
    for rank, shard in enumerate(logits.chunk(2, dim=-1)):
        fake_trainer.value = rank
        partials.append(torch.cat(score_centering._centered_scores(args, batch, shard)))
    assert torch.allclose(partials[0] + partials[1], full, atol=1e-5)


def test_loss_gradient_is_the_centered_score_gradient(fake_trainer, train_state):
    import torch

    batch = _batch(torch)
    logits = torch.randn(1, sum(batch["total_lengths"]), 6, requires_grad=True)
    args = _args(sfc_num_auxiliary_token_samples=3, sfc_kl_beta=0.0)
    loss, metrics = score_centering.score_centering_loss(args, batch, logits, _sum_of_sample_mean(batch))
    assert loss.item() == pytest.approx(0.0, abs=1e-6)  # zero-valued surrogate
    assert "sfc_kl_loss" not in metrics
    loss.backward()

    reference = logits.detach().clone().requires_grad_(True)
    scores = _expected_scores(torch, reference, batch, 1.0)
    advantages = torch.cat(batch["advantages"])
    (-_sum_of_sample_mean(batch)(advantages * scores)).backward()
    assert torch.allclose(logits.grad, reference.grad, atol=1e-6)


def test_token_sum_drops_the_per_rollout_denominators(fake_trainer, train_state):
    import torch

    batch = _batch(torch)
    logits = torch.randn(1, sum(batch["total_lengths"]), 6, requires_grad=True)
    args = _args(sfc_num_auxiliary_token_samples=3, sfc_kl_beta=0.0, sfc_loss_agg_mode="token-sum")
    loss, _ = score_centering.score_centering_loss(args, batch, logits, _sum_of_sample_mean(batch))
    loss.backward()

    reference = logits.detach().clone().requires_grad_(True)
    scores = _expected_scores(torch, reference, batch, 1.0)
    mask = torch.cat(batch["loss_masks"])
    (-(torch.cat(batch["advantages"]) * scores * mask).sum()).backward()
    assert torch.allclose(logits.grad, reference.grad, atol=1e-6)


def test_kl_is_weighted_by_token_age(fake_trainer, train_state):
    import torch

    train_state.steps = 5
    train_state.push_steps.update({1: 1, 2: 5})  # version 1 is 4 steps old, version 2 is fresh
    batch = _batch(torch)
    logits = torch.randn(1, sum(batch["total_lengths"]), 6)
    args = _args(sfc_num_auxiliary_token_samples=3, sfc_kl_beta=0.4)
    _, metrics = score_centering.score_centering_loss(args, batch, logits, _sum_of_sample_mean(batch))

    log_probs = torch.cat(
        _log_probs_and_entropy(logits, args=args, **_loss_inputs(batch), with_entropy=False)[1]["log_probs"]
    )
    mean_draw_log_probs = log_probs - _expected_scores(torch, logits, batch, 1.0)
    ages = torch.tensor([4.0, 0.0, 4.0, 0.0, 0.0])
    weights = torch.where(ages >= 1, 0.4 / ages.clamp(min=1).sqrt(), torch.zeros(()))
    expected = _sum_of_sample_mean(batch)(-weights * mean_draw_log_probs)
    assert metrics["sfc_kl_loss"].item() == pytest.approx(expected.item(), rel=1e-5)
    assert metrics["sfc_token_age"].item() == pytest.approx((4 + 4) / 2, rel=1e-6)
    assert math.isfinite(metrics["loss"].item())


def test_kl_rejects_versions_without_a_recorded_push(fake_trainer, train_state):
    import torch

    train_state.push_steps[1] = 0
    batch = _batch(torch)
    logits = torch.randn(1, sum(batch["total_lengths"]), 6)
    args = _args(sfc_num_auxiliary_token_samples=3, sfc_kl_beta=0.4)
    with pytest.raises(RuntimeError, match=r"weight versions \[2\]"):
        score_centering.score_centering_loss(args, batch, logits, _sum_of_sample_mean(batch))


def _loss_inputs(batch):
    return {key: batch[key] for key in ("unconcat_tokens", "total_lengths", "response_lengths")}


def test_empty_inputs_shape():
    inputs = score_centering.empty_train_inputs(2, 3)
    assert inputs[score_centering.SAMPLE_IDS_KEY].dtype == np.int32
    assert inputs[score_centering.SAMPLE_IDS_KEY].shape == (2, 3)
