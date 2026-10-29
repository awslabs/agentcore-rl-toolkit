"""Score function centering: draws parsing, logits-processor math, per-row alignment of
the agent loop's ``rollout_sample_ids``, and the batch-padding fix."""

import numpy as np
import pytest
import torch
from verl.workers.rollout.replica import TokenOutput

from agentcore_rl_toolkit.backends.verl import score_centering as sc
from agentcore_rl_toolkit.backends.verl.agent_loop import _TokenSamplesOutput
from agentcore_rl_toolkit.rollout_gateway.sampling_backends.vllm_token_samples import sample_rows

from .conftest import FakeLLMServerClient, make_trainer_config
from .test_agent_loop import _make_loop, _wire_result, mock_http_client  # noqa: F401 - autouse fixture


class _Flat:
    """The FlatLogprobs fields sample_rows reads."""

    def __init__(self, rows):
        self.start_indices, self.end_indices, self.token_ids, self.logprobs = [], [], [], []
        for row in rows:
            self.start_indices.append(len(self.token_ids))
            self.token_ids += [t for t, _ in row]
            self.logprobs += [lp for _, lp in row]
            self.end_indices.append(len(self.token_ids))


def test_sample_rows_drop_sampled_token_and_keep_duplicates():
    flat = _Flat([[(5, -0.1), (5, -0.1), (7, -2.0)], [(9, -0.5), (3, -1.0), (3, -1.0)]])
    assert sample_rows(flat, 2).tolist() == [[5, 7], [3, 3]]


def test_sample_rows_reject_short_rows():
    with pytest.raises(RuntimeError, match="exactly 2 token samples"):
        sample_rows(_Flat([[(5, -0.1), (5, -0.1)]]), 2)


def _packed(sequences, draws):
    input_ids = torch.nested.nested_tensor([torch.tensor(s) for s in sequences], layout=torch.jagged)
    sample_ids = torch.nested.nested_tensor([torch.tensor(d) for d in draws], layout=torch.jagged)
    return {"input_ids": input_ids, sc.SAMPLE_IDS_KEY: sample_ids}


def test_score_is_sampled_logit_minus_mean_draw_logit():
    # two packed sequences; row t holds the draws for token t + 1
    data = _packed([[1, 2, 3], [4, 0]], [[[0, 0], [3, 1], [0, 0]], [[2, 2], [0, 0]]])
    logits = torch.randn(1, 5, 6, requires_grad=True)

    (score,) = sc._score_from_logits(logits, data).values()

    z = logits[0]
    assert score.shape == (1, 5)
    assert score[0, 1].item() == pytest.approx((z[1, 3] - (z[1, 3] + z[1, 1]) / 2).item())
    assert score[0, 3].item() == pytest.approx((z[3, 0] - z[3, 2]).item())
    # gradient of c at row 1: +1 on the sampled token 3, -1/K on each draw (3 and 1)
    score[0, 1].backward()
    grad = logits.grad[0, 1]
    assert grad.tolist() == pytest.approx([0.0, -0.5, 0.0, 0.5, 0.0, 0.0])
    assert torch.count_nonzero(logits.grad[0, [0, 2, 3, 4]]) == 0


def test_score_pads_draws_to_bucketed_logits():
    data = _packed([[1, 2]], [[[2], [0]]])
    (score,) = sc._score_from_logits(torch.randn(1, 4, 3), data).values()
    assert score.shape == (1, 4)


def _sampled_turn(token_ids, k=2, version=3):
    """A generated turn whose draws for token t are [t + 100] * k, sampled under ``version``."""
    return TokenOutput(
        token_ids=token_ids,
        log_probs=[-0.5] * len(token_ids),
        stop_reason="completed",
        extra_fields={
            "response_sample_ids": np.array([[t + 100] * k for t in token_ids], dtype=np.int32),
            "response_sample_versions": np.full(len(token_ids), version),
        },
    )


@pytest.mark.asyncio
async def test_agent_loop_aligns_draws_to_logits_rows_across_turns():
    llm = FakeLLMServerClient([_sampled_turn([101, 102], version=3), _sampled_turn([101, 102], version=4)])
    loop = _make_loop(llm, trainer_config=make_trainer_config(score_centering={"num_auxiliary_token_samples": 2}))
    _wire_result(loop, {"reward": 1.0}, drive_turns=2)

    (out,) = await loop.run({"temperature": 1.0}, raw_prompt=[{"role": "user", "content": "hi"}], payload={})

    assert [call["sampling_params"]["num_auxiliary_token_samples"] for call in llm.calls] == [2, 2]
    assert isinstance(out, _TokenSamplesOutput)
    fields = out.as_dict()
    draws, versions = fields[sc.SAMPLE_IDS_KEY], fields[sc.SAMPLE_VERSIONS_KEY]
    sequence = torch.cat([fields["prompts"], fields["responses"]]).tolist()
    assert draws.shape == (len(sequence), 2)
    assert versions.shape == (len(sequence),)
    trained = torch.cat([torch.zeros_like(fields["prompts"]), fields["response_mask"]]).bool().tolist()
    first_turn_end = len(fields["prompts"]) + 2
    for t, is_trained in enumerate(trained):
        # row t - 1 predicts token t: its draws come from token t's own distribution
        expected = [sequence[t] + 100] * 2 if is_trained else [0, 0]
        assert draws[t - 1].tolist() == expected, t
        expected_version = (3 if t < first_turn_end else 4) if is_trained else -1
        assert versions[t - 1].item() == expected_version, t


@pytest.mark.asyncio
async def test_agent_loop_requires_draws_for_trained_tokens():
    loop = _make_loop(
        FakeLLMServerClient(), trainer_config=make_trainer_config(score_centering={"num_auxiliary_token_samples": 2})
    )
    _wire_result(loop, {"reward": 1.0})
    with pytest.raises(RuntimeError, match="trained tokens without sampler draws"):
        await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={})


def test_padding_template_resizes_draws(monkeypatch):
    from verl.trainer.ppo import padding_utils

    monkeypatch.setattr(
        padding_utils, "construct_minimal_padding_template", padding_utils.construct_minimal_padding_template
    )
    sc.install_padding_patch()
    source = {
        "prompts": torch.tensor([1, 2]),
        "responses": torch.tensor([3]),
        "input_ids": torch.tensor([1, 2, 3]),
        "position_ids": torch.arange(3),
        sc.SAMPLE_IDS_KEY: torch.ones((3, 4), dtype=torch.int32),
    }
    source[sc.SAMPLE_VERSIONS_KEY] = torch.full((3,), 7)
    template, _ = padding_utils.construct_minimal_padding_template(source, {}, eos_token_id=0)
    assert template[sc.SAMPLE_IDS_KEY].shape == (2, 4)
    assert template[sc.SAMPLE_IDS_KEY].dtype == torch.int32
    assert template[sc.SAMPLE_VERSIONS_KEY].tolist() == [-1, -1]


@pytest.mark.asyncio
async def test_async_client_concatenates_resumed_segments(monkeypatch):
    segments = iter([_sampled_turn([5, 6], version=1), _sampled_turn([7], version=2)])

    async def base_generate(self, *args, **kwargs):
        return next(segments)

    async def async_generate(self, *args, **kwargs):
        # verl's resume loop: two segments, token ids concatenated, extra fields dropped
        first = await sc._client_generate(self)
        second = await sc._client_generate(self)
        return TokenOutput(token_ids=first.token_ids + second.token_ids, stop_reason="completed", extra_fields={})

    originals = {"LLMServerClient.generate": base_generate, "FullyAsyncLLMServerClient.generate": async_generate}
    monkeypatch.setattr(sc, "_ORIGINAL", originals)

    output = await sc._async_client_generate(object())

    assert output.token_ids == [5, 6, 7]
    assert output.extra_fields["response_sample_ids"].tolist() == [[105, 105], [106, 106], [107, 107]]
    assert output.extra_fields["response_sample_versions"].tolist() == [1, 1, 2]


def test_token_ages_in_optimizer_steps():
    from tensordict import NonTensorData, TensorDict

    # prompt [1, 2], response [3, 4, 5]: rows 1..3 predict the response tokens
    td = TensorDict(
        {
            "prompts": torch.nested.nested_tensor([torch.tensor([1, 2])], layout=torch.jagged),
            "responses": torch.nested.nested_tensor([torch.tensor([3, 4, 5])], layout=torch.jagged),
            sc.SAMPLE_VERSIONS_KEY: torch.nested.nested_tensor([torch.tensor([-1, 2, 1, 0, -1])], layout=torch.jagged),
        },
        batch_size=[1],
    )
    td[sc.TRAINER_STEP_KEY] = NonTensorData(9)  # (g - 1) * P + k with P = 4
    td[sc.SYNC_STEP_KEY] = NonTensorData(4)

    assert sc._token_ages(td, torch.tensor([[True, True, True]])).tolist() == [[1.0, 5.0, 9.0]]


def test_loss_gradients():
    from types import SimpleNamespace

    from tensordict import NonTensorData, TensorDict

    def nested(*rows, **kw):
        return torch.nested.nested_tensor([torch.tensor(r, **kw) for r in rows], layout=torch.jagged)

    # prompt [1, 2], response [3, 4, 5] with the middle token masked; rows 1..3 predict the response
    data = TensorDict(
        {
            "prompts": nested([1, 2]),
            "responses": nested([3, 4, 5]),
            "response_mask": nested([1, 0, 1]),
            "advantages": nested([0.5, 0.5, 0.5]),
            "old_log_probs": nested([-1.0, -1.0, -1.0]),
            sc.SAMPLE_VERSIONS_KEY: nested([-1, 0, 0, 0, -1]),
        },
        batch_size=[1],
    )
    for key, value in {"dp_size": 1, "batch_num_tokens": None, "global_batch_size": 1}.items():
        data[key] = NonTensorData(value)
    data[sc.KL_BETA_KEY] = NonTensorData(0.3)
    data[sc.TRAINER_STEP_KEY] = NonTensorData(4)  # every token is 4 optimizer steps old
    data[sc.SYNC_STEP_KEY] = NonTensorData(1)
    score = torch.zeros(5, requires_grad=True)
    log_prob = torch.full((5,), -1.0, requires_grad=True)
    model_output = {
        "log_probs": torch.nested.nested_tensor_from_jagged(log_prob, torch.tensor([0, 5])),
        "sc_score": torch.nested.nested_tensor_from_jagged(score, torch.tensor([0, 5])),
    }
    config = SimpleNamespace(
        global_batch_info={},
        loss_scale_factor=None,
        loss_agg_mode="seq-mean-token-sum",
        entropy_coeff=0.0,
        use_kl_loss=False,
    )

    loss, metrics = sc.score_centering_loss(config, model_output, data)
    loss.backward()

    beta = 0.3 / 4**0.5
    # reward term: -A on trained rows; KL term: +beta on c and -beta on log p(a)
    assert score.grad.tolist() == pytest.approx([0.0, -0.5 + beta, 0.0, -0.5 + beta, 0.0])
    assert log_prob.grad.tolist() == pytest.approx([0.0, -beta, 0.0, -beta, 0.0])
    assert {"actor/ppo_kl", "actor/sfc_centered_score", "actor/sfc_kl_loss", "actor/sfc_token_age"} <= set(metrics)
