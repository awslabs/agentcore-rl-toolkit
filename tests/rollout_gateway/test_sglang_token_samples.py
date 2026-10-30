"""Token samples over SGLang: the ``SglangHttpBackend`` wire format and the sampler plugin."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from agentcore_rl_toolkit.rollout_gateway.sampling_backends import sglang_token_samples
from agentcore_rl_toolkit.rollout_gateway.sampling_backends.base import supports_auxiliary_token_samples
from agentcore_rl_toolkit.rollout_gateway.sampling_backends.sglang_http import SglangHttpBackend


def _meta(output_ids, rows, weight_version="3"):
    return {
        "output_token_logprobs": [[-0.5, t, None] for t in output_ids],
        "output_top_logprobs": [[[-1.0, t, None] for t in row] for row in rows],
        "finish_reason": {"type": "stop"},
        "weight_version": weight_version,
    }


async def _generate(meta, **kwargs):
    bodies = []

    async def handler(request):
        bodies.append(await request.json())
        return web.json_response({"meta_info": meta})

    app = web.Application()
    app.router.add_post("/generate", handler)
    async with TestServer(app) as server:
        backend = SglangHttpBackend(str(server.make_url("")))
        turn = await backend.generate(prompt_ids=[1, 2], sampling_params={"temperature": 1.0}, **kwargs)
    return turn, bodies


def test_backend_accepts_token_samples():
    assert supports_auxiliary_token_samples(SglangHttpBackend("http://x"))


@pytest.mark.asyncio
async def test_backend_requests_and_parses_draws():
    turn, (body,) = await _generate(_meta([7, 8], [[7, 4, 4], [8, 5, 6]]), num_auxiliary_token_samples=2)
    assert body["top_logprobs_num"] == 2
    assert turn.output_ids == [7, 8]
    assert turn.output_auxiliary_token_samples.ids == [[4, 4], [5, 6]]
    assert turn.output_auxiliary_token_samples.versions == [3, 3]


@pytest.mark.asyncio
async def test_backend_omits_versions_it_cannot_order():
    turn, _ = await _generate(_meta([7], [[7, 4]], weight_version="default"), num_auxiliary_token_samples=1)
    assert turn.output_auxiliary_token_samples.versions is None


@pytest.mark.asyncio
async def test_backend_without_draws_sends_no_top_logprobs():
    turn, (body,) = await _generate(_meta([7], [[7, 4]]))
    assert "top_logprobs_num" not in body
    assert turn.output_auxiliary_token_samples is None


@pytest.mark.parametrize(
    "rows",
    [
        [[7, 4], [8, 5]],  # top-K rows from a server without the plugin
        [[7, 4, 4], [8]],  # a truncated (top-p) position returns only the sampled token
        [[9, 4, 4], [8, 5, 6]],  # entry 0 is not the sampled token
        [[7, 4, 4]],  # one row short
    ],
)
@pytest.mark.asyncio
async def test_backend_rejects_rows_that_are_not_draws(rows):
    with pytest.raises(RuntimeError, match="AGENTCORE_SGLANG_TOKEN_SAMPLES=1"):
        await _generate(_meta([7, 8], rows), num_auxiliary_token_samples=2)


def test_register_is_a_no_op_when_disabled(monkeypatch):
    monkeypatch.delenv(sglang_token_samples.SAMPLES_ENV, raising=False)
    sglang_token_samples.register()  # would fail to import sglang if it patched


# ---------------------------------------------------------------------------
# Sampler patch, against a stand-in for sglang's Sampler
# ---------------------------------------------------------------------------

TOP_K_ALL = 1 << 30


@pytest.fixture
def sampler_cls(monkeypatch):
    pytest.importorskip("torch")

    class Sampler:
        def _attach_logprobs_to_output(
            self, logits_output, logprobs, top_logprobs_nums, token_ids_logprobs, sampling_info, batch_next_token_ids
        ):
            logprobs.clamp_(min=-1e30)
            logits_output.top_logprobs_nums_seen = list(top_logprobs_nums)
            logits_output.next_token_logprobs = logprobs[range(len(batch_next_token_ids)), batch_next_token_ids]

    modules = {
        "sglang": ModuleType("sglang"),
        "sglang.srt": ModuleType("sglang.srt"),
        "sglang.srt.layers": ModuleType("sglang.srt.layers"),
        "sglang.srt.layers.sampler": ModuleType("sglang.srt.layers.sampler"),
        "sglang.srt.sampling": ModuleType("sglang.srt.sampling"),
        "sglang.srt.sampling.sampling_params": ModuleType("sglang.srt.sampling.sampling_params"),
    }
    modules["sglang.srt.layers.sampler"].Sampler = Sampler
    modules["sglang.srt.layers"].sampler = modules["sglang.srt.layers.sampler"]
    modules["sglang.srt.sampling.sampling_params"].TOP_K_ALL = TOP_K_ALL
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    sglang_token_samples.install_sampler_patch()
    sglang_token_samples.install_sampler_patch()  # idempotent
    return Sampler


def _sampling_info(torch, top_ks, top_ps, min_ps):
    return SimpleNamespace(
        top_ks=torch.tensor(top_ks, dtype=torch.int32),
        top_ps=torch.tensor(top_ps),
        min_ps=torch.tensor(min_ps),
    )


def test_patch_draws_from_the_sampling_distribution(sampler_cls):
    import torch

    torch.manual_seed(0)
    probs = torch.tensor([[0.7, 0.2, 0.1, 0.0], [0.0, 0.0, 0.5, 0.5], [0.25, 0.25, 0.25, 0.25]])
    logprobs = probs.log()
    out = SimpleNamespace()
    k = 4000
    sampler_cls()._attach_logprobs_to_output(
        out,
        logprobs,
        [k, k, 0],
        [None] * 3,
        _sampling_info(torch, [TOP_K_ALL] * 3, [1.0] * 3, [0.0] * 3),
        torch.tensor([0, 3, 1]),
    )
    assert out.top_logprobs_nums_seen == [0, 0, 0]  # the original skips its top-k
    first, second, third = out.next_token_top_logprobs_idx
    assert first[0] == 0 and second[0] == 3  # the sampled token leads each row
    assert len(first) == len(second) == k + 1 and len(third) == 0
    frequencies = torch.bincount(first[1:], minlength=4).float() / k
    assert torch.allclose(frequencies, probs[0], atol=0.03)
    assert set(second[1:].tolist()) <= {2, 3}  # zero-probability tokens are never drawn
    assert torch.equal(out.next_token_top_logprobs_val[0], logprobs[0, first])


def test_patch_returns_only_the_sampled_token_for_truncated_rows(sampler_cls):
    import torch

    out = SimpleNamespace()
    sampler_cls()._attach_logprobs_to_output(
        out,
        torch.full((4, 3), 1 / 3).log(),
        [2, 2, 2, 2],
        [None] * 4,
        _sampling_info(torch, [TOP_K_ALL, 1, TOP_K_ALL, TOP_K_ALL], [1.0, 1.0, 0.9, 1.0], [0.0, 0.0, 0.0, 0.1]),
        torch.tensor([0, 1, 2, 0]),
    )
    assert [row.tolist() for row in out.next_token_top_logprobs_idx[1:]] == [[1], [2], [0]]
    assert len(out.next_token_top_logprobs_idx[0]) == 3


def test_patch_leaves_requests_without_top_logprobs_alone(sampler_cls):
    import torch

    out = SimpleNamespace()
    sampler_cls()._attach_logprobs_to_output(
        out,
        torch.zeros(1, 3),
        [0],
        [None],
        _sampling_info(torch, [TOP_K_ALL], [1.0], [0.0]),
        torch.tensor([0]),
    )
    assert not hasattr(out, "next_token_top_logprobs_idx")
