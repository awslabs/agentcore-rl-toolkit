"""``SglangHttpBackend`` top-logprob heads: the request and both response encodings."""

import base64

import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from agentcore_rl_toolkit.rollout_gateway.sampling_backends.base import supports_top_logprobs
from agentcore_rl_toolkit.rollout_gateway.sampling_backends.sglang_http import SglangHttpBackend

IDS = [[7, 4, 3], [8, 5, 6]]
LOGPROBS = [[-0.1, -2.5, -3.0], [-0.2, -2.0, -4.0]]


def _json_meta(output_ids, ids=IDS, logprobs=LOGPROBS):
    """Stock SGLang: ``output_top_logprobs`` rows of ``[logprob, id, text]``."""
    return {
        "output_token_logprobs": [[-0.5, t, None] for t in output_ids],
        "output_top_logprobs": [
            [[lp, t, None] for lp, t in zip(lps, row, strict=True)] for lps, row in zip(logprobs, ids, strict=True)
        ],
        "finish_reason": {"type": "stop"},
    }


def _binary_meta(output_ids, ids=IDS, logprobs=LOGPROBS):
    """slime's SGLang patch adds the head as base64 little-endian int32 / float32."""
    meta = _json_meta(output_ids)
    meta["output_top_logprobs"] = [[[0.0, 0, None]] * 3] * len(output_ids)  # must not be used
    meta["output_topk_token_ids"] = base64.b64encode(np.asarray(ids, dtype="<i4").tobytes()).decode()
    meta["output_topk_log_probs"] = base64.b64encode(np.asarray(logprobs, dtype="<f4").tobytes()).decode()
    meta["output_topk_shape"] = list(np.asarray(ids).shape)
    return meta


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


def test_backend_accepts_top_logprobs():
    assert supports_top_logprobs(SglangHttpBackend("http://x"))


@pytest.mark.parametrize("meta", [_json_meta([7, 8]), _binary_meta([7, 8])], ids=["json", "binary"])
@pytest.mark.asyncio
async def test_backend_requests_and_parses_the_head(meta):
    turn, (body,) = await _generate(meta, num_top_logprobs=3)
    assert body["top_logprobs_num"] == 3
    assert turn.output_ids == [7, 8]
    head = turn.output_top_logprobs
    assert head.ids.dtype == np.int32 and head.logprobs.dtype == np.float32
    assert head.ids.tolist() == IDS
    assert np.allclose(head.logprobs, LOGPROBS)


@pytest.mark.asyncio
async def test_backend_without_top_logprobs_sends_no_top_logprobs_num():
    turn, (body,) = await _generate(_json_meta([7, 8]))
    assert "top_logprobs_num" not in body
    assert turn.output_top_logprobs is None


@pytest.mark.parametrize(
    "meta",
    [
        _json_meta([7, 8], ids=[[7, 4, 3]], logprobs=[[-0.1, -2.5, -3.0]]),  # one row short
        _json_meta([7, 8], ids=[[7, 4], [8, 5]], logprobs=[[-0.1, -2.5], [-0.2, -2.0]]),  # rows too narrow
        {**_binary_meta([7, 8]), "output_topk_shape": [2, 2]},  # wrong binary shape
    ],
)
@pytest.mark.asyncio
async def test_backend_rejects_a_malformed_head(meta):
    with pytest.raises(RuntimeError, match="SGLang"):
        await _generate(meta, num_top_logprobs=3)
