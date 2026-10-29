"""Token-sample draws over vLLM's ``/inference/v1/generate``: the endpoint patch keeps
duplicate draws, and ``VllmHttpBackend`` parses them into ``TurnRecord.output_auxiliary_token_samples``."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from agentcore_rl_toolkit.rollout_gateway.sampling_backends.vllm_http import VllmHttpBackend


def _content(rows):
    """Endpoint logprobs content: per position, the sampled token then its draws."""
    return [
        {
            "token": f"token_id:{row[0]}",
            "logprob": -0.1,
            "top_logprobs": [{"token": f"token_id:{t}", "logprob": -0.1} for t in row],
        }
        for row in rows
    ]


async def _serve(choice):
    seen = {}

    async def generate(request):
        seen.update(await request.json())
        return web.json_response({"choices": [choice]})

    app = web.Application()
    app.router.add_post("/inference/v1/generate", generate)
    server = TestServer(app)
    await server.start_server()
    return server, seen


@pytest.mark.asyncio
async def test_vllm_http_backend_returns_draws():
    choice = {"token_ids": [5, 6], "finish_reason": "stop", "logprobs": {"content": _content([[5, 9, 9], [6, 6, 2]])}}
    server, seen = await _serve(choice)
    try:
        turn = await VllmHttpBackend(str(server.make_url(""))).generate(
            prompt_ids=[1], sampling_params={}, num_auxiliary_token_samples=2
        )
    finally:
        await server.close()
    assert seen["sampling_params"]["logprobs"] == 2 and seen["sampling_params"]["flat_logprobs"] is True
    assert turn.output_auxiliary_token_samples.ids == [[9, 9], [6, 2]]
    assert turn.output_log_probs == [-0.1, -0.1]


@pytest.mark.asyncio
async def test_vllm_http_backend_rejects_unpatched_server():
    # an unpatched server collapses duplicate draws, leaving fewer than K + 1 entries
    choice = {"token_ids": [5], "finish_reason": "stop", "logprobs": {"content": _content([[5, 9]])}}
    server, _ = await _serve(choice)
    try:
        with pytest.raises(RuntimeError, match="did not return 2 token samples"):
            await VllmHttpBackend(str(server.make_url(""))).generate(
                prompt_ids=[1], sampling_params={}, num_auxiliary_token_samples=2
            )
    finally:
        await server.close()


def test_generate_endpoint_patch_keeps_duplicate_draws():
    pytest.importorskip("vllm")
    from vllm.entrypoints.serve.disagg.serving import ServingTokens
    from vllm.logprobs import FlatLogprobs

    from agentcore_rl_toolkit.rollout_gateway.sampling_backends import vllm_token_samples

    original = ServingTokens._create_tokens_logprobs
    vllm_token_samples.install_generate_endpoint_patch()
    try:
        flat = FlatLogprobs()
        flat.append_fast([5, 9, 9], [-0.1, -2.0, -2.0], iter([1, 3, 3]), [None] * 3)
        content = ServingTokens._create_tokens_logprobs(None, [5], flat, 2).content
    finally:
        ServingTokens._create_tokens_logprobs = original
    assert [e.token for e in content[0].top_logprobs] == ["token_id:5", "token_id:9", "token_id:9"]
