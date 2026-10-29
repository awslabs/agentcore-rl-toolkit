"""VllmHttpBackend request payloads."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from agentcore_rl_toolkit.rollout_gateway.sampling_backends.vllm_http import VllmHttpBackend


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [None, "policy-lora"])
async def test_generate_names_the_served_adapter(model):
    requests = []

    async def generate(request):
        requests.append(await request.json())
        return web.json_response({"choices": [{"token_ids": [7, 8], "finish_reason": "stop"}]})

    app = web.Application()
    app.router.add_post("/inference/v1/generate", generate)
    async with TestServer(app) as server:
        backend = VllmHttpBackend(str(server.make_url("")), model=model)
        record = await backend.generate(prompt_ids=[1, 2], sampling_params={})
    assert record.output_ids == [7, 8]
    assert requests[0].get("model") == model
