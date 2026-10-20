"""A fresh Strands conversation for each self-contained database task."""

import asyncio
import json
from typing import Any

from strands import Agent
from strands.models import OpenAIModel

from .config import ModelConfig
from .contracts import DatabaseRequest, DatabaseResponse


class DatabaseAgent:
    def __init__(self, system_prompt: str, tools: list[Any]) -> None:
        self.system_prompt = system_prompt
        self.tools = tools

    async def answer(self, query: str, config: ModelConfig) -> DatabaseResponse:
        request = DatabaseRequest(query=query)
        params = dict(config.sampling_params)
        if not {"max_tokens", "max_completion_tokens"} & params.keys():
            params["max_tokens"] = 8192
        model = OpenAIModel(
            model_id=config.model_id,
            client_args={
                "base_url": config.base_url,
                "api_key": config.api_key.get_secret_value(),
                "timeout": min(180, config.timeout_seconds),
                "max_retries": 0,
            },
            params=params,
        )
        agent = Agent(
            model=model,
            system_prompt=self.system_prompt,
            tools=self.tools,
            context_manager=False,
            callback_handler=None,
        )
        async with asyncio.timeout(config.timeout_seconds):
            result = await agent.invoke_async(
                json.dumps(request.model_dump(), ensure_ascii=True, indent=2),
                structured_output_model=DatabaseResponse,
                limits={"turns": config.turn_limit},
            )
        if result.structured_output is None:
            raise RuntimeError("Database agent returned no structured response")
        return DatabaseResponse.model_validate(result.structured_output)
