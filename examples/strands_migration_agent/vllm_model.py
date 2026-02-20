"""vLLM model provider with token ID collection for RL training."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

import openai
from strands.models.openai import OpenAIModel
from strands.types.content import Messages
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec
from typing_extensions import override

logger = logging.getLogger(__name__)


class vLLMModel(OpenAIModel):
    """vLLM model that collects token IDs from responses."""

    def __init__(
        self,
        client: Any | None = None,
        client_args: dict[str, Any] | None = None,
        **model_config: Any,
    ) -> None:
        params = model_config.get("params", {}) or {}
        params.setdefault("logprobs", True)
        model_config["params"] = params

        super().__init__(client, client_args, **model_config)
        self._token_data: list[dict[str, Any]] = []

    def format_request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        request = super().format_request(messages, tool_specs, system_prompt, tool_choice, **kwargs)
        request["stream"] = False
        request.pop("stream_options", None)
        request["extra_body"] = {"return_token_ids": True}
        return request

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        request = self.format_request(messages, tool_specs, system_prompt, tool_choice, **kwargs)

        async with self._get_client() as client:
            try:
                response = await client.chat.completions.create(**request)
            except openai.BadRequestError as e:
                if hasattr(e, "code") and e.code == "context_length_exceeded":
                    raise ContextWindowOverflowException(str(e)) from e
                raise
            except openai.RateLimitError as e:
                raise ModelThrottledException(str(e)) from e

            # Store token data
            choice = response.choices[0]
            logprobs = []
            if choice.logprobs and choice.logprobs.content:
                logprobs = [lp.logprob for lp in choice.logprobs.content]

            self._token_data.append(
                {
                    "prompt_ids": getattr(response, "prompt_token_ids", []) or [],
                    "response_ids": getattr(choice, "token_ids", []) or [],
                    "response_logprobs": logprobs,
                }
            )

            # Yield synthetic stream events for Strands compatibility
            yield self.format_chunk({"chunk_type": "message_start"})

            message = choice.message
            if message.content:
                yield self.format_chunk({"chunk_type": "content_start", "data_type": "text"})
                yield self.format_chunk({"chunk_type": "content_delta", "data_type": "text", "data": message.content})
                yield self.format_chunk({"chunk_type": "content_stop", "data_type": "text"})

            for tool_call in message.tool_calls or []:
                yield self.format_chunk({"chunk_type": "content_start", "data_type": "tool", "data": tool_call})
                yield self.format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": tool_call})
                yield self.format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

            yield self.format_chunk({"chunk_type": "message_stop", "data": choice.finish_reason})

            if response.usage:
                yield self.format_chunk({"chunk_type": "metadata", "data": response.usage})

    def get_token_data(self) -> list[dict[str, Any]]:
        return self._token_data

    def clear_token_data(self) -> None:
        self._token_data = []
