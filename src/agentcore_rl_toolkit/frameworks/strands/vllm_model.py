"""vLLM model provider with token ID collection for RL training."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

import openai
from typing_extensions import override

from strands.models.openai import OpenAIModel
from strands.types.content import Messages
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec

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
        # Parent already sets stream=True and stream_options={"include_usage": True}.
        # Add vLLM-specific extra_body to return token IDs in each streaming chunk.
        existing_extra = request.get("extra_body", {})
        existing_extra["return_token_ids"] = True
        request["extra_body"] = existing_extra
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
        """Stream chat completions from vLLM, collecting token data for RL training.

        Follows the same streaming event protocol as the parent OpenAIModel.stream(),
        but additionally accumulates prompt_token_ids, response token_ids, and logprobs
        from each chunk (vLLM extensions via return_token_ids=True).

        Uses streaming mode so that vLLM can detect client disconnects (e.g., when an
        ACR container is killed on timeout) and abort in-flight generation, freeing
        GPU slots immediately instead of running orphaned requests to completion.
        """
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

            # Accumulators for RL token data (collected from vLLM-specific chunk fields)
            prompt_ids: list[int] = []
            response_ids: list[int] = []
            response_logprobs: list[float] = []

            yield self.format_chunk({"chunk_type": "message_start"})
            tool_calls: dict[int, list[Any]] = {}
            data_type = None
            finish_reason = None
            event = None

            async for event in response:
                raw = event.model_dump()

                # Collect prompt_token_ids from response-level field (first chunk that has it)
                if not prompt_ids and "prompt_token_ids" in raw:
                    prompt_ids = raw["prompt_token_ids"] or []

                if not getattr(event, "choices", None):
                    continue
                choice = event.choices[0]
                choice_raw = raw["choices"][0] if raw.get("choices") else {}

                # Collect per-token response IDs (vLLM extension)
                token_ids = choice_raw.get("token_ids") or choice_raw.get("delta", {}).get("token_ids")
                if token_ids:
                    if isinstance(token_ids, list):
                        response_ids.extend(token_ids)
                    else:
                        response_ids.append(token_ids)

                # Collect per-token logprobs
                if choice.logprobs and choice.logprobs.content:
                    for lp in choice.logprobs.content:
                        response_logprobs.append(lp.logprob)

                # Yield stream events (mirrors parent OpenAIModel.stream() protocol)
                if hasattr(choice.delta, "reasoning_content") and choice.delta.reasoning_content:
                    chunks, data_type = self._stream_switch_content("reasoning_content", data_type)
                    for chunk in chunks:
                        yield chunk
                    yield self.format_chunk(
                        {
                            "chunk_type": "content_delta",
                            "data_type": data_type,
                            "data": choice.delta.reasoning_content,
                        }
                    )

                if choice.delta.content:
                    chunks, data_type = self._stream_switch_content("text", data_type)
                    for chunk in chunks:
                        yield chunk
                    yield self.format_chunk(
                        {"chunk_type": "content_delta", "data_type": "text", "data": choice.delta.content}
                    )

                for tool_call in choice.delta.tool_calls or []:
                    tool_calls.setdefault(tool_call.index, []).append(tool_call)

                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                    if data_type:
                        yield self.format_chunk({"chunk_type": "content_stop", "data_type": data_type})
                    break

            for tool_deltas in tool_calls.values():
                yield self.format_chunk({"chunk_type": "content_start", "data_type": "tool", "data": tool_deltas[0]})
                for tool_delta in tool_deltas:
                    yield self.format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": tool_delta})
                yield self.format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

            yield self.format_chunk({"chunk_type": "message_stop", "data": finish_reason or "end_turn"})

            # Consume remaining chunks for usage metadata
            async for event in response:
                raw = event.model_dump()
                if not prompt_ids and "prompt_token_ids" in raw:
                    prompt_ids = raw["prompt_token_ids"] or []

            if event and hasattr(event, "usage") and event.usage:
                yield self.format_chunk({"chunk_type": "metadata", "data": event.usage})

            # Store collected token data for RL training
            self._token_data.append(
                {
                    "prompt_ids": prompt_ids,
                    "response_ids": response_ids,
                    "response_logprobs": response_logprobs,
                }
            )

    def get_token_data(self) -> list[dict[str, Any]]:
        return self._token_data

    def clear_token_data(self) -> None:
        self._token_data = []
