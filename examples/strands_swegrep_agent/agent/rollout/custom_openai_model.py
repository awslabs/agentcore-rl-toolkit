"""Strands' OpenAIModel, changed to send earlier turns back exactly as the model generated them.

The stock model drops reasoning from the history, and Strands renames a tool call with an invalid name to
INVALID_TOOL_NAME. Either way the model would see a history it didn't write, and the rollout gateway's
token record would no longer match. This model sends reasoning back as ``reasoning_content`` and
restores the original tool names. Use one instance per rollout.
"""

from typing import Any

from strands.models.openai import OpenAIModel
from strands.types.content import Messages

INVALID_TOOL_NAME = "INVALID_TOOL_NAME"


class CustomOpenAIModel(OpenAIModel):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tool_names: dict[str, str] = {}  # toolUseId -> name as generated

    @classmethod
    def _format_regular_messages(cls, messages: Messages, **kwargs: Any) -> list[dict[str, Any]]:
        formatted = []
        for message in messages:
            contents = message["content"]
            # Strip reasoning before the stock formatter, which would drop it with a warning.
            stripped = {**message, "content": [c for c in contents if "reasoningContent" not in c]}
            entries = super()._format_regular_messages([stripped], **kwargs)
            reasoning = "".join(
                (block["reasoningContent"].get("reasoningText") or {}).get("text") or ""
                for block in contents
                if isinstance(block.get("reasoningContent"), dict)
            )
            if reasoning and entries and message["role"] == "assistant":
                entries[0]["reasoning_content"] = reasoning
            formatted.extend(entries)
        return formatted

    def format_chunk(self, event: dict[str, Any], **kwargs: Any) -> Any:
        if event["chunk_type"] == "content_start" and event["data_type"] == "tool":
            call = event["data"]
            if call.function and not call.function.name:
                # Strands would drop a nameless call, leaving an empty turn the turn cap does not
                # count. Named INVALID_TOOL_NAME, it stays a tool call and gets an error result.
                call.function.name = INVALID_TOOL_NAME
            elif call.id and call.function and call.function.name:
                self._tool_names[call.id] = call.function.name
        return super().format_chunk(event, **kwargs)

    def format_request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        request = super().format_request(*args, **kwargs)
        for message in request["messages"]:
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                if fn.get("name") == INVALID_TOOL_NAME and call.get("id") in self._tool_names:
                    fn["name"] = self._tool_names[call["id"]]
        return request
