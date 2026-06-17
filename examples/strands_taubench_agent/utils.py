import copy
import json
import logging
from typing import Any

from constants import TOOL_RESULT_KEY, TOOL_USE_KEY

logger = logging.getLogger(__name__)

# Type aliases for the Strands dict shapes handled throughout this module.
ContentBlock = dict[str, Any]
Message = dict[str, Any]


def make_strands_tool(env: Any, tool_name: str, tau_tool: Any, requestor: str = "assistant") -> Any:
    """Wrap a tau-bench tool as a Strands-compatible tool.

    Tau-bench tools interact closely with the database (environment state).
    This builds a Strands ``PythonAgentTool`` whose ``tool_spec`` carries the
    full parameter schema from tau-bench's ``openai_schema``, and whose handler
    routes calls to the environment and returns a Strands ``ToolResult`` dict
    (PythonAgentTool does not auto-wrap the result the way the ``@tool``
    decorator does).

    Args:
        env: The tau-bench Environment that executes the tool and holds DB state.
        tool_name: Name of the tool to invoke on the environment.
        tau_tool: The tau-bench tool object exposing ``openai_schema`` and ``short_desc``.
        requestor: "assistant" or "user" — selects which toolkit the call routes
            to (``env.use_tool`` vs ``env.use_user_tool``).

    Returns:
        A ``PythonAgentTool`` ready to be passed to a Strands ``Agent``.
    """
    from strands.tools.tools import PythonAgentTool

    # Convert tau-bench openai_schema to Strands tool_spec
    func_schema = tau_tool.openai_schema["function"]
    tool_spec = {
        "name": func_schema["name"],
        "description": func_schema.get("description", tau_tool.short_desc),
        "inputSchema": {"json": func_schema.get("parameters", {})},
    }

    def tool_func(tool_use: dict, **kwargs: Any) -> dict:
        # PythonAgentTool passes tool_use as first arg: {"toolUseId": ..., "name": ..., "input": {...}}
        tool_input = tool_use.get("input", {})
        tool_use_id = str(tool_use.get("toolUseId", ""))
        try:
            if requestor == "user":
                result = env.use_user_tool(tool_name, **tool_input)
            else:
                result = env.use_tool(tool_name, **tool_input)
            env.sync_tools()
            result_str = env.to_json_str(result)
            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": result_str}],
            }
        except Exception as e:
            logger.error("Tool error: %s: %s", tool_name, e)
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Error: {e}"}],
            }

    return PythonAgentTool(tool_name, tool_spec, tool_func)


def _has_tool_use(message: Message) -> bool:
    """Check whether a message contains any toolUse content block.

    Args:
        message: A Strands message.

    Returns:
        True if at least one content block is a tool call.
    """
    return any(TOOL_USE_KEY in block for block in message.get("content", []))


def _has_tool_result(message: Message) -> bool:
    """Check whether a message contains any toolResult content block.

    Args:
        message: A Strands message.

    Returns:
        True if at least one content block is a tool result.
    """
    return any(TOOL_RESULT_KEY in block for block in message.get("content", []))


def extract_text(message: Message) -> str:
    """Concatenate all text blocks of a message.

    Thinking (reasoningContent) and tool (toolUse/toolResult) blocks are skipped,
    so this returns only the model's user-facing text.

    Args:
        message: A Strands message.

    Returns:
        The message's text blocks joined by spaces (empty string if none).
    """
    return " ".join(
        block["text"] for block in message.get("content", []) if isinstance(block, dict) and "text" in block
    )


def _strip_thinking(content: list[ContentBlock]) -> list[ContentBlock]:
    """Remove thinking from a content-block list.

    Handles both Bedrock format (``reasoningContent`` blocks, dropped entirely)
    and vLLM format (``<think>...</think>`` embedded in a text block, where only
    the text after ``</think>`` is kept).

    Args:
        content: The content blocks of a single message.

    Returns:
        A new list of content blocks with thinking removed.
    """
    result: list[ContentBlock] = []
    for block in content:
        if "reasoningContent" in block:
            continue
        if "text" in block and "</think>" in block["text"]:
            text_after = block["text"].split("</think>", 1)[1].strip()
            if text_after:
                result.append({"text": text_after})
        else:
            result.append(block)
    return result


def convert_message_role(messages: list[Message], to_role: str, strip_thinking: bool = False) -> list[Message]:
    """Re-project the shared conversation history into one agent's perspective.

    Shared messages are in Strands/Bedrock format, tagged with "from" for origin.
    Each agent is always "assistant" in its own Strands perspective, so the other
    agent's turns must be re-cast as "user" input.

    For the target agent:
      - Messages without "from" (e.g. system): passed through as-is.
      - Own messages (from == to_role): kept, optionally with thinking stripped, "from" stripped.
      - Other agent's text (no toolUse/toolResult): included as "user" input, thinking stripped.
      - Other agent's tool calls and tool results: excluded.
      - Any other (unexpected) origin: dropped.

    Args:
        messages: The shared conversation history (each tagged with "from").
        to_role: Whose perspective to build for — "user" or "assistant".
        strip_thinking: If True, strip thinking from own messages in history.
            Used for both agents to force self-contained reasoning per turn and
            avoid context blowup when thinking is enabled on self-hosted inference
            servers.

    Returns:
        A new message list in the target agent's Strands perspective.
    """
    other_role = "assistant" if to_role == "user" else "user"
    filtered: list[Message] = []
    for m in messages:
        if "from" not in m:
            # System / untagged message — pass through as-is.
            filtered.append(copy.deepcopy(m))
            continue
        origin = m["from"]
        if origin == to_role:
            # Own message — preserve tool calls/results, optionally strip thinking. "from" stripped.
            new_m = copy.deepcopy(m)
            new_m.pop("from", None)
            if strip_thinking:
                new_m["content"] = _strip_thinking(new_m["content"])
            filtered.append(new_m)
        elif origin == other_role:
            # Other agent — drop its tool calls/results, include only its text as "user" input.
            if _has_tool_use(m) or _has_tool_result(m):
                continue
            clean_content = _strip_thinking(m["content"])
            if clean_content:
                filtered.append({"role": "user", "content": clean_content})
        # else: tagged with an unexpected origin — dropped (matches original behavior)
    return filtered


def _convert_think_tags_to_reasoning_blocks(content_blocks: list[ContentBlock]) -> list[ContentBlock]:
    """Convert inline ``<think>...</think>`` text into separate reasoningContent blocks.

    vLLM embeds thinking in text content with ``<think>...</think>`` tags.
    This extracts them into ``reasoningContent`` blocks to match Bedrock's format.

    Args:
        content_blocks: The content blocks of a single message.

    Returns:
        A new list of content blocks where any inline thinking has been split
        into a dedicated reasoningContent block followed by the remaining text.
    """
    new_blocks: list[ContentBlock] = []
    for block in content_blocks:
        if "text" in block and "</think>" in block["text"]:
            parts = block["text"].split("</think>", 1)
            thinking = parts[0].replace("<think>", "").strip()
            text = parts[1].strip() if len(parts) > 1 else ""
            if thinking:
                new_blocks.append({"reasoningContent": {"reasoningText": {"text": thinking}}})
            if text:
                new_blocks.append({"text": text})
        else:
            new_blocks.append(block)
    return new_blocks


def log_turn(turn: int, role: str, global_messages: list[Message], orchestrator_config: dict) -> list[Message]:
    """Log a turn header and return the messages converted for the given role.

    Args:
        turn: Zero-based turn index, used in the log line.
        role: Whose turn it is — "user" or "assistant".
        global_messages: The shared conversation history.
        orchestrator_config: Orchestrator settings; the "strip_thinking_from_history"
            flag controls whether thinking is stripped from own messages.

    Returns:
        The history converted to ``role``'s Strands perspective (see
        ``convert_message_role``).
    """
    logger.info("TURN %d — %s", turn, role.upper())
    msgs = convert_message_role(
        global_messages,
        to_role=role,
        strip_thinking=orchestrator_config.get("strip_thinking_from_history", False),
    )
    logger.debug("[%s-turn %d] full messages:\n%s", role, turn, json.dumps(msgs, indent=2, default=str))
    return msgs


def to_real_world_roles(global_messages: list[Message]) -> list[Message]:
    """Normalize the shared history for saving as the final trajectory.

    Sets each message's ``role`` to its real-world identity (the "from" value)
    while keeping "from" for downstream analysis, and splits inline
    ``<think>...</think>`` tags into reasoningContent blocks.

    Only the assistant is vLLM-backed (its text carries inline ``<think>`` tags),
    so splitting is applied to assistant messages only. The user simulator is
    Bedrock-backed and already returns reasoningContent natively.

    Args:
        global_messages: The shared conversation history.

    Returns:
        A new message list with real-world roles and normalized reasoning blocks.
    """
    result: list[Message] = []
    for m in global_messages:
        new_m = copy.deepcopy(m)
        if "from" in new_m:
            new_m["role"] = new_m["from"]
        if new_m.get("from") == "assistant":
            new_m["content"] = _convert_think_tags_to_reasoning_blocks(new_m["content"])
        result.append(new_m)
    return result
