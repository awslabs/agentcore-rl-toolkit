"""SGLang-backed parsers for the rollout gateway's derender seams.

The gateway samples via SGLang's native ``/generate`` (token-in/token-out), which
bypasses the server's OpenAI layer — so the reasoning/tool-call parsing that layer
would normally do must happen gateway-side. This module builds the two stage
callables :class:`HfTemplateRenderer` accepts, from SGLang's own detectors, so the
gateway derenders exactly what the served engine would have:

- :func:`build_tool_parser` wraps ``FunctionCallParser`` (slime's
  ``--sglang-tool-call-parser``, e.g. ``qwen`` for Qwen2.5/Qwen3 JSON ``<tool_call>``);
- :func:`build_reasoning_parser` wraps ``ReasoningParser`` (slime's
  ``--sglang-reasoning-parser``, e.g. ``qwen3``).

Both are independent: pass only the one whose format the built-in default gets wrong
and the renderer keeps its dependency-free default for the other stage.

Lives in the slime backend (not ``rollout_gateway``) because slime trainers always
have sglang importable — the gateway package itself stays engine-free.
"""

import json
import logging

logger = logging.getLogger(__name__)


def _import_sglang(what: str, flag: str):
    """Import an sglang parser symbol, or raise with actionable context."""
    try:
        if what == "tool":
            from sglang.srt.entrypoints.openai.protocol import Tool
            from sglang.srt.function_call.function_call_parser import FunctionCallParser

            return FunctionCallParser, Tool
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        return ReasoningParser, None
    except ImportError as err:
        raise ImportError(
            f"{flag} needs sglang's parsers, which the slime trainer environment normally "
            f"provides; unset {flag} to fall back to the gateway's built-in "
            "dependency-free parsing."
        ) from err


def build_tool_parser(parser_name: str):
    """Build a renderer ``tool_parser``: ``(body_text, tools_schema) -> (text, tool_uses, ill_formed)``.

    ``parser_name`` is an SGLang ``--tool-call-parser`` name and must match the served
    model. Imported and validated eagerly so a bad name or environment fails at gateway
    construction, not on the first rollout turn.
    """
    FunctionCallParser, Tool = _import_sglang("tool", "--sglang-tool-call-parser")
    if parser_name not in FunctionCallParser.ToolCallParserEnum:
        raise ValueError(
            f"unknown tool_call_parser {parser_name!r}; choose one of {sorted(FunctionCallParser.ToolCallParserEnum)}"
        )

    def parse_tool_uses(body_text: str, tools_schema: list[dict]) -> tuple[str, list[dict], bool]:
        # detectors carry streaming state -> fresh parser per turn
        parser = FunctionCallParser([Tool.model_validate(t) for t in tools_schema], parser_name)
        normal_text, calls = parser.parse_non_stream(body_text)

        tool_uses: list[dict] = []
        ill_formed = False
        for call in calls:
            if not call.name:
                ill_formed = True
                continue
            try:
                # ToolCallItem.parameters is a JSON string; adapters want a dict
                args = json.loads(call.parameters) if call.parameters else {}
            except json.JSONDecodeError:
                ill_formed = True
                continue
            tool_uses.append({"name": call.name, "input": args if isinstance(args, dict) else {}})
        # the model opened a tool call but the parser extracted nothing usable
        if not tool_uses and parser.has_tool_call(body_text):
            ill_formed = True
        return normal_text or "", tool_uses, ill_formed

    return parse_tool_uses


def build_reasoning_parser(parser_name: str):
    """Build a renderer ``reasoning_parser``: ``raw_output -> (reasoning, body_text)``.

    ``parser_name`` is an SGLang ``--reasoning-parser`` name (e.g. ``qwen3``). Use this
    for models whose reasoning is not a plain ``</think>`` block; the gateway's default
    handles that common case with no dependency.
    """
    ReasoningParser, _ = _import_sglang("reasoning", "--sglang-reasoning-parser")
    if parser_name.lower() not in ReasoningParser.DetectorMap:
        raise ValueError(
            f"unknown reasoning_parser {parser_name!r}; choose one of {sorted(ReasoningParser.DetectorMap)}"
        )

    def split_reasoning(raw_output: str) -> tuple[str, str]:
        reasoning, body = ReasoningParser(model_type=parser_name).parse_non_stream(raw_output)
        return reasoning or "", body or ""

    return split_reasoning


__all__ = ["build_reasoning_parser", "build_tool_parser"]
