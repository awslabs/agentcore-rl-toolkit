"""Dependency-free model-output parsing for agent harnesses.

Splits raw decoded model text into reasoning, visible text, and tool calls using only
the stdlib:
- reasoning is separated on a ``</think>`` marker (the common thinking-model convention);
- tool calls are matched with a regex for the ``<tool_call><function=...>`` format.

This covers common models (e.g. Qwen3-Coder) with no inference-engine dependency. Models
whose formats need engine-grade parsing are handled by supplying a custom parser to the
renderer (see the ``Renderer`` seam), not here.
"""

import dataclasses
import re
from typing import Any


@dataclasses.dataclass(frozen=True)
class ParsedModelOutput:
    """Structured view of one decoded model output."""

    reasoning: str
    text: str
    tool_uses: list[dict[str, Any]]
    ill_formed: bool = False


def parse_model_output(
    raw_output: str,
    *,
    tools_schema: list[dict] | None,
) -> ParsedModelOutput:
    """Parse raw model text into reasoning, visible text, and tool uses."""
    reasoning, body_text = "", raw_output
    if "</think>" in body_text:
        reasoning, body_text = body_text.split("</think>", 1)
        reasoning = reasoning.removeprefix("<think>")

    tool_uses: list[dict[str, Any]] = []
    if tools_schema:
        body_text, tool_uses = parse_xml_tool_uses(body_text, tools_schema)

    return ParsedModelOutput(
        reasoning=reasoning.strip(),
        text=(body_text or "").strip(),
        tool_uses=tool_uses,
    )


def parse_xml_tool_uses(body_text: str, tools_schema: list[dict]) -> tuple[str, list[dict[str, Any]]]:
    """Parse ``<tool_call><function=NAME>...<parameter=K>V</parameter>...</function></tool_call>``
    blocks, returning the text with those blocks removed plus the extracted tool calls."""
    valid_tools = {t.get("function", {}).get("name") for t in tools_schema}
    tool_uses: list[dict[str, Any]] = []
    cleaned_parts: list[str] = []
    last = 0
    for m in re.finditer(
        r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
        body_text,
        flags=re.DOTALL,
    ):
        name, inner = m.group(1), m.group(2)
        if name in valid_tools:
            args = {
                p.group(1): p.group(2).strip()
                for p in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", inner, flags=re.DOTALL)
            }
            tool_uses.append({"name": name, "input": args})
            cleaned_parts.append(body_text[last : m.start()])
            last = m.end()
    cleaned_parts.append(body_text[last:])
    return "".join(cleaned_parts), tool_uses


__all__ = ["ParsedModelOutput", "parse_model_output", "parse_xml_tool_uses"]
