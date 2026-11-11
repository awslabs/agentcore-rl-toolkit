"""The agent's answer: the last <answer> block of its final message, if that message calls no tool."""

import re
from dataclasses import dataclass, field

from scoring.regions import Region, parse_regions, validate_regions

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass
class Answer:
    submitted: bool = False
    regions: list[Region] = field(default_factory=list)
    invalid_regions: list[str] = field(default_factory=list)
    search_turns: int = 0  # assistant turns that called a tool


def _tool_calls(content) -> list:
    return [b for b in content if isinstance(b, dict) and "toolUse" in b] if isinstance(content, list) else []


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and "text" in b)
    return ""


def extract_answer(messages: list[dict], repo_root: str = "") -> Answer:
    """Parse the answer; with repo_root, also drop regions that don't exist in the checkout."""
    turns = [m.get("content", []) for m in messages if m.get("role") == "assistant"]
    answer = Answer(search_turns=sum(bool(_tool_calls(content)) for content in turns))
    if not turns or _tool_calls(turns[-1]):
        return answer
    # Reasoning can sit in the text (its opening <think> may be part of the prompt), and an answer
    # example quoted in it must not count.
    text = _text(turns[-1])
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = text.split("<think>", 1)[0]
    blocks = _ANSWER.findall(text)
    if "<tool_call>" in text or not blocks:
        return answer
    answer.submitted = True
    answer.regions, answer.invalid_regions = parse_regions(blocks[-1])
    if repo_root:
        answer.regions, invalid = validate_regions(answer.regions, repo_root)
        answer.invalid_regions += invalid
    return answer
