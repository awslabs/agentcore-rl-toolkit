"""HfTemplateRenderer round-trip tests.

Uses a tiny stub tokenizer (no transformers download) to assert the renderer calls
apply_chat_template with the expected kwargs and decodes/parses output correctly. A
real-tokenizer parity check lives in the Step 2 E2E path (against the live server).
The schema-derender path (recognized chat template -> tokenizer.parse_response) is
covered in test_response_schemas.py; this module covers the two-stage fallback.
"""

import sys

import pytest

from agentcore_rl_toolkit.rollout_gateway.parsing import parse_tool_uses
from agentcore_rl_toolkit.rollout_gateway.render import HfTemplateRenderer, ParsedOutput


class StubTokenizer:
    def __init__(self):
        self.last_kwargs = None

    def apply_chat_template(self, messages, *, tools=None, tokenize=True, add_generation_prompt=True, return_dict=True):
        self.last_kwargs = dict(
            tools=tools, tokenize=tokenize, add_generation_prompt=add_generation_prompt, return_dict=return_dict
        )
        # deterministic: one id per message, +99 sentinel for the generation prompt
        ids = [len(m.get("content") or "") for m in messages]
        if add_generation_prompt:
            ids.append(99)
        # mirror the real API: the default dict form bundles extras the renderer
        # must opt out of with return_dict=False
        return ids if not return_dict else {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids)


def test_render_passes_expected_kwargs_and_returns_list():
    tok = StubTokenizer()
    r = HfTemplateRenderer(tok)
    ids = r.render([{"role": "user", "content": "abc"}], tools=None, add_generation_prompt=True)
    assert ids == [3, 99]
    assert tok.last_kwargs == {"tools": None, "tokenize": True, "add_generation_prompt": True, "return_dict": False}


def test_parse_no_tools_returns_plain_text():
    tok = StubTokenizer()
    r = HfTemplateRenderer(tok)
    out = r.parse([1, 2, 3], tools_schema=None)
    assert isinstance(out, ParsedOutput)
    assert out.text == "1 2 3"
    assert out.tool_uses == []
    assert out.ill_formed is False


def test_parse_empty_output():
    tok = StubTokenizer()
    r = HfTemplateRenderer(tok)
    out = r.parse([], tools_schema=None)
    assert out.text == ""
    assert out.tool_uses == []


def test_xml_tool_calls_parsed_dependency_free():
    """<tool_call><function=...> output is parsed by the regex path with no inference
    engine (sglang/vllm) imported. The regex requires explicit injection — with tools
    in play, an unrecognized template plus no injected parser is rejected."""

    # decode returns the raw XML tool-call text
    class XmlTok(StubTokenizer):
        def decode(self, ids, skip_special_tokens=False):
            return "<tool_call>\n<function=search>\n<parameter=q>cats</parameter>\n</function>\n</tool_call>"

    r = HfTemplateRenderer(XmlTok(), tool_parser=parse_tool_uses)
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    out = r.parse([1], tools_schema=tools)
    assert "sglang" not in sys.modules and "vllm" not in sys.modules
    assert len(out.tool_uses) == 1
    assert out.tool_uses[0]["name"] == "search"
    assert out.tool_uses[0]["input"] == {"q": "cats"}


def test_parse_extracts_reasoning_from_think_block():
    """Reasoning is split on </think> with no engine parser."""

    class ThinkTok(StubTokenizer):
        def decode(self, ids, skip_special_tokens=False):
            return "<think>weighing options</think>the answer is 4"

    r = HfTemplateRenderer(ThinkTok())
    out = r.parse([1], tools_schema=None)
    assert out.reasoning == "weighing options"
    assert out.text == "the answer is 4"


# ---------------------------------------------------------------------------
# reasoning_parser / tool_parser: the two injectable derender stages
# ---------------------------------------------------------------------------
# Each override leaves the other stage on its dependency-free default, and the
# stages run in sequence: reasoning first, tool calls on what remains.

TOOLS = [{"type": "function", "function": {"name": "x", "parameters": {}}}]


class ThinkTok(StubTokenizer):
    def decode(self, ids, skip_special_tokens=False):
        return "<think>hmm</think>body"


def test_stages_run_in_sequence_reasoning_then_tools():
    seen = {}

    def reasoning_parser(raw_output):
        seen["raw"] = raw_output
        return "R", "BODY"

    def tool_parser(body_text, tools_schema):
        seen["body"] = body_text
        seen["tools"] = tools_schema
        return "T", [{"name": "x", "input": {}}], True

    r = HfTemplateRenderer(ThinkTok(), reasoning_parser=reasoning_parser, tool_parser=tool_parser)
    out = r.parse([1], tools_schema=TOOLS)

    assert seen["raw"] == "<think>hmm</think>body"  # reasoning stage sees raw text
    assert seen["body"] == "BODY"  # tool stage sees the reasoning stage's remainder
    assert seen["tools"] == TOOLS
    assert out == ParsedOutput(reasoning="R", text="T", tool_uses=[{"name": "x", "input": {}}], ill_formed=True)


def test_tool_parser_override_keeps_default_reasoning_split():
    """The common case: engine-grade tool parsing, </think> reasoning left to the default."""

    def tool_parser(body_text, tools_schema):
        return "", [{"name": "x", "input": {"got": body_text}}], False

    r = HfTemplateRenderer(ThinkTok(), tool_parser=tool_parser)
    out = r.parse([1], tools_schema=TOOLS)
    assert out.reasoning == "hmm"
    assert out.tool_uses == [{"name": "x", "input": {"got": "body"}}]


def test_reasoning_parser_override_requires_explicit_tool_parser_for_tools():
    """Overriding only the reasoning stage says nothing about the tool format, so
    tool parsing still refuses to run on the implicit default regex; injecting
    parse_tool_uses opts back in and the stages compose as before."""

    class XmlTok(StubTokenizer):
        def decode(self, ids, skip_special_tokens=False):
            return "REASON||<tool_call><function=x><parameter=q>v</parameter></function></tool_call>"

    def split_on_bars(raw):
        left, right = raw.split("||", 1)
        return left, right

    with pytest.raises(ValueError, match="matched no response schema"):
        HfTemplateRenderer(XmlTok(), reasoning_parser=split_on_bars).parse([1], tools_schema=TOOLS)

    r = HfTemplateRenderer(XmlTok(), reasoning_parser=split_on_bars, tool_parser=parse_tool_uses)
    out = r.parse([1], tools_schema=TOOLS)
    assert out.reasoning == "REASON"
    assert out.tool_uses == [{"name": "x", "input": {"q": "v"}}]


def test_tool_parser_skipped_without_tools_schema():
    calls = []

    def tool_parser(body_text, tools_schema):
        calls.append(body_text)
        return "", [], False

    r = HfTemplateRenderer(ThinkTok(), tool_parser=tool_parser)
    out = r.parse([1], tools_schema=None)
    assert calls == []  # no tools -> tool stage never runs
    assert out.reasoning == "hmm"
    assert out.text == "body"
