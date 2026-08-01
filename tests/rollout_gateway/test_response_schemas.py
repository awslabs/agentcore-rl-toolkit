"""Schema-based derendering: detection by chat-template hash + parse_response parsing.

Uses a stub tokenizer whose ``parse_response`` delegates to transformers' real
schema-parsing engine (``recursive_parse`` — exactly what
``PreTrainedTokenizerBase.parse_response`` wraps), so the vendored schemas are
exercised against the real parser without any hub download.
"""

import hashlib

from transformers.utils.chat_parsing_utils import recursive_parse

from agentcore_rl_toolkit.rollout_gateway.render import HfTemplateRenderer, ParsedOutput
from agentcore_rl_toolkit.rollout_gateway.response_schemas import (
    _TEMPLATE_HASHES,
    RESPONSE_SCHEMAS,
    resolve_schema_name,
)

# Any template string whose hash is registered works for detection; tests use a
# sentinel mapped to the schema under test.
QWEN3_TEMPLATE = "{# test-template: qwen3 #}"
QWEN3_5_TEMPLATE = "{# test-template: qwen3_5 #}"
TOOLS = [{"type": "function", "function": {"name": "calculator", "parameters": {}}}]


def _register(template: str, schema_name: str):
    _TEMPLATE_HASHES[hashlib.sha256(template.encode("utf-8")).hexdigest()] = schema_name


_register(QWEN3_TEMPLATE, "qwen3")
_register(QWEN3_5_TEMPLATE, "qwen3_5")


class SchemaStubTokenizer:
    """Chat-template-bearing stub whose parse_response is the real engine."""

    name_or_path = "stub"

    def __init__(self, decoded: str = "", chat_template: str | None = QWEN3_TEMPLATE):
        self.decoded = decoded
        self.chat_template = chat_template

    def decode(self, ids, skip_special_tokens=False):
        return self.decoded

    def parse_response(self, response, schema=None):
        return recursive_parse(response, schema)


# ---------------------------------------------------------------------------
# resolve_schema_name
# ---------------------------------------------------------------------------


def test_resolve_known_template():
    assert resolve_schema_name(SchemaStubTokenizer()) == "qwen3"


def test_resolve_unknown_or_missing_template():
    assert resolve_schema_name(SchemaStubTokenizer(chat_template="{# nope #}")) is None
    assert resolve_schema_name(SchemaStubTokenizer(chat_template=None)) is None


def test_all_registered_hashes_point_at_existing_schemas():
    assert set(_TEMPLATE_HASHES.values()) <= set(RESPONSE_SCHEMAS)


# ---------------------------------------------------------------------------
# Schema parsing through HfTemplateRenderer.parse
# ---------------------------------------------------------------------------


def _parse(decoded: str, *, chat_template: str = QWEN3_TEMPLATE, tools=TOOLS) -> ParsedOutput:
    tok = SchemaStubTokenizer(decoded, chat_template=chat_template)
    return HfTemplateRenderer(tok).parse([1], tools_schema=tools)


def test_qwen3_json_tool_call_is_extracted():
    # The regression this module exists for: Qwen3 emits JSON <tool_call> blocks,
    # which the default XML regex silently missed.
    out = _parse(
        'Let me compute.\n<tool_call>\n{"name": "calculator", "arguments": {"expr": "2+2"}}\n</tool_call><|im_end|>'
    )
    assert out.tool_uses == [{"name": "calculator", "input": {"expr": "2+2"}}]
    assert out.text == "Let me compute."
    assert out.ill_formed is False


def test_reasoning_and_tool_parsed_in_one_pass():
    out = _parse(
        "<think>\nneed the tool\n</think>\nSure.\n"
        '<tool_call>\n{"name": "calculator", "arguments": {"e": "1"}}\n</tool_call><|im_end|>'
    )
    assert out.reasoning == "need the tool"
    assert out.text == "Sure."
    assert out.tool_uses == [{"name": "calculator", "input": {"e": "1"}}]


def test_tool_call_marker_inside_think_does_not_corrupt():
    # Adversarial ordering case: a literal <tool_call> inside the think block must
    # not shift the reasoning/tool split (the failure mode of strip-tools-first
    # parsers).
    out = _parse(
        "<think>\nmaybe <tool_call> here\n</think>\nvisible\n"
        '<tool_call>\n{"name": "calculator", "arguments": {"x": 1}}\n</tool_call><|im_end|>'
    )
    assert out.reasoning == "maybe <tool_call> here"
    assert out.text == "visible"
    assert out.tool_uses == [{"name": "calculator", "input": {"x": 1}}]


def test_parallel_tool_calls():
    out = _parse(
        '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n'
        '<tool_call>\n{"name": "b", "arguments": {}}\n</tool_call><|im_end|>'
    )
    assert [t["name"] for t in out.tool_uses] == ["a", "b"]


def test_plain_reply_without_tools_offered():
    out = _parse("just 4.<|im_end|>", tools=None)
    assert out.text == "just 4."
    assert out.tool_uses == []
    assert out.ill_formed is False


def test_malformed_tool_json_degrades_flagged():
    raw = '<tool_call>\n{"name": "calculator", "arguments": {oops}\n</tool_call><|im_end|>'
    out = _parse(raw)
    assert out.ill_formed is True
    assert out.tool_uses == []
    assert out.text == raw  # raw text preserved, turn not failed


def test_schema_returns_none_degrades_flagged():
    # A <tool_call> mention in plain content makes the anchored regex miss entirely.
    raw = "I would use <tool_call> syntax normally.<|im_end|>"
    out = _parse(raw)
    assert out.ill_formed is True
    assert out.text == raw


def test_missing_arguments_normalized_to_empty_dict():
    out = _parse('<tool_call>\n{"name": "calculator"}\n</tool_call><|im_end|>')
    assert out.tool_uses == [{"name": "calculator", "input": {}}]
    assert out.ill_formed is False


def test_structurally_invalid_tool_call_degrades_whole_turn():
    # Valid JSON, wrong structure: `function` parses to a list, not a dict. The
    # model DID emit a tool-call block, so extraction must not silently drop it —
    # the whole turn degrades to raw text, flagged.
    raw = "<tool_call>\n[1, 2]\n</tool_call><|im_end|>"
    out = _parse(raw)
    assert out.ill_formed is True
    assert out.tool_uses == []
    assert out.text == raw


def test_non_dict_arguments_degrades_whole_turn():
    # Double-encoded arguments (a JSON string, not an object) must not be
    # laundered into a no-arg call.
    raw = '<tool_call>\n{"name": "calculator", "arguments": "{\\"x\\": 1}"}\n</tool_call><|im_end|>'
    out = _parse(raw)
    assert out.ill_formed is True
    assert out.tool_uses == []


def test_qwen3_5_xml_format():
    # The Qwen3.5/3.6 family format: <function=NAME><parameter=KEY>VALUE</parameter>.
    out = _parse(
        "<think>\nuse it\n</think>\nOk.\n<tool_call>\n<function=calculator>\n"
        "<parameter=expr>\n12*7\n</parameter>\n</function>\n</tool_call><|im_end|>",
        chat_template=QWEN3_5_TEMPLATE,
    )
    assert out.reasoning == "use it"
    assert out.text == "Ok."
    assert out.tool_uses == [{"name": "calculator", "input": {"expr": "12*7"}}]


# ---------------------------------------------------------------------------
# Precedence and fallback
# ---------------------------------------------------------------------------


def test_injected_stage_parser_suppresses_schema_detection():
    tok = SchemaStubTokenizer('<tool_call>\n{"name": "calculator", "arguments": {}}\n</tool_call>')
    r = HfTemplateRenderer(tok, tool_parser=lambda body, tools: ("STAGE", [], False))
    out = r.parse([1], tools_schema=TOOLS)
    assert out.text == "STAGE"  # the injected parser ran; the schema path did not


def test_injected_tool_parser_is_never_rejected():
    # An explicitly injected tool parser is a deliberate configuration — no
    # schema required, tools parse fine.
    tok = SchemaStubTokenizer("anything", chat_template="{# unknown #}")
    r = HfTemplateRenderer(tok, tool_parser=lambda body, tools: (body, [], False))
    out = r.parse([1], tools_schema=TOOLS)
    assert out.ill_formed is False


def test_tools_without_schema_or_injected_parser_raises():
    # The implicit default (unrecognized template, nothing injected) refuses to
    # parse tool calls: the XML regex would silently miss other formats on every
    # rollout. Tool-free parsing on the same renderer still works.
    import pytest

    tok = SchemaStubTokenizer("plain text", chat_template="{# unknown #}")
    r = HfTemplateRenderer(tok)
    assert r.parse([1], tools_schema=None).ill_formed is False
    with pytest.raises(ValueError, match="matched no response schema"):
        r.parse([1], tools_schema=TOOLS)


def test_regex_opt_in_via_explicit_injection():
    # parse_tool_uses stays available for models that genuinely use the XML format —
    # by explicit injection, not as an implicit default.
    from agentcore_rl_toolkit.rollout_gateway.parsing import parse_tool_uses

    class XmlTok(SchemaStubTokenizer):
        def decode(self, ids, skip_special_tokens=False):
            return "<tool_call>\n<function=calculator>\n<parameter=q>7</parameter>\n</function>\n</tool_call>"

    r = HfTemplateRenderer(XmlTok(chat_template="{# unknown #}"), tool_parser=parse_tool_uses)
    out = r.parse([1], tools_schema=TOOLS)
    assert out.tool_uses == [{"name": "calculator", "input": {"q": "7"}}]
