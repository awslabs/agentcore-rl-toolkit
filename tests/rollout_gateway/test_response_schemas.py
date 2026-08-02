"""Schema derendering: the vendored schema dicts (one golden-path test per schema)
and ``_parse_with_schema``'s degradation contract (parse failures degrade to raw
text flagged ``ill_formed``, never an exception or a silently dropped call).

The parsing engine is transformers' real ``recursive_parse``; only decode is
canned. Template-hash detection is covered by test_template_hashes_live.py, so
tests assign the schema directly.
"""

from transformers.utils.chat_parsing_utils import recursive_parse

from agentcore_rl_toolkit.rollout_gateway.render import HfTemplateRenderer, ParsedOutput
from agentcore_rl_toolkit.rollout_gateway.response_schemas import _TEMPLATE_HASHES, RESPONSE_SCHEMAS

TOOLS = [{"type": "function", "function": {"name": "calculator", "parameters": {}}}]


class SchemaStubTokenizer:
    name_or_path = "stub"
    chat_template = None

    def __init__(self, decoded: str):
        self.decoded = decoded

    def decode(self, ids, skip_special_tokens=False):
        return self.decoded

    def parse_response(self, response, schema=None):
        return recursive_parse(response, schema)


def _parse(decoded: str, *, schema: str = "qwen3") -> ParsedOutput:
    r = HfTemplateRenderer(SchemaStubTokenizer(decoded))
    r._schema = RESPONSE_SCHEMAS[schema]
    return r.parse([1], tools_schema=TOOLS)


def test_registered_hashes_point_at_existing_schemas():
    assert set(_TEMPLATE_HASHES.values()) <= set(RESPONSE_SCHEMAS)


def test_qwen3_schema_parses_native_output():
    # Qwen3's format: optional <think> block, then JSON tool calls in <tool_call>
    # blocks — which the dependency-free XML regex silently missed (the regression
    # that motivated schema derendering).
    out = _parse(
        "<think>\nneed the tool\n</think>\nSure.\n"
        '<tool_call>\n{"name": "calculator", "arguments": {"expr": "2+2"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "calculator", "arguments": {}}\n</tool_call><|im_end|>'
    )
    assert out.reasoning == "need the tool"
    assert out.text == "Sure."
    assert out.tool_uses == [
        {"name": "calculator", "input": {"expr": "2+2"}},
        {"name": "calculator", "input": {}},
    ]
    assert out.ill_formed is False


def test_qwen3_5_schema_parses_native_output():
    # The Qwen3.5/3.6/Nemotron-3 format: XML <function=NAME><parameter=KEY> blocks.
    out = _parse(
        "<think>\nuse it\n</think>\nOk.\n<tool_call>\n<function=calculator>\n"
        "<parameter=expr>\n12*7\n</parameter>\n</function>\n</tool_call><|im_end|>",
        schema="qwen3_5",
    )
    assert out.reasoning == "use it"
    assert out.text == "Ok."
    assert out.tool_uses == [{"name": "calculator", "input": {"expr": "12*7"}}]
    assert out.ill_formed is False


def test_malformed_tool_json_degrades_flagged():
    raw = '<tool_call>\n{"name": "calculator", "arguments": {oops}\n</tool_call><|im_end|>'
    out = _parse(raw)
    assert (out.ill_formed, out.tool_uses, out.text) == (True, [], raw)


def test_unparseable_output_degrades_flagged():
    # A stray <tool_call> mention in plain content makes the schema's anchored
    # regex miss entirely (parse_response returns None) — degrade, don't raise.
    raw = "I would use <tool_call> syntax normally.<|im_end|>"
    out = _parse(raw)
    assert (out.ill_formed, out.tool_uses, out.text) == (True, [], raw)


def test_invalid_tool_structure_degrades_whole_turn():
    # Tool extraction is all-or-nothing: the model DID emit a tool-call block, so
    # a structural anomaly (non-dict call, double-encoded arguments) must degrade
    # the whole turn rather than silently drop or mangle the call.
    raw = "<tool_call>\n[1, 2]\n</tool_call><|im_end|>"
    out = _parse(raw)
    assert (out.ill_formed, out.tool_uses, out.text) == (True, [], raw)

    raw = '<tool_call>\n{"name": "calculator", "arguments": "{\\"x\\": 1}"}\n</tool_call><|im_end|>'
    out = _parse(raw)
    assert (out.ill_formed, out.tool_uses) == (True, [])
