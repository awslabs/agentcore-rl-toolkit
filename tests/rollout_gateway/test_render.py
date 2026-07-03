"""HfTemplateRenderer round-trip tests.

Uses a tiny stub tokenizer (no transformers download) to assert the renderer calls
apply_chat_template with the expected kwargs and decodes/parses output correctly. A
real-tokenizer parity check lives in the Step 2 E2E path (against the live server).
"""

from agentcore_rl_toolkit.rollout_gateway.render import HfTemplateRenderer, ParsedOutput


class StubTokenizer:
    def __init__(self):
        self.last_kwargs = None

    def apply_chat_template(self, messages, *, tools=None, tokenize=True, add_generation_prompt=True):
        self.last_kwargs = dict(tools=tools, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
        # deterministic: one id per message, +99 sentinel for the generation prompt
        ids = [len(m.get("content") or "") for m in messages]
        if add_generation_prompt:
            ids.append(99)
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids)


def test_render_passes_expected_kwargs_and_returns_list():
    tok = StubTokenizer()
    r = HfTemplateRenderer(tok)
    ids = r.render([{"role": "user", "content": "abc"}], tools=None, add_generation_prompt=True)
    assert ids == [3, 99]
    assert tok.last_kwargs == {"tools": None, "tokenize": True, "add_generation_prompt": True}


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
    engine (sglang/vllm) imported."""
    import sys

    # decode returns the raw XML tool-call text
    class XmlTok(StubTokenizer):
        def decode(self, ids, skip_special_tokens=False):
            return "<tool_call>\n<function=search>\n<parameter=q>cats</parameter>\n</function>\n</tool_call>"

    r = HfTemplateRenderer(XmlTok())
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
