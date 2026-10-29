"""Template derendering (transformers >= 5.13): the vendored ``response_template`` dicts
and the renderer's template path — ``prefix=`` from the generation prompt and
``tools=`` argument typing.

The parsing engine is transformers' real ``parse_response``; only decode and the
generation prompt are canned.
"""

import pytest
from transformers import PreTrainedTokenizerBase

from agentcore_rl_toolkit.rollout_gateway import response_schemas
from agentcore_rl_toolkit.rollout_gateway.render import HfTemplateRenderer, ParsedOutput
from agentcore_rl_toolkit.rollout_gateway.response_schemas import (
    _TEMPLATE_HASHES,
    RESPONSE_SCHEMAS,
    RESPONSE_TEMPLATES,
)

pytestmark = pytest.mark.skipif(
    not response_schemas.SUPPORTS_RESPONSE_TEMPLATE, reason="needs transformers >= 5.13 response templates"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}},
            },
        },
    }
]
CHATML_PROMPT = "<|im_start|>user\n.<|im_end|>\n<|im_start|>assistant\n"


class TemplateStubTokenizer:
    name_or_path = "stub"
    chat_template = None

    def __init__(self, decoded: str, prompt: str = CHATML_PROMPT, response_template: dict | None = None):
        self.decoded = decoded
        self.prompt = prompt
        if response_template is not None:
            self.response_template = response_template

    def decode(self, ids, skip_special_tokens=False):
        return self.decoded

    def apply_chat_template(self, messages, **kwargs):
        return self.prompt

    parse_response = PreTrainedTokenizerBase.parse_response


def _parse(decoded: str, *, template: str = "qwen3_5", prompt: str = CHATML_PROMPT, tools=TOOLS) -> ParsedOutput:
    # Ship the template on the tokenizer so the renderer adopts it (and its prefix) at construction.
    r = HfTemplateRenderer(TemplateStubTokenizer(decoded, prompt, response_template=RESPONSE_TEMPLATES[template]))
    return r.parse([1], tools_schema=tools)


def test_every_schema_has_a_template():
    assert set(RESPONSE_TEMPLATES) == set(RESPONSE_SCHEMAS)
    assert set(_TEMPLATE_HASHES.values()) <= set(RESPONSE_TEMPLATES)


def test_qwen3_template_parses_native_output():
    out = _parse(
        "<think>\nneed the tool\n</think>\nSure.\n"
        '<tool_call>\n{"name": "read", "arguments": {"path": "a.py"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "read", "arguments": {}}\n</tool_call><|im_end|>',
        template="qwen3",
    )
    assert out.reasoning == "need the tool"
    assert out.text == "Sure."
    assert out.tool_uses == [{"name": "read", "input": {"path": "a.py"}}, {"name": "read", "input": {}}]
    assert out.ill_formed is False


def test_qwen3_5_keeps_narration_around_calls():
    # The legacy schema rejected text between or after calls; the template keeps both.
    call = "<tool_call>\n<function=read>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>"
    out = _parse(f"<think>\nplan\n</think>\n\nLet me read.\n\n{call}\n{call}<|im_end|>")
    assert out.reasoning == "plan"
    assert out.text == "Let me read."
    assert out.tool_uses == [{"name": "read", "input": {"path": "a.py"}}] * 2
    assert out.ill_formed is False


def test_qwen3_5_reads_prefilled_think_from_the_generation_prompt():
    # Thinking templates end the prompt with "<think>\n", so the output starts mid-reasoning.
    out = _parse(
        "still thinking\n</think>\n\nDone.<|im_end|>",
        prompt=CHATML_PROMPT + "<think>\n",
    )
    assert out.reasoning == "still thinking"
    assert out.text == "Done."
    assert out.tool_uses == []


@pytest.mark.skipif(not response_schemas.SUPPORTS_PARSE_RESPONSE_TOOLS, reason="needs transformers >= 5.15")
def test_qwen3_5_types_arguments_from_the_tool_schema():
    out = _parse(
        "<tool_call>\n<function=read>\n<parameter=path>\n86\n</parameter>\n"
        "<parameter=start_line>\n86\n</parameter>\n</function>\n</tool_call><|im_end|>"
    )
    assert out.tool_uses == [{"name": "read", "input": {"path": "86", "start_line": 86}}]


def test_unknown_tool_still_parses():
    out = _parse("<tool_call>\n<function=ghost>\n<parameter=x>\n5\n</parameter>\n</function>\n</tool_call><|im_end|>")
    assert [u["name"] for u in out.tool_uses] == ["ghost"]
    assert out.ill_formed is False


def test_nemotron_3_template_parses_prefilled_reasoning():
    out = _parse(
        "plan\n</think>\n<tool_call>\n<function=read>\n<parameter=path>\na.py\n</parameter>\n"
        "</function>\n</tool_call><|im_end|>",
        template="nemotron_3",
        prompt=CHATML_PROMPT + "<think>\n",
    )
    assert out.reasoning == "plan"
    assert out.tool_uses == [{"name": "read", "input": {"path": "a.py"}}]


def test_glm4moe_template_parses_native_output():
    out = _parse(
        "<think>plan</think>\nOk.<tool_call>read\n<arg_key>path</arg_key>\n<arg_value>a.py</arg_value>\n</tool_call>",
        template="glm4moe",
        prompt="[gMASK]<sop><|user|>\n.<|assistant|>",
    )
    assert out.reasoning == "plan"
    assert out.text == "Ok."
    assert out.tool_uses == [{"name": "read", "input": {"path": "a.py"}}]


@pytest.mark.parametrize(
    "call",
    [
        # What the model generates: the recipient on the channel line, content type "code".
        '<|channel|>commentary to=functions.read code<|message|>{"path": "a.py"}<|call|>',
        # What the stock template renders: the recipient in the role header.
        ' to=functions.read<|channel|>commentary json<|message|>{"path": "a.py"}<|call|>',
    ],
)
def test_gptoss_template_parses_both_recipient_positions(call):
    out = _parse(
        "<|channel|>analysis<|message|>plan<|end|><|start|>assistant" + call,
        template="gptoss",
        prompt="<|start|>user<|message|>.<|end|><|start|>assistant",
    )
    assert out.reasoning == "plan"
    assert out.tool_uses == [{"name": "read", "input": {"path": "a.py"}}]


def test_tokenizer_response_template_takes_precedence():
    own = {
        "defaults": {"role": "assistant"},
        "start_anchor": "<|im_start|>assistant\n",
        "fields": {"content": {"close_pattern": r"<\|im_end\|>\s*", "content": "text"}},
    }
    r = HfTemplateRenderer(TemplateStubTokenizer("hello<|im_end|>", response_template=own))
    assert r._template is own
    assert r.parse([1]).text == "hello"


def test_structurally_invalid_call_degrades_whole_turn():
    raw = "<tool_call>\n[1, 2]\n</tool_call><|im_end|>"
    out = _parse(raw, template="qwen3")
    assert (out.ill_formed, out.tool_uses, out.text) == (True, [], raw)
