"""Template derendering: the vendored response templates (golden paths per template)
and ``_parse_with_template``'s degradation contract (parse failures and turns cut off
by the token limit degrade to raw text flagged ``ill_formed``, never an exception or
a silently dropped call).

The parsing engine is transformers' real tokenizer parsing; only decode is canned.
Template-hash detection is covered by test_template_hashes_live.py, so tests assign
the template directly.
"""

import pytest
from transformers import PreTrainedTokenizerBase

from agentcore_rl_toolkit.rollout_gateway.render import HfTemplateRenderer, ParsedOutput
from agentcore_rl_toolkit.rollout_gateway.response_templates import (
    _TEMPLATE_HASHES,
    RESPONSE_TEMPLATES,
    reasoning_key,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string"}, "precision": {"type": "integer"}},
            },
        },
    }
]

OUTPUT_ID = 1


class TemplateStubTokenizer:
    name_or_path = "stub"
    chat_template = None

    def __init__(self, output: str):
        self.pieces = {OUTPUT_ID: output}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces[i] for i in ids)

    parse_response = PreTrainedTokenizerBase.parse_response
    get_response_parser = PreTrainedTokenizerBase.get_response_parser


def _parse(
    output: str, *, template: str = "qwen3", prompt: str = "<|im_start|>assistant\n", finish: str = "stop"
) -> ParsedOutput:
    """Parse ``output`` as a turn whose generation prompt is ``prompt``."""
    r = HfTemplateRenderer(TemplateStubTokenizer(output))
    r._template = RESPONSE_TEMPLATES[template]
    r._parse_prefix = prompt
    r._reasoning_key = reasoning_key(template)
    return r.parse([OUTPUT_ID], finish_reason=finish, tools_schema=TOOLS)


def test_registered_hashes_point_at_existing_templates():
    assert set(_TEMPLATE_HASHES.values()) <= set(RESPONSE_TEMPLATES)


def test_qwen3_template_parses_native_output():
    # Qwen3's format: optional <think> block, then JSON tool calls in <tool_call>
    # blocks — which the dependency-free XML regex silently missed (the regression
    # that motivated template derendering).
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


def test_qwen3_5_template_reads_prefilled_think_from_prefix():
    # Qwen3.5/3.6 thinking templates prefill "<think>\n", so the output starts inside
    # the reasoning region; only the prefix tells the parser it is open.
    out = _parse(
        "use it\n</think>\nOk.\n<tool_call>\n<function=calculator>\n"
        "<parameter=expr>\n12*7\n</parameter>\n</function>\n</tool_call><|im_end|>",
        template="qwen3_5",
        prompt="<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n",
    )
    assert out.reasoning == "use it"
    assert out.text == "Ok."
    assert out.tool_uses == [{"name": "calculator", "input": {"expr": "12*7"}}]
    assert out.ill_formed is False


def test_prefilled_think_the_model_skips_parses_as_its_answer():
    # When earlier turns carry no reasoning, Qwen3.5 often skips thinking: the template
    # still prefills "<think>\n", but the model answers directly and ends its turn
    # without "</think>".
    out = _parse(
        "Let me look at the pom.xml.\n\n<tool_call>\n<function=calculator>\n"
        "<parameter=expr>\n1+1\n</parameter>\n</function>\n</tool_call><|im_end|>",
        template="qwen3_5",
        prompt="</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n",
    )
    assert (out.reasoning, out.text, out.ill_formed) == ("", "Let me look at the pom.xml.", False)
    assert out.tool_uses == [{"name": "calculator", "input": {"expr": "1+1"}}]


def test_xml_arguments_are_typed_by_the_offered_tools():
    # XML renders every argument as text; the offered tool's schema decides the type,
    # so a numeric-looking string argument stays a string and an integer is cast.
    # A tool that was not offered keeps its text arguments rather than being dropped.
    out = _parse(
        "<tool_call>\n<function=calculator>\n<parameter=expr>\n007\n</parameter>\n"
        "<parameter=precision>\n3\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=search>\n<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call><|im_end|>",
        template="qwen3_5",
    )
    assert out.tool_uses == [
        {"name": "calculator", "input": {"expr": "007", "precision": 3}},
        {"name": "search", "input": {"limit": "5"}},
    ]


@pytest.mark.parametrize(
    "value",
    ["cat <<'EOF' > f\nhi\nEOF\n", "  echo indented", "echo a\n\n", "\nleading", ""],
)
def test_qwen3_5_parameter_value_keeps_its_whitespace(value):
    # The template renders <parameter=K>\n{value}\n</parameter>; stripping more than
    # those two newlines re-renders differently from the sampled tokens.
    out = _parse(
        f"<tool_call>\n<function=calculator>\n<parameter=expr>\n{value}\n</parameter>\n"
        "</function>\n</tool_call><|im_end|>",
        template="qwen3_5",
    )
    assert out.tool_uses == [{"name": "calculator", "input": {"expr": value}}]


def test_qwen3_5_parameter_without_newlines_still_parses():
    out = _parse(
        "<tool_call>\n<function=calculator>\n<parameter=expr>1+1</parameter>"
        "<parameter=precision>3</parameter>\n</function>\n</tool_call><|im_end|>",
        template="qwen3_5",
    )
    assert out.tool_uses == [{"name": "calculator", "input": {"expr": "1+1", "precision": 3}}]


def test_qwen3_5_template_parses_qwen3_coder_output():
    out = _parse(
        "I'll list the directory.\n\n<tool_call>\n<function=shell>\n"
        "<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call><|im_end|>",
        template="qwen3_5",
    )
    assert out.reasoning == ""
    assert out.text == "I'll list the directory."
    assert out.tool_uses == [{"name": "shell", "input": {"command": "ls -la"}}]
    assert out.ill_formed is False


@pytest.mark.parametrize(
    "template,tool_call",
    [
        ("qwen3", '<tool_call>\n{"name": "calculator", "arguments": {"expr": "1+1"}}\n</tool_call>'),
        (
            "qwen3_5",
            "<tool_call>\n<function=calculator>\n<parameter=expr>\n1+1\n</parameter>\n</function>\n</tool_call>",
        ),
    ],
)
@pytest.mark.parametrize("end", ["<|im_end|>", "<|endoftext|>"])
def test_qwen_turn_ends_at_either_stop_token(template, tool_call, end):
    # Qwen generation configs stop on both tokens, and the stop token is decoded with
    # the output, so either must close the turn rather than leak into its content.
    out = _parse(f"<think>\nr\n</think>\n\nDone.{end}", template=template)
    assert (out.reasoning, out.text, out.ill_formed) == ("r", "Done.", False)
    out = _parse(f"{tool_call}{end}", template=template)
    assert (out.text, out.tool_uses) == ("", [{"name": "calculator", "input": {"expr": "1+1"}}])


@pytest.mark.parametrize(
    "reasoning,value",
    [
        ("I will emit <tool_call> next.", "1+1"),
        ("r", "grep -n '</function>' t.py"),
        ("r", "grep -n '<parameter=' t.py"),
        ("r", "echo '<tool_call>'"),
    ],
)
def test_qwen3_5_literal_tags_in_well_formed_output_stay_in_place(reasoning, value):
    # Regions close only at their own closing tag. Lookahead closers such as
    # (?=<tool_call>) or (?=</function>), as in the response template Qwen ships in
    # tokenizer_config.json, would also accept a malformed turn missing its closers,
    # but split well-formed reasoning or arguments that mention a tag literally.
    out = _parse(
        f"{reasoning}\n</think>\n\n<tool_call>\n<function=calculator>\n<parameter=expr>\n{value}\n"
        "</parameter>\n</function>\n</tool_call><|im_end|>",
        template="qwen3_5",
        prompt="<|im_start|>assistant\n<think>\n",
    )
    assert (out.reasoning, out.text) == (reasoning, "")
    assert out.tool_uses == [{"name": "calculator", "input": {"expr": value}}]


def test_gptoss_template_parses_generated_and_rendered_tool_call_headers():
    # The model puts the recipient inside the channel line with "code"; the chat
    # template renders it in the role header with "json". Both must parse, and
    # harmony's "thinking" field is read back as reasoning.
    generated = _parse(
        "<|channel|>analysis<|message|>plan<|end|><|start|>assistant<|channel|>commentary "
        'to=functions.calculator code<|message|>{"expr": "1+1"}<|call|>',
        template="gptoss",
        prompt="<|start|>assistant",
    )
    assert generated.reasoning == "plan"
    assert generated.tool_uses == [{"name": "calculator", "input": {"expr": "1+1"}}]
    assert generated.ill_formed is False

    rendered = _parse(
        ' to=functions.calculator<|channel|>commentary json<|message|>{"expr": "2"}<|call|>',
        template="gptoss",
        prompt="<|start|>assistant",
    )
    assert rendered.tool_uses == [{"name": "calculator", "input": {"expr": "2"}}]

    final = _parse(
        "<|channel|>analysis<|message|>plan<|end|><|start|>assistant<|channel|>final<|message|>4<|return|>",
        template="gptoss",
        prompt="<|start|>assistant",
    )
    assert (final.reasoning, final.text, final.ill_formed) == ("plan", "4", False)


def test_turn_cut_off_by_the_token_limit_degrades_flagged():
    # The parser would close the open region with what it holds and report a call
    # missing its arguments.
    raw = "<tool_call>\n<function=calculator>\n<parameter=expr>\n1+"
    out = _parse(raw, template="qwen3_5", finish="length")
    assert (out.ill_formed, out.tool_uses, out.text) == (True, [], raw)


def test_malformed_tool_json_degrades_flagged():
    raw = '<tool_call>\n{"name": "calculator", "arguments": {oops}\n</tool_call><|im_end|>'
    out = _parse(raw)
    assert (out.ill_formed, out.tool_uses, out.text) == (True, [], raw)


def test_unparseable_output_degrades_flagged():
    # A stray <tool_call> mention in plain content opens a tool-call region whose
    # body is not JSON — degrade, don't raise.
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
