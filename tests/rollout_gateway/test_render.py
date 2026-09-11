"""HfTemplateRenderer round-trip tests.

Rendering uses a locally built HF/Rust tokenizer, compared directly with HF's
apply_chat_template. Parsing uses small stubs. Neither needs model downloads.
The schema-derender path (recognized chat template -> tokenizer.parse_response) is
covered in test_response_schemas.py; this module covers the two-stage fallback.
"""

import asyncio
import hashlib
import sys
from unittest.mock import patch

import pytest

from agentcore_rl_toolkit.rollout_gateway.parsing import parse_tool_uses
from agentcore_rl_toolkit.rollout_gateway.render import HfTemplateRenderer, ParsedOutput


class StubTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids)


@pytest.mark.asyncio
async def test_render_passes_expected_kwargs_and_returns_list(fast_tokenizer):
    tok = fast_tokenizer
    r = HfTemplateRenderer(tok)
    messages = [{"role": "user", "content": "abc"}]
    expected = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    with patch.object(tok, "apply_chat_template", wraps=tok.apply_chat_template) as template:
        ids = await r.render(messages, tools=None, add_generation_prompt=True)
    assert isinstance(ids, list) and ids == expected
    template.assert_called_once_with(
        messages, tools=None, tokenize=False, add_generation_prompt=True, return_dict=False
    )


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


@pytest.mark.asyncio
async def test_render_forwards_chat_template_kwargs_renderer_default_and_per_call(fast_tokenizer):
    tok = fast_tokenizer
    r = HfTemplateRenderer(tok, chat_template_kwargs={"enable_thinking": False})
    with patch.object(tok, "apply_chat_template", wraps=tok.apply_chat_template) as template:
        await r.render([{"role": "user", "content": "abc"}])
        assert template.call_args.kwargs["enable_thinking"] is False
        # Request variables override renderer defaults.
        await r.render([{"role": "user", "content": "abc"}], chat_template_kwargs={"enable_thinking": True, "x": 1})
        assert template.call_args.kwargs["enable_thinking"] is True
        assert template.call_args.kwargs["x"] == 1
        await HfTemplateRenderer(tok).render([{"role": "user", "content": "abc"}])
        assert "enable_thinking" not in template.call_args.kwargs
        assert "x" not in template.call_args.kwargs


@pytest.fixture
def fast_tokenizer():
    """Actual HF/Rust tokenizer, built locally without model downloads."""
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>", "<extra>"],
        chat_template=(
            "{% for m in messages %}{{ '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}"
            "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}"
            "{% if enable_thinking | default(false) %}{{ '<think>\\n' }}{% endif %}{% endif %}"
        ),
    )


@pytest.mark.asyncio
async def test_native_encoding_matches_hf(fast_tokenizer):
    tok = fast_tokenizer
    renderer = HfTemplateRenderer(tok, chat_template_kwargs={"enable_thinking": False})
    messages = [{"role": "user", "content": "中文🙂 e\u0301 <extra>\r\n" * 100}]
    expected = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False, enable_thinking=True
    )
    outputs = await asyncio.gather(
        *[renderer.render(messages, chat_template_kwargs={"enable_thinking": True}) for _ in range(16)]
    )
    assert all(ids == expected for ids in outputs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {
            "padding": False,
            "truncation": False,
            "continue_final_message": False,
            "return_assistant_tokens_mask": False,
            "return_tensors": None,
            "tokenizer_kwargs": {"return_overflowing_tokens": False},
        },
        {"max_length": 8},  # HF ignores max_length when truncation/padding are off.
        {"truncation": True, "max_length": 16},
        {"truncation": "only_first", "max_length": 16},
        {"padding": "max_length", "max_length": 256, "tokenizer_kwargs": {"padding_side": "left"}},
        {"padding": True, "tokenizer_kwargs": {"pad_to_multiple_of": 64}},
        {"tokenizer_kwargs": {"split_special_tokens": True}},
        {"tokenizer_kwargs": {"text_pair": "another sequence"}},
        {"continue_final_message": True},
        {"continue_final_message": "content"},
    ],
)
async def test_async_render_preserves_hf_options(fast_tokenizer, options):
    renderer = HfTemplateRenderer(fast_tokenizer)
    messages = [{"role": "user", "content": "中文 <extra> question"}, {"role": "assistant", "content": "prefill"}]
    kwargs = {"add_generation_prompt": not options.get("continue_final_message"), "chat_template_kwargs": options}
    expected = fast_tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=kwargs["add_generation_prompt"], return_dict=False, **options
    )
    actual = await renderer.render(messages, **kwargs)
    assert actual == expected
    assert all(isinstance(token_id, int) for token_id in actual)


@pytest.mark.asyncio
async def test_concurrent_async_render_uses_each_requests_hf_options(fast_tokenizer):
    renderer = HfTemplateRenderer(fast_tokenizer, chat_template_kwargs={"max_length": 16, "truncation": True})
    messages = [{"role": "user", "content": "中文 <extra>" * 20}]
    options = [
        {},
        {"max_length": 8},
        {"truncation": False},
        {"padding": "max_length", "max_length": 512},
        {"tokenizer_kwargs": {"split_special_tokens": True}},
    ] * 4
    expected = [
        fast_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            **{"max_length": 16, "truncation": True, **kw},
        )
        for kw in options
    ]
    actual = await asyncio.gather(*(renderer.render(messages, chat_template_kwargs=kw) for kw in options))
    assert actual == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"continue_final_message": True},  # incompatible with add_generation_prompt
        {"return_assistant_tokens_mask": True},  # renderer returns IDs, not a dict
        {"padding": True, "truncation": True, "max_length": 15, "tokenizer_kwargs": {"pad_to_multiple_of": 8}},
    ],
)
async def test_async_render_preserves_hf_validation(fast_tokenizer, options):
    renderer = HfTemplateRenderer(fast_tokenizer)
    messages = [{"role": "assistant", "content": "prefill"}]
    with pytest.raises(ValueError) as expected:
        fast_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False, **options
        )
    with pytest.raises(ValueError) as actual:
        await renderer.render(messages, chat_template_kwargs=options)
    assert str(actual.value) == str(expected.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["", "<|im_end|>", "<|im_end|>\n", "<|endoftext|>"])
async def test_incremental_tokens_equal_full_healing_and_preserve_sampled_prefix(fast_tokenizer, monkeypatch, ending):
    from agentcore_rl_toolkit.rollout_gateway import render
    from agentcore_rl_toolkit.rollout_gateway.linear import LinearHealer

    # This tiny template has the same tested closer contract. Production admission
    # is hash-gated; the actual Qwen template/BPE is also exercised in local replay.
    monkeypatch.setattr(
        render, "_QWEN_CODER_TEMPLATE", hashlib.sha256(fast_tokenizer.chat_template.encode()).hexdigest()
    )
    incremental = HfTemplateRenderer(fast_tokenizer)
    full = HfTemplateRenderer(fast_tokenizer)
    full.render_delta = None
    prior = [{"role": "user", "content": "old context" * 100}]
    last = {"role": "assistant", "content": "Client-replayed text"}
    new = [{"role": "tool", "content": "中文 <extra>"}, {"role": "tool", "content": "second result"}]
    seed = fast_tokenizer.apply_chat_template(prior, tokenize=True, add_generation_prompt=True, return_dict=False)
    output = fast_tokenizer.encode("Raw  sampled tokens." + ending, add_special_tokens=False)
    hs = [LinearHealer(r) for r in (incremental, full)]
    for h in hs:
        h.commit("s", fed_prompt_ids=seed, output_ids=output, messages=prior, response_message=last, tools=None)

    # Normal append must not call full render on the incremental renderer.
    async def no_full_render(*args, **kwargs):
        raise AssertionError("re-encoded full history")

    incremental.render = no_full_render
    actual = await hs[0].heal("s", prior + [last] + new, None)
    assert actual == await hs[1].heal("s", prior + [last] + new, None)
    assert actual[: len(seed) + len(output)] == seed + output


@pytest.mark.asyncio
async def test_unknown_template_does_not_use_parser_schema_as_incremental_contract(fast_tokenizer):
    renderer = HfTemplateRenderer(fast_tokenizer)
    renderer._schema = {"same": "parser"}
    assert await renderer.render_delta({"role": "assistant", "content": "x"}, []) is None
