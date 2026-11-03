"""Response templates for ``tokenizer.parse_response`` and chat-template detection.

The template dicts are adapted from huggingface/trl ``trl/chat_template_utils.py``
(Apache-2.0; see NOTICE) at commit ``775de52a``, with local changes marked inline.
The hashes in ``_TEMPLATE_HASHES`` are sha256 over the chat templates the model
repos ship on the Hugging Face Hub, so detection needs no template copies here;
test_template_hashes_live.py checks them against the hub. When adopting upstream TRL
changes, keep the round-trip tests there passing: parsing a template's own rendering
must recover the message exactly.

A template is pure data in transformers' region-based ``response_template`` format
(``transformers/utils/chat_parsing``): ``start_anchor`` locates the assistant turn in
the prompt passed as ``prefix=`` (so regions the chat template prefills, such as an
opening ``<think>``, are already open when generation starts), and each field is a
region delimited by open/close anchors. ``parse_response(..., tools=...)`` types
tool-call arguments from the offered tools' JSON schemas.
"""

import hashlib

# --- Response templates (adapted from TRL) -----------------------------------------

# Qwen2.5 / Qwen3 (thinking) / Qwen3-Instruct-2507 / Qwen3-VL: JSON tool calls inside
# <tool_call>...</tool_call>, optional <think> block, <|im_end|> end-of-turn.
QWEN3_TEMPLATE = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {
        "reasoning_content": {
            "open": "<think>",
            "close_pattern": r"</think>\s*",
            "content": "text",
        },
        "tool_calls": {
            "open": "<tool_call>",
            "close_pattern": r"</tool_call>\s*",
            "repeats": True,
            "content": "json",
            "transform": {"type": "function", "function": "{content}"},
        },
        "content": {
            "close_pattern": r"<\|im_end\|>\s*",
            "content": "text",
        },
    },
}

# XML tool-call formats render a string argument as raw text, so only the tool's JSON
# schema (``parse_response(..., tools=...)``) can type it; values stay text otherwise.
# Local change: TRL strips the value; kept verbatim here (see QWEN3_5_TEMPLATE).
_XML_INLINE_VALUE_PARSER = {"name": "text", "args": {"strip": False}}

# Qwen3.5 / Qwen3.6 / Qwen3.8: XML tool calls (<function=...>/<parameter=...>).
# Local change: TRL's parameter pattern strips all surrounding whitespace, but the
# template renders ``<parameter=KEY>\nVALUE\n</parameter>``, so a value with leading
# indentation or a trailing newline (e.g. a heredoc) re-renders differently from the
# sampled tokens and the next turn's prompt forks the trajectory. The pattern consumes
# only the template's own newline on each side.
QWEN3_5_TEMPLATE = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {
        "reasoning_content": {
            "open": "<think>",
            "close_pattern": r"</think>\s*",
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": r"<tool_call>\s*<function=(?P<name>[^\n>]+)>",
            "close_pattern": r"</tool_call>\s*",
            "repeats": True,
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": r"<parameter=(?P<key>[^>\n]+)>\n?(?P<value>.*?)\n?</parameter>",
                "value_parser": _XML_INLINE_VALUE_PARSER,
            },
            "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
        },
        "content": {
            "close_pattern": r"<\|im_end\|>\s*",
            "content": "text",
        },
    },
}

# GPT-OSS (harmony): reasoning/content are channels; tool calls carry the function
# name in the recipient. Local change: TRL matches only the rendered tool-call header
# (``to=functions.NAME<|channel|>commentary json``), but the model generates the
# recipient inside the channel line (``<|channel|>commentary to=functions.NAME code``;
# see the render-side repair below), so the open pattern accepts both.
GPTOSS_TEMPLATE = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|start|>assistant",
    "fields": {
        "thinking": {
            "open": "<|channel|>analysis<|message|>",
            "close_pattern": r"<\|end\|>\s*",
            "content": "text",
        },
        "content": {
            "open": "<|channel|>final<|message|>",
            "close_pattern": r"<\|return\|>\s*",
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": (
                r"(?:to=functions\.(?P<name>\S+?)<\|channel\|>commentary\s+json"
                r"|<\|channel\|>(?:commentary|analysis)\s+to=functions\.(?P<name>[a-zA-Z0-9_-]+)\s+(?:code|json))"
                r"<\|message\|>"
            ),
            "close_pattern": r"<\|call\|>\s*",
            "repeats": True,
            "content": "json",
            "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
        },
    },
}

RESPONSE_TEMPLATES: dict[str, dict] = {
    "qwen3": QWEN3_TEMPLATE,
    "qwen3_5": QWEN3_5_TEMPLATE,
    "gptoss": GPTOSS_TEMPLATE,
}

# --- Chat-template detection --------------------------------------------------------

# sha256(chat_template) -> template name, for each hub chat template known to use one
# of the templates above. Byte-exact matching (no fuzzy predicates): fine-tunes inherit
# the template verbatim and keep matching; a template a vendor has revised stops
# matching until its hash is added here — visible, never silently misparsed.
_TEMPLATE_HASHES: dict[str, str] = {
    "a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146": "gptoss",  # GPT-OSS
    "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f": "qwen3",  # Qwen2.5 / Qwen2-VL
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8": "qwen3",  # Qwen3 (thinking)
    "64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326": "qwen3",  # Qwen3-Instruct-2507
    "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4": "qwen3",  # Qwen3-VL
    "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80": "qwen3_5",  # Qwen3.5 (nothink)
    "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715": "qwen3_5",  # Qwen3.5 (think)
    "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259": "qwen3_5",  # Qwen3.6
    "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041": "qwen3_5",  # Qwen3.8
    # Qwen3-Coder uses Qwen3.5's XML tool-call syntax.
    "5a38bfa05833266240066aedc497decc9b00cc0d3e3b8cceea98cf530196ab06": "qwen3_5",  # Qwen3-Coder
}


# --- Render-side template repairs ---------------------------------------------------

# GPT-OSS renders a tool call with the recipient in the *role header* and defaults the
# content type to "json":
#     <|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>ARGS<|call|>
# The model instead generates the recipient *inside the channel line*, with "code":
#     <|start|>assistant<|channel|>commentary to=functions.NAME code<|message|>ARGS<|call|>
# (Verified at token level: the model emits a bare ` code` text token, not the harmony
# spec's `<|constrain|>` special token 200003 — that token decodes visibly and never
# appeared in captured rollouts.)
#
# Rendering a replayed history therefore does not reproduce the tokens that were
# sampled, so the next turn's prompt stops extending the captured sequence and the
# trajectory manager has to FORK (emitting an extra row). Reordering the render to
# match generation restores the prefix.
#
# Applies to the render path only; GPTOSS_TEMPLATE accepts both headers when parsing.
_GPTOSS_TOOLCALL_RENDER_ORIG = (
    '            {{- "<|start|>assistant to=" }}\n'
    '            {{- "functions." + tool_call.name + "<|channel|>commentary " }}\n'
    '            {{- (tool_call.content_type if tool_call.content_type is defined else "json") + "<|message|>" }}'
)
_GPTOSS_TOOLCALL_RENDER_FIXED = (
    '            {{- "<|start|>assistant<|channel|>commentary to=" }}\n'
    '            {{- "functions." + tool_call.name + " " }}\n'
    '            {{- (tool_call.content_type if tool_call.content_type is defined else "code") + "<|message|>" }}'
)


# Canonical messages carry reasoning under ``reasoning_content`` (both adapters
# normalize to it). Templates disagree on the key: Qwen3 reads ``reasoning_content``
# directly, while GPT-OSS/harmony reads ``thinking`` and silently ignores anything
# else — so replayed reasoning renders as nothing and the prompt stops reproducing
# the sampled tokens. The renderer renames at the render seam for templates that
# need it, and reads parsed reasoning back from the same key.
_REASONING_KEY: dict[str, str] = {"gptoss": "thinking"}


def reasoning_key(template_name: str | None) -> str | None:
    """Template-specific key for assistant reasoning, or None for ``reasoning_content``."""
    return _REASONING_KEY.get(template_name or "")


def patch_chat_template(template: str, template_name: str | None) -> str:
    """Return ``template`` with render-side repairs applied, or unchanged.

    Only rewrites a block that is present exactly once, so a template revised upstream
    silently keeps its own rendering rather than being half-patched.
    """
    if template_name != "gptoss" or template.count(_GPTOSS_TOOLCALL_RENDER_ORIG) != 1:
        return template
    return template.replace(_GPTOSS_TOOLCALL_RENDER_ORIG, _GPTOSS_TOOLCALL_RENDER_FIXED)


def resolve_template_name(tokenizer) -> str | None:
    """Map a tokenizer to a response-template name by hashing its chat template.

    Returns ``None`` when the template is absent or not recognized — callers fall
    back to their existing parsing.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        return None
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    return _TEMPLATE_HASHES.get(digest)
