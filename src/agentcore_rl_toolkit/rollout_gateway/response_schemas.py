"""Response schemas for ``tokenizer.parse_response`` and chat-template detection.

The schema dicts and the template→schema mapping are vendored from huggingface/trl
``trl/chat_template_utils.py`` (Apache-2.0; see NOTICE), baseline commit ``7073af94``.
The hashes in ``_TEMPLATE_HASHES`` are sha256 over the bundled chat templates in
``trl/chat_templates/*.jinja`` at that commit — the same byte-exact template
identity TRL's ``add_response_schema`` matches on, encoded as hashes so no template
copies need to ship here. Re-sync workflow: diff the upstream file against the
baseline commit, update the dicts/hashes, and bump the commit here and in NOTICE.

A schema is pure data: the parsing engine is transformers'
``PreTrainedTokenizerBase.parse_response`` (``recursive_parse`` in
``transformers/utils/chat_parsing_utils.py``), which extracts ``reasoning_content``,
``content``, and ``tool_calls`` in one pass, ordered by the schema's anchored regex —
so a literal ``<tool_call>`` inside a think block never corrupts the split. Note for
schema authors: transformers >= 5.13 replaces this legacy ``response_schema`` format
with the region-based ``response_template`` format (and a ``prefix=`` argument);
model repos are expected to eventually ship those in ``tokenizer_config.json``, at
which point this vendored registry becomes a fallback for models that haven't.
"""

import hashlib

# --- Schema dicts (vendored verbatim from TRL) -------------------------------------

# Qwen2.5 / Qwen3 (thinking) / Qwen3-Instruct-2507 / Qwen3-VL: JSON tool calls inside
# <tool_call>...</tool_call>, optional <think> block, <|im_end|> end-of-turn.
QWEN3_SCHEMA = {
    "x-regex": r"^(?:<think>\n?(?:(?P<reasoning_content>.*?\S.*?)\n?|[\s]*)</think>\s*)?(?P<content>(?:(?!<tool_call>)[\s\S])*?)(?:\n(?=<tool_call>))?(?=(?:<tool_call>|<\|im_end\|>|$))(?P<tool_calls>(?:<tool_call>(?:(?!</tool_call>)[\s\S])+</tool_call>\s*)+)?\s*(?:<\|im_end\|>|$)",  # noqa: E501
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string"},
        "reasoning_content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "x-regex-iterator": r"<tool_call>\s*(.+?)\s*</tool_call>",
            "items": {
                "x-parser": "json",
                "x-parser-args": {"transform": "{type: 'function', function: @}"},
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {
                                "type": "object",
                                "additionalProperties": {},
                            },
                        },
                    },
                },
            },
        },
    },
}

# Qwen3.5 / Qwen3.6 / Nemotron-3: XML tool calls
# (<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>).
QWEN3_5_SCHEMA = {
    "x-regex": r"^(?:(?:<think>\n?)?(?:(?P<reasoning_content>.*?\S.*?)\n?|[\s]*)</think>\s*)?(?P<content>(?:(?!<tool_call>)[\s\S])*?)(?:\n+(?=<tool_call>))?(?=(?:<tool_call>|<\|im_end\|>|$))(?P<tool_calls>(?:<tool_call>(?:(?!</tool_call>)[\s\S])+</tool_call>\s*)+)?\s*(?:<\|im_end\|>|$)",  # noqa: E501
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string"},
        "reasoning_content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "x-regex-iterator": r"<tool_call>\s*(.+?)\s*</tool_call>",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "x-regex": r"<function=([^\n>]+)>"},
                            "arguments": {
                                "type": "object",
                                "x-regex-key-value": r"<parameter=(?P<key>[^>\n]+)>\n(?P<value>.*?)\n</parameter>",
                                "default": {},
                                "additionalProperties": {
                                    "x-parser": "json",
                                    "x-parser-args": {"allow_non_json": True},
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}

# GLM4-MoE: tool name on the <tool_call> line, arguments as <arg_key>/<arg_value> pairs.
GLM4MOE_SCHEMA = {
    "x-regex": r"^(?:\n?<think>\n?(?:(?P<reasoning_content>.*?\S.*?)\n?|[\s]*)</think>\s*)?(?P<content>(?:(?!<tool_call>)[\s\S])*?)(?:\n(?=<tool_call>))?(?=(?:<tool_call>|$))(?P<tool_calls>(?:<tool_call>(?:(?!</tool_call>)[\s\S])+</tool_call>\s*)+)?$",  # noqa: E501
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string"},
        "reasoning_content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "x-regex-iterator": r"<tool_call>\s*(.+?)\s*</tool_call>",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "x-regex": r"^(\S+)"},
                            "arguments": {
                                "type": "object",
                                "x-regex-key-value": r"<arg_key>(?P<key>[^<]+)</arg_key>\s*\n<arg_value>(?P<value>.*?)</arg_value>",  # noqa: E501
                                "default": {},
                                "additionalProperties": {
                                    "x-parser": "json",
                                    "x-parser-args": {"allow_non_json": True},
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}

# GPT-OSS (harmony): reasoning/content are channels, tool calls carry the function
GPTOSS_SCHEMA = {
    "x-regex-substitutions": [
        [r"<\|channel\|>final<\|message\|>(.*?)<\|return\|>", r"<|channel|>analysis<|message|>\1<|end|>"],
    ],
    "x-regex": r"^(?:<\|channel\|>analysis<\|message\|>(?P<reasoning_content>.*?)<\|end\|>(?:<\|start\|>assistant)?\s*(?=<\|channel\|>))?(?:<\|channel\|>analysis<\|message\|>(?P<content>.*?)<\|end\|>)?\s*(?P<tool_calls>(?:<\|channel\|>(?:commentary|analysis)\s+to=functions\.[a-zA-Z0-9_-]+\s+(?:code|json)<\|message\|>.*?<\|call\|>.*?(?:<\|end\|>|$)\s*)+)?$",  # noqa: E501
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string"},
        "reasoning_content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "x-regex-iterator": r"(<\|channel\|>(?:commentary|analysis)\s+to=functions\.[a-zA-Z0-9_-]+\s+(?:code|json)<\|message\|>.*?<\|call\|>)",  # noqa: E501
            "items": {
                "x-regex-substitutions": [
                    [
                        r"<\|channel\|>(?:commentary|analysis)\s+to=functions\.([a-zA-Z0-9_-]+)\s+(?:code|json)<\|message\|>(.*?)<\|call\|>",
                        r'{"name": "\1", "arguments": \2}',
                    ],
                ],
                "x-parser": "json",
                "x-parser-args": {"transform": "{type: 'function', function: @}"},
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {
                                "type": "object",
                                "additionalProperties": {},
                            },
                        },
                    },
                },
            },
        },
    },
}

RESPONSE_SCHEMAS: dict[str, dict] = {
    "qwen3": QWEN3_SCHEMA,
    "qwen3_5": QWEN3_5_SCHEMA,
    "glm4moe": GLM4MOE_SCHEMA,
    "gptoss": GPTOSS_SCHEMA,
}

# --- Chat-template detection --------------------------------------------------------

# sha256(chat_template) -> schema name, for every chat template TRL maps to one of
# the schemas above. Byte-exact matching (no fuzzy predicates): fine-tunes inherit
# the template verbatim and keep matching; a template a vendor has revised stops
# matching until its hash is added here — visible, never silently misparsed.
_TEMPLATE_HASHES: dict[str, str] = {
    "44f815868bf02fa458dd2f741a338046f4bf45f398eb6d067766726b9d96cce3": "glm4moe",  # GLM4-MoE
    "a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146": "gptoss",  # GPT-OSS
    "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f": "qwen3",  # Qwen2.5 / Qwen2-VL
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8": "qwen3",  # Qwen3 (thinking)
    "64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326": "qwen3",  # Qwen3-Instruct-2507
    "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4": "qwen3",  # Qwen3-VL
    "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80": "qwen3_5",  # Qwen3.5 (nothink)
    "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715": "qwen3_5",  # Qwen3.5 (think)
    "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259": "qwen3_5",  # Qwen3.6
    "ab7813c3abdd9cb655905a410728b26c7884eca45ddfab8d9f931553485a7862": "qwen3_5",  # Nemotron-3 Nano
    "575fb74f54ed264df9047d0ecce3c98938aae953fb4f50356675706264cbb68a": "qwen3_5",  # Nemotron-3 Super
    "82753bef5cedc4932c1ed509b5c9a12be680fd86d1adb65bc3f7398d11c8eebc": "qwen3_5",  # Nemotron-3 Ultra
    # Local additions without TRL equivalents; keep below the vendored entries.
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
# trajectory manager has to REALIGN (dropping the turn's loss mask) or FORK (emitting
# an extra row). Reordering the render to match generation restores the prefix.
#
# Applies to the render path only; the parse path is handled by GPTOSS_SCHEMA.
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
# the sampled tokens. Rename at the render seam for templates that need it.
_REASONING_RENDER_KEY: dict[str, str] = {"gptoss": "thinking"}


def reasoning_render_key(schema_name: str | None) -> str | None:
    """Template-specific key for assistant reasoning, or None to leave messages alone."""
    return _REASONING_RENDER_KEY.get(schema_name or "")


def patch_chat_template(template: str, schema_name: str | None) -> str:
    """Return ``template`` with render-side repairs applied, or unchanged.

    Only rewrites a block that is present exactly once, so a template revised upstream
    silently keeps its own rendering rather than being half-patched.
    """
    if schema_name != "gptoss" or template.count(_GPTOSS_TOOLCALL_RENDER_ORIG) != 1:
        return template
    return template.replace(_GPTOSS_TOOLCALL_RENDER_ORIG, _GPTOSS_TOOLCALL_RENDER_FIXED)


def resolve_schema_name(tokenizer) -> str | None:
    """Map a tokenizer to a response-schema name by hashing its chat template.

    Returns ``None`` when the template is absent or not recognized — callers fall
    back to their existing parsing.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        return None
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    return _TEMPLATE_HASHES.get(digest)
