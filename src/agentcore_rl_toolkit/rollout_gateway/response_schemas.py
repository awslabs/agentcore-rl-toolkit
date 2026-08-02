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
# name in the channel header.
GPTOSS_SCHEMA = {
    # Normalize final content to analysis format so both map to the same "content" group.
    "x-regex-substitutions": [
        [r"<\|channel\|>final<\|message\|>(.*?)<\|return\|>", r"<|channel|>analysis<|message|>\1<|end|>"],
    ],
    "x-regex": r"^(?:<\|channel\|>analysis<\|message\|>(?P<content>.*?)<\|end\|>(?:<\|start\|>assistant)?)?\s*(?P<tool_calls>to=functions\.\S+<\|channel\|>commentary json<\|message\|>.*?<\|call\|>)?$",  # noqa: E501
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "x-regex-iterator": r"(to=functions\.\S+<\|channel\|>commentary json<\|message\|>.*?<\|call\|>)",
            "items": {
                # Convert "to=functions.NAME<|channel|>commentary json<|message|>ARGS<|call|>"
                # into '{"name": "NAME", "arguments": ARGS}' so it can be parsed as JSON.
                "x-regex-substitutions": [
                    [
                        r"to=functions\.(\S+)<\|channel\|>commentary json<\|message\|>(.*?)<\|call\|>",
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
}


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
