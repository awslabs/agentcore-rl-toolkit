"""Tokenization for the rollout gateway.

The gateway owns tokenization: it renders canonical chat messages (+ tools) into
``token_ids`` and derenders sampled ``token_ids`` back into text / reasoning / tool
calls. Owning both directions in one place makes loss-masking well-defined and
eliminates cross-backend retokenization drift. It is also required for sample-only
inference backends (e.g. Tinker) that cannot render themselves.

Two implementations behind one :class:`Renderer` protocol:

* :class:`HfTemplateRenderer` (default, lightweight) — renders with the HF tokenizer's
  ``apply_chat_template``. Derendering picks the strongest available path: when the
  tokenizer's chat template is recognized (:func:`response_schemas.resolve_schema_name`),
  the whole output is parsed in one pass via ``tokenizer.parse_response`` with the
  matching response schema — reasoning, text, and tool calls together, in the model's
  actual format. Otherwise it falls back to two stages — a reasoning parser then a
  tool parser — each independently overridable via ``reasoning_parser`` /
  ``tool_parser`` and each defaulting to the dependency-free implementation
  (``</think>`` split / ``<tool_call><function=...>`` regex). Passing either stage
  parser disables schema detection: explicit injection (e.g. slime supplying SGLang's
  detectors) always wins. Tool-bearing requests with no matched schema and no
  injected ``tool_parser`` are rejected rather than guessed at. Needs only
  ``transformers``.
* :class:`TinkerRenderer` — wraps a tinker-cookbook ``Renderer`` (install ``tinker``
  and ``tinker-cookbook`` manually; they require Python >=3.11). Its ``tinker_cookbook``
  import (which pulls torch) is deferred into ``__init__``, so importing this module
  stays torch-free; the heavy import fires only when a ``TinkerRenderer`` is constructed.
"""

import dataclasses
import logging
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from .parsing import split_reasoning
from .response_schemas import RESPONSE_SCHEMAS, resolve_schema_name

logger = logging.getLogger(__name__)

# The two derender stages, as injectable callables:
#   ReasoningParser: raw_output -> (reasoning, body_text)
#   ToolParser     : (body_text, tools_schema) -> (text, tool_uses, ill_formed)
ReasoningParserFn = Callable[[str], tuple[str, str]]
ToolParserFn = Callable[[str, list[dict]], tuple[str, list[dict[str, Any]], bool]]


@dataclasses.dataclass(frozen=True)
class ParsedOutput:
    """Derender result: what the model produced this turn, protocol-agnostic.

    ``ill_formed`` flags a parse that could not cleanly terminate / decode.
    """

    reasoning: str
    text: str
    tool_uses: list[dict[str, Any]]
    ill_formed: bool = False


@runtime_checkable
class Renderer(Protocol):
    """The gateway's tokenization seam.

    ``render``             : canonical chat messages (+ tools) -> prompt ``token_ids``
    ``get_stop_sequences`` : stop strings / token ids for sampling
    ``parse``              : sampled response ``token_ids`` -> :class:`ParsedOutput`
    """

    def render(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        ...

    def get_stop_sequences(self) -> list[str] | list[int]:
        ...

    def parse(
        self,
        output_ids: list[int],
        *,
        tools_schema: list[dict] | None = None,
    ) -> ParsedOutput:
        ...


class HfTemplateRenderer:
    """Default renderer: HF ``apply_chat_template`` for rendering, schema-or-stages
    for derendering.

    Depends only on a HF tokenizer (``transformers``). Derendering:

    * **Schema path (default when the chat template is recognized).** The tokenizer's
      chat template is hashed against the vendored registry in ``response_schemas``;
      on a match, ``tokenizer.parse_response(raw_text, schema=...)`` extracts
      reasoning, text, and tool calls in one pass, in the model family's actual
      output format (e.g. Qwen3's JSON ``<tool_call>``). A parse failure degrades in
      place — raw text is returned with ``ill_formed=True`` — never an exception.
    * **Two-stage path (fallback, and the explicit-injection seam).** Reasoning first,
      then tool calls on what remains, each independently overridable and each
      defaulting to the dependency-free implementation in ``parsing``:

      - ``reasoning_parser``: ``raw_output -> (reasoning, body_text)``.
        Default: split on ``</think>``.
      - ``tool_parser``: ``(body_text, tools_schema) -> (text, tool_uses, ill_formed)``,
        called only when the request carries a tools schema. There is no implicit
        default: with no matched schema and no injected ``tool_parser``, a
        tools-bearing ``parse`` raises — the XML regex (``parsing.parse_tool_uses``)
        understands one format and would silently miss every other, so using it is
        an explicit opt-in (``tool_parser=parse_tool_uses``).

      Passing either parser disables schema detection entirely: explicit injection
      always wins (see ``backends.slime.integration.sglang_parsing``, which supplies
      SGLang's detectors without this package importing an engine).
    """

    def __init__(
        self,
        tokenizer,
        *,
        stop_sequences: list[str] | list[int] | None = None,
        reasoning_parser: ReasoningParserFn | None = None,
        tool_parser: ToolParserFn | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self._stop_sequences: list = list(stop_sequences) if stop_sequences else []
        self.reasoning_parser: ReasoningParserFn = reasoning_parser or split_reasoning
        # No implicit tool parser: None means tools-bearing requests are rejected at
        # parse time unless a response schema matched. The reasoning stage keeps a
        # default (</think> split) because its no-marker case is the identity
        # function — harmless for non-reasoning models — whereas a wrong tool parser
        # extracts nothing while reporting success, silently, on every rollout (the
        # failure mode that produced a plausible-looking no-tool training run).
        # parsing.parse_tool_uses (the <tool_call><function=...> XML regex) remains
        # available by explicit injection.
        self.tool_parser: ToolParserFn | None = tool_parser
        self._schema: dict | None = None
        if reasoning_parser is None and tool_parser is None:
            # tokenizer.parse_response is guaranteed by the transformers>=5.0 floor
            # (the [gateway] extra); no availability guard needed.
            schema_name = resolve_schema_name(tokenizer)
            if schema_name is not None:
                self._schema = RESPONSE_SCHEMAS[schema_name]

    def render(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        # return_dict=False: we want only the token ids. The dict form (the
        # transformers>=5 default) bundles an attention mask, but that is a padding
        # artifact the training backend builds itself when it batches rows.
        ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=False,
        )
        return list(ids)

    def get_stop_sequences(self) -> list[str] | list[int]:
        return list(self._stop_sequences)

    def parse(
        self,
        output_ids: list[int],
        *,
        tools_schema: list[dict] | None = None,
    ) -> ParsedOutput:
        raw_output = self.tokenizer.decode(output_ids, skip_special_tokens=False) if output_ids else ""
        if self._schema is not None:
            return self._parse_with_schema(raw_output)
        if tools_schema and self.tool_parser is None:
            raise ValueError(
                f"tools offered but the chat template of "
                f"{getattr(self.tokenizer, 'name_or_path', '<tokenizer>')!r} matched no "
                "response schema, and no tool_parser was injected — there is no "
                "implicit tool parser to guess with. Fix: add this template's hash to "
                "rollout_gateway/response_schemas.py, or inject an explicit "
                "tool_parser (parsing.parse_tool_uses for the "
                "<tool_call><function=...> XML format)."
            )
        reasoning, body_text = self.reasoning_parser(raw_output)
        tool_uses: list[dict[str, Any]] = []
        ill_formed = False
        if tools_schema and self.tool_parser is not None:
            body_text, tool_uses, ill_formed = self.tool_parser(body_text, tools_schema)
        return ParsedOutput(
            reasoning=(reasoning or "").strip(),
            text=(body_text or "").strip(),
            tool_uses=tool_uses,
            ill_formed=ill_formed,
        )

    def _parse_with_schema(self, raw_output: str) -> ParsedOutput:
        # The tools schema is deliberately not passed: response schemas describe the
        # model's output format, not the offered tools, so parsing never filters by
        # tool name — a hallucinated tool flows through for the agent framework to
        # reject. parse_response also accepts token ids, but only to decode them
        # internally (parsing is text-level); decoding ourselves keeps a single
        # decode with our flags on every path, including the ill_formed fallback.
        # The >=5.13 template API adds a `prefix=` (prompt) argument for
        # template-prefilled regions; adopting it will need prompt ids at this seam.
        try:
            parsed = self.tokenizer.parse_response(raw_output, schema=self._schema)
        except Exception as e:
            logger.warning("response-schema parse failed (%s); degrading to raw text", e)
            parsed = None
        if not isinstance(parsed, dict):
            # None = the schema's anchored regex did not match (e.g. output truncated
            # mid-tool-call). Degrade in place, flagged — never fail the turn.
            return ParsedOutput(reasoning="", text=raw_output, tool_uses=[], ill_formed=True)
        # Tool-call extraction is all-or-nothing (TRL's convention): missing/None
        # arguments are a valid no-arg call and normalize to {}, but any structural
        # anomaly — non-dict call/function, missing name, non-dict arguments —
        # degrades the WHOLE turn to raw text. Partial extraction would silently
        # drop or mangle a call the model did emit.
        tool_uses = []
        for call in parsed.get("tool_calls") or []:
            fn = call.get("function") if isinstance(call, dict) else None
            args = (fn.get("arguments") if isinstance(fn, dict) else None) or {}
            if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not isinstance(args, dict):
                logger.warning("structurally invalid tool call in schema parse; degrading turn to raw text")
                return ParsedOutput(reasoning="", text=raw_output, tool_uses=[], ill_formed=True)
            tool_uses.append({"name": fn["name"], "input": args})
        return ParsedOutput(
            reasoning=parsed.get("reasoning_content") or "",
            text=parsed.get("content") or "",
            tool_uses=tool_uses,
            ill_formed=False,
        )


class TinkerRenderer:
    """Wrap a tinker-cookbook ``Renderer`` (built via its ``get_renderer`` factory).

    Required for the Tinker sampling backend (Tinker is sample-only and cannot render
    itself), and usable with any backend. The ``tinker_cookbook`` import is deferred
    into ``__init__`` so importing ``render`` stays torch-free; constructing a
    ``TinkerRenderer`` is what pulls tinker + torch (install ``tinker`` and
    ``tinker-cookbook`` manually; they require Python >=3.11).

    Maps the tinker-cookbook API onto the :class:`Renderer` protocol:
    - ``render``             -> ``build_generation_prompt(messages).to_ints()`` (+ tool prefix)
    - ``get_stop_sequences`` -> passthrough
    - ``parse``              -> ``parse_response(ids)`` -> :class:`ParsedOutput`
      (``ParseTermination.MALFORMED`` -> ``ill_formed=True``)
    """

    def __init__(self, model_name: str, *, renderer_name: str | None = None, tokenizer: Any = None) -> None:
        from tinker_cookbook import renderers, tokenizer_utils

        self.model_name = model_name
        tok = tokenizer if tokenizer is not None else tokenizer_utils.get_tokenizer(model_name)
        self.tokenizer = tok
        name = renderer_name or self._default_renderer_name(model_name)
        self._renderer = renderers.get_renderer(name, tok)

    @staticmethod
    def _default_renderer_name(model_name: str) -> str:
        m = model_name.lower()
        if "qwen3" in m or "qwen-3" in m:
            return "qwen3"
        if "llama-3" in m or "llama3" in m:
            return "llama3"
        if "deepseek" in m:
            return "deepseekv3"
        # Fall back to qwen3 for the common instruct case; caller can pass renderer_name.
        return "qwen3"

    def render(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        msgs = list(messages)
        if tools:
            # tinker-cookbook tool schemas are ToolSpec dicts {name, description, parameters};
            # our tools_schema is OpenAI shape {"function": {name, description, parameters}}.
            tool_specs = [self._to_tool_spec(t) for t in tools]
            prefix = self._renderer.create_conversation_prefix_with_tools(tool_specs)
            msgs = list(prefix) + msgs
        if add_generation_prompt:
            model_input = self._renderer.build_generation_prompt(msgs)
        else:
            # supervised-style render of the full conversation without the trailing
            # generation header; build_supervised_example returns (ModelInput, weights).
            model_input, _ = self._renderer.build_supervised_example(msgs)
        return list(model_input.to_ints())

    @staticmethod
    def _to_tool_spec(tool: dict) -> dict:
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        return {
            "name": fn.get("name"),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
        }

    def get_stop_sequences(self) -> list[str] | list[int]:
        return list(self._renderer.get_stop_sequences())

    def parse(self, output_ids: list[int], *, tools_schema: list[dict] | None = None) -> ParsedOutput:
        message, termination = self._renderer.parse_response(list(output_ids))
        ill_formed = not termination.is_clean

        reasoning_parts: list[str] = []
        text_parts: list[str] = []
        tool_uses: list[dict[str, Any]] = []

        content = message.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "thinking":
                    reasoning_parts.append(part.get("thinking", "") or part.get("text", ""))
                elif ptype in ("text", "output_text"):
                    text_parts.append(part.get("text", ""))

        for call in message.get("tool_calls") or []:
            fn = call.get("function", call) if isinstance(call, dict) else {}
            name = fn.get("name") or (call.get("name") if isinstance(call, dict) else None) or "tool"
            args = fn.get("arguments")
            if isinstance(args, str):
                import json

                try:
                    args = json.loads(args or "{}")
                except json.JSONDecodeError:
                    args = {"_raw_arguments": args}
                    ill_formed = True
            tool_uses.append({"name": name, "input": args if isinstance(args, dict) else {}})

        return ParsedOutput(
            reasoning="".join(reasoning_parts),
            text="".join(text_parts),
            tool_uses=tool_uses,
            ill_formed=ill_formed,
        )


__all__ = ["HfTemplateRenderer", "ParsedOutput", "Renderer", "TinkerRenderer"]
