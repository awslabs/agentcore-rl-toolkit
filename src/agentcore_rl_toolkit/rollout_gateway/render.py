"""Tokenization for the rollout gateway.

The gateway owns tokenization: it renders canonical chat messages (+ tools) into
``token_ids`` and derenders sampled ``token_ids`` back into text / reasoning / tool
calls. Owning both directions in one place makes loss-masking well-defined and
eliminates cross-backend retokenization drift. It is also required for sample-only
inference backends (e.g. Tinker) that cannot render themselves.

Two implementations behind one :class:`Renderer` protocol:

* :class:`HfTemplateRenderer` (default, lightweight) — renders with the HF tokenizer's
  ``apply_chat_template`` and parses output via ``parse_model_output`` (dependency-free
  regex + ``</think>`` split). Needs only ``transformers``.
* :class:`TinkerRenderer` — wraps a tinker-cookbook ``Renderer`` (install ``tinker``
  and ``tinker-cookbook`` manually; they require Python >=3.11). Its ``tinker_cookbook``
  import (which pulls torch) is deferred into ``__init__``, so importing this module
  stays torch-free; the heavy import fires only when a ``TinkerRenderer`` is constructed.
"""

import dataclasses
from typing import Any, Protocol, runtime_checkable

from .parsing import ParsedModelOutput, parse_model_output


@dataclasses.dataclass(frozen=True)
class ParsedOutput:
    """Derender result: what the model produced this turn, protocol-agnostic.

    ``ill_formed`` flags a parse that could not cleanly terminate / decode.
    """

    reasoning: str
    text: str
    tool_uses: list[dict[str, Any]]
    ill_formed: bool = False


def _to_parsed_output(p: ParsedModelOutput) -> ParsedOutput:
    return ParsedOutput(reasoning=p.reasoning, text=p.text, tool_uses=p.tool_uses, ill_formed=p.ill_formed)


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
    """Default renderer: HF ``apply_chat_template`` for rendering, ``parse_model_output``
    for derendering.

    Depends only on a HF tokenizer (``transformers``). Tool calls are parsed with a
    dependency-free regex (the common ``<tool_call>`` format) and reasoning via a
    ``</think>`` split — no inference-engine dependency. A model whose output needs
    engine-grade parsing is handled by a different ``Renderer`` implementation, not here.
    """

    def __init__(
        self,
        tokenizer,
        *,
        stop_sequences: list[str] | list[int] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self._stop_sequences: list = list(stop_sequences) if stop_sequences else []

    def render(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        enc = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        # apply_chat_template may return a BatchEncoding (dict-like with "input_ids")
        # or a bare list of ids depending on transformers version / return_dict.
        ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
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
        parsed = parse_model_output(raw_output, tools_schema=tools_schema)
        return _to_parsed_output(parsed)


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
