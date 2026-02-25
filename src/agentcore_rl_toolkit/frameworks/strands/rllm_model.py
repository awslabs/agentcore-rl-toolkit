"""
Strands Model adapter that calls rLLM's ``/v1/model_response`` endpoint.

Unlike the standard ``OpenAIModel`` (which uses ``/v1/chat/completions``),
this model captures the full ``ModelOutput`` returned by rLLM's inference
server -- including ``prompt_ids``, ``completion_ids``, and ``logprobs`` --
which are required for policy gradient RL training.

Usage::

    model = RLLMRemoteModel(base_url="http://trainer:8089/v1", model_id="Qwen/Qwen3-4B")
    rollout_collector = StrandsRolloutCollector()
    agent = Agent(model=model, tools=[calculator], hooks=[rollout_collector])

    # After agent invocation, enrich the rollout data:
    model.clear_model_outputs()
    response = await agent.invoke_async(prompt)
    rollout_data = rollout_collector.get_rollout_data()
    for turn, output in zip(rollout_data, model.get_model_outputs()):
        turn["model_output"] = output
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterable
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class RLLMRemoteModel:
    """Strands-compatible Model that calls rLLM's inference API with token capture.

    Implements the Strands ``Model`` interface (``stream``, ``format_request``,
    ``get_config``, ``update_config``) so it can be used as a drop-in
    replacement for ``OpenAIModel`` inside a Strands ``Agent``.

    Each call to the model stores the full ``ModelOutput`` dict (including
    ``prompt_ids``, ``completion_ids``, ``logprobs``) in an internal list.
    The entrypoint function can retrieve these via ``get_model_outputs()``
    to enrich the rollout data before saving to S3.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        timeout: float = 300.0,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._max_retries = max_retries
        self._config: dict[str, Any] = {"model_id": model_id, "params": dict(kwargs)}

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        )

        # Stores ModelOutput dicts for each model call in the current episode.
        self._model_outputs: list[dict] = []

    # ------------------------------------------------------------------
    # Strands Model interface
    # ------------------------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        return {
            "model_id": self._config.get("model_id"),
            "params": dict(self._config.get("params") or {}),
        }

    def update_config(self, **model_config: Any) -> None:
        if "model_id" in model_config:
            self._config["model_id"] = model_config.pop("model_id")
        params = self._config.get("params") or {}
        params.update(model_config)
        self._config["params"] = params

    def format_request(
        self,
        messages,
        tool_specs=None,
        system_prompt: str | None = None,
    ) -> dict:
        """Format a request for the model.

        Called by ``StrandsRolloutCollector`` hooks to capture the
        conversation history before each model invocation.

        Returns a dict with a ``messages`` key containing OpenAI-style
        message dicts.
        """
        chat_messages = self._convert_messages_to_chat_format(messages, system_prompt)
        result: dict[str, Any] = {"messages": chat_messages}

        if tool_specs:
            result["tools"] = self._convert_tool_specs_to_openai_format(tool_specs)

        return result

    async def stream(
        self,
        messages,
        tool_specs=None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[dict]:
        """Stream conversation with the model via rLLM's ``/v1/model_response``.

        Yields standard Strands ``StreamEvent`` dicts consumed by the
        Agent's event loop.
        """
        chat_messages = self._convert_messages_to_chat_format(messages, system_prompt)

        # Build the request payload
        payload: dict[str, Any] = {"messages": chat_messages}

        if tool_specs:
            tools_param = self._convert_tool_specs_to_openai_format(tool_specs)
            payload["extra_params"] = {"tools": tools_param}

        # Forward only known serializable sampling kwargs.
        # Strands may pass non-serializable objects (Agent, callbacks, etc.)
        # through **kwargs -- we must NOT forward those into the JSON payload.
        _KNOWN_KEYS = ("temperature", "top_p", "max_tokens", "max_completion_tokens", "stop")
        for key in _KNOWN_KEYS:
            if key in kwargs:
                payload[key] = kwargs[key]

        # Call rLLM's /v1/model_response
        model_output = await self._call_model_response(payload)

        # Store for rollout enrichment
        self._model_outputs.append(model_output)

        # Yield Strands StreamEvents
        yield {"messageStart": {"role": "assistant"}}

        tool_calls = model_output.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                tool_call_info = self._extract_tool_call_info(tc)

                yield {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "toolUseId": tool_call_info["id"],
                                "name": tool_call_info["name"],
                            }
                        }
                    }
                }

                input_str = json.dumps(tool_call_info["input"])
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": input_str}}}}
                yield {"contentBlockStop": {}}

            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            response_text = model_output.get("text") or model_output.get("content") or ""
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": response_text}}}
            yield {"contentBlockStop": {}}

            stop_reason = model_output.get("finish_reason") or "end_turn"
            yield {"messageStop": {"stopReason": stop_reason}}

    # ------------------------------------------------------------------
    # Token capture API
    # ------------------------------------------------------------------

    def get_model_outputs(self) -> list[dict]:
        """Return all stored ModelOutput dicts for the current episode.

        Each dict contains ``prompt_ids``, ``completion_ids``, ``logprobs``,
        ``text``, ``finish_reason``, etc.
        """
        return list(self._model_outputs)

    def clear_model_outputs(self) -> None:
        """Clear stored model outputs.  Call before each new episode."""
        self._model_outputs.clear()

    # ------------------------------------------------------------------
    # HTTP call to rLLM inference API
    # ------------------------------------------------------------------

    async def _call_model_response(self, payload: dict) -> dict:
        """POST to ``/v1/model_response`` with retry logic.

        Returns the raw ``model_output`` dict from the response.
        """
        import asyncio

        # Sanitize the payload: ensure it is JSON-serializable.
        # Strands may sneak non-serializable objects (Agent, hooks, etc.)
        # into kwargs or tool specs.  Round-tripping through json.dumps
        # catches this early with a clear error instead of failing deep
        # inside httpx.
        try:
            sanitized = json.loads(json.dumps(payload, default=str))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"RLLMRemoteModel: payload is not JSON-serializable: {exc}.  "
                "This usually means a non-serializable Strands object leaked "
                "into the request.  Check tool_specs and kwargs."
            ) from exc

        url = f"{self._base_url}/model_response"
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._client.post(url, json=sanitized)
                response.raise_for_status()
                data = response.json()
                return data["model_output"]
            except Exception as e:
                last_error = e
                logger.warning(
                    f"RLLMRemoteModel call failed (attempt {attempt}/{self._max_retries}): {e}"
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(min(2**attempt, 10))

        raise RuntimeError(
            f"RLLMRemoteModel failed after {self._max_retries} attempts: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Message / tool conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages_to_chat_format(
        messages,
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Convert Strands ``Messages`` to OpenAI chat-completion dicts."""
        chat_messages: list[dict[str, str]] = []

        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})

        for message in messages:
            role = message["role"]
            content = message["content"]

            text_content = ""
            for block in content:
                if "text" in block:
                    text_content += block["text"]
                elif "toolUse" in block:
                    tool_use = block["toolUse"]
                    text_content += f"[Tool: {tool_use['name']} with input: {tool_use.get('input', {})}]"
                elif "toolResult" in block:
                    tool_result = block["toolResult"]
                    text_content += f"[Tool Result: {tool_result.get('content', [])}]"

            if text_content.strip():
                chat_messages.append({"role": role, "content": text_content})

        return chat_messages

    @staticmethod
    def _convert_tool_specs_to_openai_format(tool_specs) -> list[dict]:
        """Convert Strands tool specs to OpenAI tools format."""
        tools_param: list[dict] = []

        for spec in tool_specs:
            try:
                if hasattr(spec, "tool_spec"):
                    tool_spec = spec.tool_spec
                    if isinstance(tool_spec, dict):
                        name = tool_spec.get("name")
                        input_schema = tool_spec.get("inputSchema", {})
                        if isinstance(input_schema, dict) and "json" in input_schema:
                            params = input_schema["json"]
                            if name and isinstance(params, dict):
                                tools_param.append({"type": "function", "function": {"name": name, "parameters": params}})
                                continue

                if hasattr(spec, "TOOL_SPEC"):
                    tool_spec = spec.TOOL_SPEC
                    if isinstance(tool_spec, dict):
                        name = tool_spec.get("name")
                        input_schema = tool_spec.get("inputSchema", {})
                        if isinstance(input_schema, dict) and "json" in input_schema:
                            params = input_schema["json"]
                            if name and isinstance(params, dict):
                                tools_param.append({"type": "function", "function": {"name": name, "parameters": params}})
                                continue

                if hasattr(spec, "to_openai_tool"):
                    maybe = spec.to_openai_tool()
                    if isinstance(maybe, dict):
                        tools_param.append(maybe)
                        continue

                if isinstance(spec, dict):
                    if spec.get("type") == "function" and isinstance(spec.get("function"), dict):
                        tools_param.append(spec)
                        continue
                    name = spec.get("name")
                    params = spec.get("parameters") or spec.get("input_schema")
                    if not params and "inputSchema" in spec:
                        input_schema = spec["inputSchema"]
                        if isinstance(input_schema, dict) and "json" in input_schema:
                            params = input_schema["json"]
                    if name and isinstance(params, dict):
                        tools_param.append({"type": "function", "function": {"name": name, "parameters": params}})
                    continue

                name = getattr(spec, "name", None)
                params = getattr(spec, "input_schema", None) or getattr(spec, "parameters", None)
                if name and isinstance(params, dict):
                    tools_param.append({"type": "function", "function": {"name": name, "parameters": params}})

            except Exception as e:
                logger.warning(f"Failed to convert tool spec {spec}: {e}")
                continue

        return tools_param

    @staticmethod
    def _extract_tool_call_info(tool_call) -> dict:
        """Extract tool call info from a ModelOutput tool_call entry."""
        try:
            if isinstance(tool_call, dict):
                name = tool_call.get("name")
                arguments = tool_call.get("arguments", {})
            else:
                name = getattr(tool_call, "name", None)
                arguments = getattr(tool_call, "arguments", {})

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments.strip() else {}
                except Exception:
                    arguments = {"_raw": arguments}

            tool_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
            return {"id": tool_id or "unknown", "name": name or "tool", "input": arguments}
        except Exception:
            return {"id": "unknown", "name": "unknown", "input": {}}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()
