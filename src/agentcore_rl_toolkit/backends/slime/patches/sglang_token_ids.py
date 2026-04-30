"""Apply SGLang v0.5.9 patch to expose prompt/completion token IDs in chat completions.

SGLang's /v1/chat/completions endpoint does not include tokenized prompt or
completion IDs in the response. This patch adds four fields to the Pydantic
models and populates them from the engine's `output_ids` (always available)
and `adapted_request.input_ids`:

| Mode           | Model                                        | Field added          |
|----------------|----------------------------------------------|----------------------|
| Non-streaming  | ChatCompletionResponse                       | prompt_token_ids     |
| Non-streaming  | ChatCompletionResponseChoice                 | token_ids            |
| Streaming      | ChatCompletionStreamResponse (first chunk)   | prompt_token_ids     |
| Streaming      | ChatCompletionResponseStreamChoice           | token_ids (delta)    |

Each Pydantic field is wrapped with a `@model_serializer` that drops the key
when None, so vanilla OpenAI clients see wire-identical responses.

For streaming, the patch handles three subtleties:

1. `output_ids` is cumulative when `server_args.stream_output=False` (the
   default); we slice past a per-index watermark to get the delta.
2. The tool-call parser may yield 0, 1, or multiple chunks per engine yield,
   so delta token IDs are buffered in `_pending_token_ids` and flushed onto
   the first yielded chunk. If the parser yields nothing (partial JSON), the
   IDs are re-buffered for the next iteration.
3. When the detokenizer holds back text at word boundaries (empty `delta`),
   or when the final token is EOS (empty `delta`), the buffer is preserved
   and flushed onto the finish-reason chunk at stream end.

This enables rllm-model-gateway to capture exact RL-training trace data
without any gateway modifications and without requiring `logprobs=True`.

Usage:
    python -m agentcore_rl_toolkit.backends.slime.patches.sglang_token_ids

Or in code:
    from agentcore_rl_toolkit.backends.slime.patches.sglang_token_ids import apply_patch
    apply_patch()
"""

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# protocol.py — add Pydantic fields + None-drop serializers
# ---------------------------------------------------------------------------


def _patch_protocol(protocol_path: str) -> bool:
    """Add token_ids / prompt_token_ids fields to the four relevant models."""
    patched = False
    try:
        with open(protocol_path) as f:
            content = f.read()

        if "token_ids" in content and "prompt_token_ids" in content:
            logger.info("protocol.py already patched")
            return False

        # 1. ChatCompletionResponseChoice: add `token_ids` and extend serializer.
        #    Anchor includes the class header to disambiguate from CompletionResponseChoice
        #    (the non-chat Completion API class has an identical matched_stop/hidden_states
        #    structure earlier in the file).
        old = (
            "class ChatCompletionResponseChoice(BaseModel):\n"
            "    index: int\n"
            "    message: ChatMessage\n"
            "    logprobs: Optional[Union[LogProbs, ChoiceLogprobs]] = None\n"
            "    finish_reason: Optional[\n"
            "        Literal[\n"
            '            "stop", "length", "tool_calls", "content_filter", "function_call", "abort"\n'
            "        ]\n"
            "    ] = None\n"
            "    matched_stop: Union[None, int, str] = None\n"
            "    hidden_states: Optional[object] = None\n"
            "\n"
            '    @model_serializer(mode="wrap")\n'
            "    def _serialize(self, handler):\n"
            "        data = handler(self)\n"
            "        if self.hidden_states is None:\n"
            '            data.pop("hidden_states", None)\n'
            "        return data"
        )
        new = (
            "class ChatCompletionResponseChoice(BaseModel):\n"
            "    index: int\n"
            "    message: ChatMessage\n"
            "    logprobs: Optional[Union[LogProbs, ChoiceLogprobs]] = None\n"
            "    finish_reason: Optional[\n"
            "        Literal[\n"
            '            "stop", "length", "tool_calls", "content_filter", "function_call", "abort"\n'
            "        ]\n"
            "    ] = None\n"
            "    matched_stop: Union[None, int, str] = None\n"
            "    hidden_states: Optional[object] = None\n"
            "    token_ids: Optional[List[int]] = None\n"
            "\n"
            '    @model_serializer(mode="wrap")\n'
            "    def _serialize(self, handler):\n"
            "        data = handler(self)\n"
            "        if self.hidden_states is None:\n"
            '            data.pop("hidden_states", None)\n'
            "        if self.token_ids is None:\n"
            '            data.pop("token_ids", None)\n'
            "        return data"
        )
        if old in content:
            content = content.replace(old, new, 1)
            patched = True
            logger.info("Patched ChatCompletionResponseChoice (non-streaming)")
        else:
            logger.warning("Could not find ChatCompletionResponseChoice anchor in protocol.py")

        # 2a. ChatCompletionResponse: add `prompt_token_ids` and extend serializer.
        #     Anchor includes the class header to disambiguate from CompletionResponse.
        old = (
            "class ChatCompletionResponse(BaseModel):\n"
            "    id: str\n"
            '    object: str = "chat.completion"\n'
            "    created: int = Field(default_factory=lambda: int(time.time()))\n"
            "    model: str\n"
            "    choices: List[ChatCompletionResponseChoice]\n"
            "    usage: UsageInfo\n"
            "    metadata: Optional[Dict[str, Any]] = None\n"
            "    sglext: Optional[SglExt] = None\n"
            "\n"
            '    @model_serializer(mode="wrap")\n'
            "    def _serialize(self, handler):\n"
            "        data = handler(self)\n"
            "        if self.sglext is None:\n"
            '            data.pop("sglext", None)\n'
            "        return data"
        )
        new = (
            "class ChatCompletionResponse(BaseModel):\n"
            "    id: str\n"
            '    object: str = "chat.completion"\n'
            "    created: int = Field(default_factory=lambda: int(time.time()))\n"
            "    model: str\n"
            "    choices: List[ChatCompletionResponseChoice]\n"
            "    usage: UsageInfo\n"
            "    metadata: Optional[Dict[str, Any]] = None\n"
            "    sglext: Optional[SglExt] = None\n"
            "    prompt_token_ids: Optional[List[int]] = None\n"
            "\n"
            '    @model_serializer(mode="wrap")\n'
            "    def _serialize(self, handler):\n"
            "        data = handler(self)\n"
            "        if self.sglext is None:\n"
            '            data.pop("sglext", None)\n'
            "        if self.prompt_token_ids is None:\n"
            '            data.pop("prompt_token_ids", None)\n'
            "        return data"
        )
        if old in content:
            content = content.replace(old, new, 1)
            patched = True
            logger.info("Patched ChatCompletionResponse (non-streaming)")
        else:
            logger.warning("Could not find ChatCompletionResponse anchor in protocol.py")

        # 3. ChatCompletionResponseStreamChoice: add `token_ids` + serializer
        #    (this model previously had no custom serializer).
        old = (
            "    matched_stop: Union[None, int, str] = None\n"
            "\n"
            "\n"
            "class ChatCompletionStreamResponse(BaseModel):"
        )
        new = (
            "    matched_stop: Union[None, int, str] = None\n"
            "    token_ids: Optional[List[int]] = None\n"
            "\n"
            '    @model_serializer(mode="wrap")\n'
            "    def _serialize(self, handler):\n"
            "        data = handler(self)\n"
            "        if self.token_ids is None:\n"
            '            data.pop("token_ids", None)\n'
            "        return data\n"
            "\n"
            "\n"
            "class ChatCompletionStreamResponse(BaseModel):"
        )
        if old in content:
            content = content.replace(old, new, 1)
            patched = True
            logger.info("Patched ChatCompletionResponseStreamChoice (streaming)")
        else:
            logger.warning("Could not find ChatCompletionResponseStreamChoice anchor in protocol.py")

        # 2b. ChatCompletionStreamResponse: add `prompt_token_ids` and extend serializer.
        #     After patch 3, the ChatCompletionStreamResponse class sits just below the
        #     newly-added ChatCompletionResponseStreamChoice serializer.
        old = (
            "class ChatCompletionStreamResponse(BaseModel):\n"
            "    id: str\n"
            '    object: str = "chat.completion.chunk"\n'
            "    created: int = Field(default_factory=lambda: int(time.time()))\n"
            "    model: str\n"
            "    choices: List[ChatCompletionResponseStreamChoice]\n"
            "    usage: Optional[UsageInfo] = None\n"
            "    sglext: Optional[SglExt] = None\n"
            "\n"
            '    @model_serializer(mode="wrap")\n'
            "    def _serialize(self, handler):\n"
            "        data = handler(self)\n"
            "        if self.sglext is None:\n"
            '            data.pop("sglext", None)\n'
            "        return data"
        )
        new = (
            "class ChatCompletionStreamResponse(BaseModel):\n"
            "    id: str\n"
            '    object: str = "chat.completion.chunk"\n'
            "    created: int = Field(default_factory=lambda: int(time.time()))\n"
            "    model: str\n"
            "    choices: List[ChatCompletionResponseStreamChoice]\n"
            "    usage: Optional[UsageInfo] = None\n"
            "    sglext: Optional[SglExt] = None\n"
            "    prompt_token_ids: Optional[List[int]] = None\n"
            "\n"
            '    @model_serializer(mode="wrap")\n'
            "    def _serialize(self, handler):\n"
            "        data = handler(self)\n"
            "        if self.sglext is None:\n"
            '            data.pop("sglext", None)\n'
            "        if self.prompt_token_ids is None:\n"
            '            data.pop("prompt_token_ids", None)\n'
            "        return data"
        )
        if old in content:
            content = content.replace(old, new, 1)
            patched = True
            logger.info("Patched ChatCompletionStreamResponse (streaming)")
        else:
            logger.warning("Could not find ChatCompletionStreamResponse anchor in protocol.py")

        if patched:
            with open(protocol_path, "w") as f:
                f.write(content)
    except Exception as e:
        logger.error("Failed to patch protocol.py: %s", e)

    return patched


# ---------------------------------------------------------------------------
# serving_chat.py — extract + thread token IDs through streaming / non-streaming
# ---------------------------------------------------------------------------


def _patch_serving_chat(chat_path: str) -> bool:
    """Patch streaming and non-streaming chat completion paths to emit token_ids."""
    patched = False
    try:
        with open(chat_path) as f:
            content = f.read()

        if "_pending_token_ids" in content and "prompt_token_ids=_prompt_ids" in content:
            logger.info("serving_chat.py already patched")
            return False

        # ----- Streaming path -----

        # 1. Add state dicts at top of _generate_chat_stream.
        old = "        n_prev_tokens = {}"
        new = (
            "        n_prev_tokens = {}\n"
            "        _pending_token_ids = {}  # Buffer delta token_ids when tool call parser yields 0 chunks\n"
            "        _n_prev_ids = {}  # Track prev output_ids length per index"
            " (output_ids is cumulative when stream_output=False)"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched streaming state dicts")
        else:
            logger.warning("Could not find n_prev_tokens init in streaming code")

        # 2. Add _first_chunk_sent dict just before the `try:` block.
        old = "        hidden_states = {}\n" "        routed_experts = {}\n" "\n" "        try:"
        new = (
            "        hidden_states = {}\n"
            "        routed_experts = {}\n"
            "\n"
            "        _first_chunk_sent = {}\n"
            "        try:"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched _first_chunk_sent init")
        else:
            logger.warning("Could not find `try:` anchor in streaming code")

        # 3. Inject prompt_token_ids into the first (role-announcement) chunk.
        old = (
            "                        finish_reason=None,\n"
            "                        logprobs=None,\n"
            "                    )\n"
            "                    chunk = ChatCompletionStreamResponse(\n"
            '                        id=content["meta_info"]["id"],\n'
            "                        created=int(time.time()),\n"
            "                        choices=[choice_data],\n"
            "                        model=request.model,\n"
            "                    )\n"
            '                    yield f"data: {chunk.model_dump_json()}\\n\\n"'
        )
        new = (
            "                        finish_reason=None,\n"
            "                        logprobs=None,\n"
            "                    )\n"
            "                    # Include prompt_token_ids in the first chunk for RL training trace capture\n"
            "                    _prompt_ids = None\n"
            "                    if index not in _first_chunk_sent:\n"
            "                        _first_chunk_sent[index] = True\n"
            "                        _p_ids = adapted_request.input_ids\n"
            "                        if _p_ids and not isinstance(_p_ids, str):\n"
            "                            if isinstance(_p_ids[0], list):\n"
            "                                _p_ids = _p_ids[0]  # unwrap batch dim\n"
            "                            _prompt_ids = _p_ids\n"
            "\n"
            "                    chunk = ChatCompletionStreamResponse(\n"
            '                        id=content["meta_info"]["id"],\n'
            "                        created=int(time.time()),\n"
            "                        choices=[choice_data],\n"
            "                        model=request.model,\n"
            "                        prompt_token_ids=_prompt_ids,\n"
            "                    )\n"
            '                    yield f"data: {chunk.model_dump_json()}\\n\\n"'
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched first-chunk prompt_token_ids injection")
        else:
            logger.warning("Could not find first-chunk yield in streaming code")

        # 4. Extract streaming delta token_ids from output_ids, accumulate into
        #    _pending_token_ids. Also flush pending on tool-call branch with
        #    re-buffer on zero-yield.
        old = (
            "                # Handle tool calls\n"
            "                if (\n"
            '                    request.tool_choice != "none"\n'
            "                    and request.tools\n"
            "                    and self.tool_call_parser\n"
            "                ):\n"
            "                    async for chunk in self._process_tool_call_stream(\n"
            "                        index,\n"
            "                        delta,\n"
            "                        parser_dict,\n"
            "                        content,\n"
            "                        request,\n"
            "                        has_tool_calls,\n"
            "                    ):\n"
            "                        if chunk:\n"
            "                            yield chunk\n"
            "\n"
            "                    # Send any remaining tool call arguments when generation finishes"
        )
        new = (
            "                # Extract streaming delta token_ids from output_ids. When\n"
            "                # server_args.stream_output is False (the default), output_ids is\n"
            "                # cumulative across yields; slice past the previous length for the\n"
            "                # delta. When True, output_ids is already the per-yield delta.\n"
            '                _all_ids = content.get("output_ids") or []\n'
            "                if _all_ids:\n"
            '                    if getattr(self.tokenizer_manager.server_args, "stream_output", False):\n'
            "                        _delta_ids = _all_ids\n"
            "                    else:\n"
            "                        _prev = _n_prev_ids.get(index, 0)\n"
            "                        _delta_ids = _all_ids[_prev:]\n"
            "                        _n_prev_ids[index] = len(_all_ids)\n"
            "                    if _delta_ids:\n"
            "                        _pending_token_ids.setdefault(index, []).extend(_delta_ids)\n"
            "\n"
            "                # Handle tool calls\n"
            "                if (\n"
            '                    request.tool_choice != "none"\n'
            "                    and request.tools\n"
            "                    and self.tool_call_parser\n"
            "                ):\n"
            "                    # Flush accumulated token_ids when a chunk is actually yielded\n"
            "                    _flush_ids = _pending_token_ids.pop(index, None)\n"
            "                    async for chunk in self._process_tool_call_stream(\n"
            "                        index,\n"
            "                        delta,\n"
            "                        parser_dict,\n"
            "                        content,\n"
            "                        request,\n"
            "                        has_tool_calls,\n"
            "                        _flush_ids,\n"
            "                    ):\n"
            "                        if chunk:\n"
            "                            _flush_ids = None  # Only attach to first yielded chunk\n"
            "                            yield chunk\n"
            "                    # If no chunk was yielded, put the ids back for next iteration\n"
            "                    if _flush_ids is not None:\n"
            "                        _pending_token_ids.setdefault(index, []).extend(_flush_ids)\n"
            "\n"
            "                    # Send any remaining tool call arguments when generation finishes"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched delta-token extraction + tool-call branch flushing")
        else:
            logger.warning("Could not find tool-call branch in streaming code")

        # 5. Regular-content branch: conditional pop (only when delta non-empty)
        #    so the empty-delta case (detokenizer boundary, EOS) leaves IDs
        #    buffered for later flush.
        old = (
            "                else:\n"
            "                    # Regular content\n"
            "                    if delta:\n"
            "                        choice_data = ChatCompletionResponseStreamChoice(\n"
            "                            index=index,\n"
            "                            delta=DeltaMessage(content=delta),\n"
            "                            finish_reason=None,\n"
            "                            matched_stop=None,\n"
            "                            logprobs=choice_logprobs,\n"
            "                        )"
        )
        new = (
            "                else:\n"
            "                    # Regular content — flush all pending token_ids only when a chunk\n"
            "                    # is actually emitted. If `delta` is empty (detokenizer buffering\n"
            "                    # across word boundaries, or a trailing stop token with no text),\n"
            "                    # leave the ids buffered so the next emitted chunk or the finish\n"
            "                    # chunk at the end picks them up.\n"
            "                    _flush_ids = None\n"
            "                    if delta:\n"
            "                        _flush_ids = _pending_token_ids.pop(index, None)\n"
            "                        choice_data = ChatCompletionResponseStreamChoice(\n"
            "                            index=index,\n"
            "                            delta=DeltaMessage(content=delta),\n"
            "                            finish_reason=None,\n"
            "                            matched_stop=None,\n"
            "                            logprobs=choice_logprobs,\n"
            "                            token_ids=_flush_ids,\n"
            "                        )"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched regular-content branch (conditional pop)")
        else:
            logger.warning("Could not find regular-content branch in streaming code")

        # 6. Flush residual IDs onto the finish-reason chunk so the final EOS
        #    token (which has empty delta) is captured.
        old = (
            '                if has_tool_calls.get(idx, False) and finish_reason_type == "stop":\n'
            '                    final_finish_reason = "tool_calls"\n'
            "\n"
            "                finish_reason_chunk = ChatCompletionStreamResponse("
        )
        new = (
            '                if has_tool_calls.get(idx, False) and finish_reason_type == "stop":\n'
            '                    final_finish_reason = "tool_calls"\n'
            "\n"
            "                # Flush any remaining buffered token_ids on the finish chunk so callers\n"
            "                # see the full completion sequence (including the EOS/matched_stop token,\n"
            "                # which has no associated text delta and would otherwise be dropped).\n"
            "                _final_ids = _pending_token_ids.pop(idx, None)\n"
            "\n"
            "                finish_reason_chunk = ChatCompletionStreamResponse("
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched finish-reason chunk prep (residual ID flush)")
        else:
            logger.warning("Could not find finish-reason chunk anchor")

        # 7. Attach _final_ids to the finish-reason stream choice.
        old = (
            "                            finish_reason=final_finish_reason,\n"
            "                            matched_stop=(\n"
            '                                finish_reason_data["matched"]\n'
            '                                if "matched" in finish_reason_data\n'
            "                                else None\n"
            "                            ),\n"
            "                        )\n"
            "                    ],\n"
            "                    model=request.model,"
        )
        new = (
            "                            finish_reason=final_finish_reason,\n"
            "                            matched_stop=(\n"
            '                                finish_reason_data["matched"]\n'
            '                                if "matched" in finish_reason_data\n'
            "                                else None\n"
            "                            ),\n"
            "                            token_ids=_final_ids,\n"
            "                        )\n"
            "                    ],\n"
            "                    model=request.model,"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched finish-reason choice (attach token_ids)")
        else:
            logger.warning("Could not find finish-reason choice constructor")

        # ----- Non-streaming path -----

        # 8. Attach prompt_token_ids to ChatCompletionResponse in _handle_non_streaming_request.
        old = (
            "            int(time.time()),\n"
            "        )\n"
            "\n"
            "        return response\n"
            "\n"
            "    def _build_chat_response("
        )
        new = (
            "            int(time.time()),\n"
            "        )\n"
            "\n"
            "        # Add prompt_token_ids for RL training trace capture\n"
            "        if isinstance(response, ChatCompletionResponse):\n"
            "            _p_ids = adapted_request.input_ids\n"
            "            if _p_ids and not isinstance(_p_ids, str):\n"
            "                if isinstance(_p_ids[0], list):\n"
            "                    _p_ids = _p_ids[0]  # unwrap batch dim\n"
            "                response.prompt_token_ids = _p_ids\n"
            "\n"
            "        return response\n"
            "\n"
            "    def _build_chat_response("
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched non-streaming prompt_token_ids injection")
        else:
            logger.warning("Could not find non-streaming response anchor")

        # 9. Attach token_ids to ChatCompletionResponseChoice in _build_chat_response.
        old = (
            "                    history_tool_calls_cnt,\n"
            "                )\n"
            "\n"
            "            choice_data = ChatCompletionResponseChoice(\n"
            "                index=idx,"
        )
        new = (
            "                    history_tool_calls_cnt,\n"
            "                )\n"
            "\n"
            "            # Extract completion token IDs (always available, no logprobs dependency)\n"
            '            _token_ids = ret_item.get("output_ids") or None\n'
            "\n"
            "            choice_data = ChatCompletionResponseChoice(\n"
            "                index=idx,"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched non-streaming choice prep")
        else:
            logger.warning("Could not find non-streaming choice anchor")

        # 10. Add token_ids= to the ChatCompletionResponseChoice constructor.
        old = (
            "                hidden_states=hidden_states,\n" "            )\n" "            choices.append(choice_data)"
        )
        new = (
            "                hidden_states=hidden_states,\n"
            "                token_ids=_token_ids,\n"
            "            )\n"
            "            choices.append(choice_data)"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched non-streaming choice constructor")
        else:
            logger.warning("Could not find non-streaming choice constructor")

        # ----- Tool-call parser signature + internal attachment -----

        # 11. Add delta_token_ids parameter to _process_tool_call_stream.
        old = (
            "    async def _process_tool_call_stream(\n"
            "        self,\n"
            "        index: int,\n"
            "        delta: str,\n"
            "        parser_dict: Dict[int, FunctionCallParser],\n"
            "        content: Dict[str, Any],\n"
            "        request: ChatCompletionRequest,\n"
            "        has_tool_calls: Dict[int, bool],\n"
            "    ):"
        )
        new = (
            "    async def _process_tool_call_stream(\n"
            "        self,\n"
            "        index: int,\n"
            "        delta: str,\n"
            "        parser_dict: Dict[int, FunctionCallParser],\n"
            "        content: Dict[str, Any],\n"
            "        request: ChatCompletionRequest,\n"
            "        has_tool_calls: Dict[int, bool],\n"
            "        delta_token_ids=None,\n"
            "    ):"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched _process_tool_call_stream signature")
        else:
            logger.warning("Could not find _process_tool_call_stream signature")

        # 12. Attach delta_token_ids to the normal-text chunk inside the parser.
        old = (
            "            choice_data = ChatCompletionResponseStreamChoice(\n"
            "                index=index,\n"
            "                delta=DeltaMessage(content=normal_text),\n"
            "                finish_reason=None,\n"
            "            )"
        )
        new = (
            "            choice_data = ChatCompletionResponseStreamChoice(\n"
            "                index=index,\n"
            "                delta=DeltaMessage(content=normal_text),\n"
            "                finish_reason=None,\n"
            "                token_ids=delta_token_ids,\n"
            "            )"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched parser normal-text chunk")
        else:
            logger.warning("Could not find parser normal-text chunk anchor")

        # 13. Attach delta_token_ids to the tool-call chunk inside the parser, and
        #     null it after the yield to enforce first-claims-all.
        old = (
            "            choice_data = ChatCompletionResponseStreamChoice(\n"
            "                index=index,\n"
            "                delta=DeltaMessage(tool_calls=[tool_call]),\n"
            "                finish_reason=None,\n"
            "            )"
        )
        new = (
            "            choice_data = ChatCompletionResponseStreamChoice(\n"
            "                index=index,\n"
            "                delta=DeltaMessage(tool_calls=[tool_call]),\n"
            "                finish_reason=None,\n"
            "                token_ids=delta_token_ids,\n"
            "            )\n"
            "            delta_token_ids = None  # Only attach to first chunk"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched parser tool-call chunk")
        else:
            logger.warning("Could not find parser tool-call chunk anchor")

        patched = True
        with open(chat_path, "w") as f:
            f.write(content)
    except Exception as e:
        logger.error("Failed to patch serving_chat.py: %s", e)

    return patched


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def apply_patch() -> bool:
    """Patch the installed SGLang package.

    Returns True if either file was patched, False if both were already
    patched, SGLang wasn't found, or patching failed.
    """
    try:
        import sglang

        sglang_dir = __import__("os").path.dirname(sglang.__file__)
    except ImportError:
        logger.warning("SGLang not installed, skipping patch")
        return False

    protocol_path = f"{sglang_dir}/srt/entrypoints/openai/protocol.py"
    chat_path = f"{sglang_dir}/srt/entrypoints/openai/serving_chat.py"

    p1 = _patch_protocol(protocol_path)
    p2 = _patch_serving_chat(chat_path)
    return p1 or p2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if apply_patch():
        print("SGLang token_ids patch applied successfully")
    else:
        print("No patch applied (already patched or SGLang not found)")
