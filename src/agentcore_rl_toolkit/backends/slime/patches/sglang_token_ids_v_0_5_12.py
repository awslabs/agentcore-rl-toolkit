"""Apply SGLang v0.5.12.post1 patch to expose prompt/completion token IDs in chat completions.

Same goal as sglang_token_ids.py (v0.5.9) but for the refactored v0.5.12 streaming
path, which replaced per-Pydantic-model chunk construction with a msgspec fast path
(_fast_sse_content / _StreamChunk / _StreamChoice). The non-streaming path and
protocol.py anchors are identical to v0.5.9.

Wire-format:
  Streaming  → choices[0].token_ids (per chunk, via _StreamChoice)
               prompt_token_ids at top level of the first chunk (via _StreamChunk)
  Non-stream → choices[0].token_ids
               prompt_token_ids at response top level

Usage:
    python -m agentcore_rl_toolkit.backends.slime.patches.sglang_token_ids_v5_12

Or in code:
    from agentcore_rl_toolkit.backends.slime.patches.sglang_token_ids_v5_12 import apply_patch
    apply_patch()
"""

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# protocol.py — add Pydantic fields + None-drop serializers
# (anchors identical to v0.5.9)
# ---------------------------------------------------------------------------


def _patch_protocol(protocol_path: str) -> bool:
    patched = False
    try:
        with open(protocol_path) as f:
            content = f.read()

        if "token_ids" in content and "prompt_token_ids" in content:
            logger.info("protocol.py already patched")
            return False

        # 1. ChatCompletionResponseChoice: add token_ids + extend serializer.
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

        # 2. ChatCompletionResponse: add prompt_token_ids + extend serializer.
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

        # 3. ChatCompletionResponseStreamChoice: add token_ids + new serializer.
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
            logger.info("Patched ChatCompletionResponseStreamChoice (streaming Pydantic)")
        else:
            logger.warning("Could not find ChatCompletionResponseStreamChoice anchor in protocol.py")

        # 4. ChatCompletionStreamResponse: add prompt_token_ids + extend serializer.
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
            logger.info("Patched ChatCompletionStreamResponse (Pydantic, tool-call/sglext path)")
        else:
            logger.warning("Could not find ChatCompletionStreamResponse anchor in protocol.py")

        if patched:
            with open(protocol_path, "w") as f:
                f.write(content)
    except Exception as e:
        logger.error("Failed to patch protocol.py: %s", e)

    return patched


# ---------------------------------------------------------------------------
# serving_chat.py — v0.5.12 streaming path
#
# The fast path uses msgspec structs (_StreamChunk/_StreamChoice) serialized
# by _fast_sse_content(). The verify script checks choices[0].token_ids, so
# token_ids must go on _StreamChoice (inside choices[]), NOT on _StreamChunk.
# prompt_token_ids goes on _StreamChunk (top-level of the first chunk).
# ---------------------------------------------------------------------------


def _patch_serving_chat(chat_path: str) -> bool:
    patched = False
    try:
        with open(chat_path) as f:
            content = f.read()

        if "_pending_token_ids" in content and "prompt_token_ids=_prompt_ids" in content:
            logger.info("serving_chat.py already patched")
            return False

        # ----- msgspec fast-path structs -----

        # 1. Add token_ids to _StreamChoice (shows up as choices[0].token_ids).
        old = (
            "class _StreamChoice(msgspec.Struct):\n"
            "    index: int\n"
            "    delta: _StreamDelta\n"
            "    logprobs: Optional[dict] = None\n"
            "    finish_reason: Optional[str] = None\n"
            "    matched_stop: Union[None, int, str] = None"
        )
        new = (
            "class _StreamChoice(msgspec.Struct, omit_defaults=True):\n"
            "    index: int\n"
            "    delta: _StreamDelta\n"
            "    logprobs: Optional[dict] = None\n"
            "    finish_reason: Optional[str] = None\n"
            "    matched_stop: Union[None, int, str] = None\n"
            "    token_ids: Optional[List[int]] = None"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched _StreamChoice (added token_ids)")
        else:
            logger.warning("Could not find _StreamChoice anchor")

        # 2. Add prompt_token_ids to _StreamChunk (top-level first-chunk field).
        old = (
            "class _StreamChunk(msgspec.Struct, omit_defaults=True):\n"
            "    id: str\n"
            "    object: str\n"
            "    created: int\n"
            "    model: str\n"
            "    choices: List[_StreamChoice]\n"
            "    usage: Optional[dict] = None"
        )
        new = (
            "class _StreamChunk(msgspec.Struct, omit_defaults=True):\n"
            "    id: str\n"
            "    object: str\n"
            "    created: int\n"
            "    model: str\n"
            "    choices: List[_StreamChoice]\n"
            "    usage: Optional[dict] = None\n"
            "    prompt_token_ids: Optional[List[int]] = None"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched _StreamChunk (added prompt_token_ids)")
        else:
            logger.warning("Could not find _StreamChunk anchor")

        # 3. Extend _fast_sse_content to accept and forward token_ids / prompt_token_ids.
        old = (
            "def _fast_sse_content(\n"
            "    chunk_id: str,\n"
            "    created: int,\n"
            "    model: str,\n"
            "    index: int,\n"
            "    role: Optional[str] = None,\n"
            "    content: Optional[str] = None,\n"
            "    reasoning_content: Optional[str] = None,\n"
            "    finish_reason: Optional[str] = None,\n"
            "    logprobs: Optional[dict] = None,\n"
            "    matched_stop: Union[None, int, str] = None,\n"
            "    usage: Optional[dict] = None,\n"
            ") -> str:\n"
            "    delta = _StreamDelta(\n"
            "        role=role, content=content, reasoning_content=reasoning_content\n"
            "    )\n"
            "    choice = _StreamChoice(\n"
            "        index=index,\n"
            "        delta=delta,\n"
            "        logprobs=logprobs,\n"
            "        finish_reason=finish_reason,\n"
            "        matched_stop=matched_stop,\n"
            "    )\n"
            "    chunk = _StreamChunk(\n"
            "        id=chunk_id,\n"
            '        object="chat.completion.chunk",\n'
            "        created=created,\n"
            "        model=model,\n"
            "        choices=[choice],\n"
            "        usage=usage,\n"
            "    )\n"
            "    return (_SSE_DATA_B + _stream_encoder.encode(chunk) + _SSE_NL_B).decode()"
        )
        new = (
            "def _fast_sse_content(\n"
            "    chunk_id: str,\n"
            "    created: int,\n"
            "    model: str,\n"
            "    index: int,\n"
            "    role: Optional[str] = None,\n"
            "    content: Optional[str] = None,\n"
            "    reasoning_content: Optional[str] = None,\n"
            "    finish_reason: Optional[str] = None,\n"
            "    logprobs: Optional[dict] = None,\n"
            "    matched_stop: Union[None, int, str] = None,\n"
            "    usage: Optional[dict] = None,\n"
            "    token_ids: Optional[List[int]] = None,\n"
            "    prompt_token_ids: Optional[List[int]] = None,\n"
            ") -> str:\n"
            "    delta = _StreamDelta(\n"
            "        role=role, content=content, reasoning_content=reasoning_content\n"
            "    )\n"
            "    choice = _StreamChoice(\n"
            "        index=index,\n"
            "        delta=delta,\n"
            "        logprobs=logprobs,\n"
            "        finish_reason=finish_reason,\n"
            "        matched_stop=matched_stop,\n"
            "        token_ids=token_ids,\n"
            "    )\n"
            "    chunk = _StreamChunk(\n"
            "        id=chunk_id,\n"
            '        object="chat.completion.chunk",\n'
            "        created=created,\n"
            "        model=model,\n"
            "        choices=[choice],\n"
            "        usage=usage,\n"
            "        prompt_token_ids=prompt_token_ids,\n"
            "    )\n"
            "    return (_SSE_DATA_B + _stream_encoder.encode(chunk) + _SSE_NL_B).decode()"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched _fast_sse_content (token_ids on choice, prompt_token_ids on chunk)")
        else:
            logger.warning("Could not find _fast_sse_content anchor")

        # ----- _generate_chat_stream streaming state -----

        # 4. Add token-ID state dicts just before stream_started = False.
        old = "        stream_started = False\n" "        try:"
        new = (
            "        _pending_token_ids = {}  # Buffer delta token_ids until a chunk is yielded\n"
            "        _n_prev_ids = {}  # Track cumulative output_ids length per index\n"
            "        _first_chunk_sent = {}\n"
            "        stream_started = False\n"
            "        try:"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched streaming state dicts")
        else:
            logger.warning("Could not find stream_started anchor in streaming code")

        # 5. Replace the first-chunk yield with one that:
        #    (a) extracts delta token IDs from output_ids before any yield, and
        #    (b) injects prompt_token_ids into the first (role-announcement) chunk.
        old = (
            "                # First chunk with role\n"
            "                if is_firsts.get(index, True):\n"
            "                    is_firsts[index] = False\n"
            "                    yield _fast_sse_content(\n"
            '                        chunk_id=content["meta_info"]["id"],\n'
            "                        created=int(time.time()),\n"
            "                        model=request.model,\n"
            "                        index=index,\n"
            '                        role="assistant",\n'
            '                        content="",\n'
            "                    )\n"
            "                    stream_started = True"
        )
        new = (
            "                # Extract delta token IDs from output_ids.\n"
            "                # output_ids is cumulative when stream_output=False (the default);\n"
            "                # slice past the previous length to get the delta.\n"
            '                _all_ids = content.get("output_ids") or []\n'
            "                if _all_ids:\n"
            '                    if getattr(self.tokenizer_manager.server_args, "stream_output", False):\n'
            "                        _delta_ids = list(_all_ids)\n"
            "                    else:\n"
            "                        _prev = _n_prev_ids.get(index, 0)\n"
            "                        _delta_ids = list(_all_ids[_prev:])\n"
            "                        _n_prev_ids[index] = len(_all_ids)\n"
            "                    if _delta_ids:\n"
            "                        _pending_token_ids.setdefault(index, []).extend(_delta_ids)\n"
            "\n"
            "                # First chunk with role — include prompt_token_ids for RL trace capture\n"
            "                if is_firsts.get(index, True):\n"
            "                    is_firsts[index] = False\n"
            "                    _first_chunk_sent[index] = True\n"
            "                    _prompt_ids = None\n"
            "                    _p_ids = adapted_request.input_ids\n"
            "                    if _p_ids and not isinstance(_p_ids, str):\n"
            "                        _p_ids = list(_p_ids[0] if isinstance(_p_ids[0], list) else _p_ids)\n"
            "                        _prompt_ids = _p_ids\n"
            "                    yield _fast_sse_content(\n"
            '                        chunk_id=content["meta_info"]["id"],\n'
            "                        created=int(time.time()),\n"
            "                        model=request.model,\n"
            "                        index=index,\n"
            '                        role="assistant",\n'
            '                        content="",\n'
            "                        prompt_token_ids=_prompt_ids,\n"
            "                    )\n"
            "                    stream_started = True"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched first-chunk (prompt_token_ids) + delta token extraction")
        else:
            logger.warning("Could not find first-chunk yield anchor in streaming code")

        # 6. Tool-call branch: pop pending token_ids, pass to _process_tool_call_stream,
        #    re-buffer if nothing was yielded.
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
            "                        continuous_usage_stats,\n"
            "                    ):\n"
            "                        if chunk:\n"
            "                            yield chunk"
        )
        new = (
            "                # Handle tool calls\n"
            "                if (\n"
            '                    request.tool_choice != "none"\n'
            "                    and request.tools\n"
            "                    and self.tool_call_parser\n"
            "                ):\n"
            "                    _flush_ids = _pending_token_ids.pop(index, None)\n"
            "                    _yielded_tool = False\n"
            "                    async for chunk in self._process_tool_call_stream(\n"
            "                        index,\n"
            "                        delta,\n"
            "                        parser_dict,\n"
            "                        content,\n"
            "                        request,\n"
            "                        has_tool_calls,\n"
            "                        continuous_usage_stats,\n"
            "                        _flush_ids,\n"
            "                    ):\n"
            "                        if chunk:\n"
            "                            _flush_ids = None  # only first yielded chunk carries IDs\n"
            "                            _yielded_tool = True\n"
            "                            yield chunk\n"
            "                    # If nothing was yielded, put the IDs back for next iteration\n"
            "                    if not _yielded_tool and _flush_ids is not None:\n"
            "                        _pending_token_ids.setdefault(index, []).extend(_flush_ids)"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched tool-call branch (delta token_ids forwarding)")
        else:
            logger.warning("Could not find tool-call branch anchor in streaming code")

        # 7. Regular-content branch: flush pending token_ids onto the yielded chunk.
        #    Empty delta (detokenizer boundary or EOS) leaves IDs buffered.
        old = (
            "                else:\n"
            "                    # Regular content\n"
            "                    if delta:\n"
            "                        usage = None\n"
            "                        if continuous_usage_stats:\n"
            "                            usage = UsageProcessor.calculate_token_usage(\n"
            "                                prompt_tokens=prompt_tokens.get(index, 0),\n"
            "                                reasoning_tokens=reasoning_tokens.get(index, 0),\n"
            "                                completion_tokens=completion_tokens.get(index, 0),\n"
            "                            ).model_dump()\n"
            "\n"
            "                        yield _fast_sse_content(\n"
            '                            chunk_id=content["meta_info"]["id"],\n'
            "                            created=int(time.time()),\n"
            "                            model=request.model,\n"
            "                            index=index,\n"
            "                            content=delta,\n"
            "                            logprobs=choice_logprobs,\n"
            "                            usage=usage,\n"
            "                        )"
        )
        new = (
            "                else:\n"
            "                    # Regular content — flush pending token_ids only when a chunk is\n"
            "                    # actually emitted. Empty delta (detokenizer boundary or EOS)\n"
            "                    # leaves IDs buffered so the finish chunk picks them up.\n"
            "                    if delta:\n"
            "                        _flush_ids = _pending_token_ids.pop(index, None)\n"
            "                        usage = None\n"
            "                        if continuous_usage_stats:\n"
            "                            usage = UsageProcessor.calculate_token_usage(\n"
            "                                prompt_tokens=prompt_tokens.get(index, 0),\n"
            "                                reasoning_tokens=reasoning_tokens.get(index, 0),\n"
            "                                completion_tokens=completion_tokens.get(index, 0),\n"
            "                            ).model_dump()\n"
            "\n"
            "                        yield _fast_sse_content(\n"
            '                            chunk_id=content["meta_info"]["id"],\n'
            "                            created=int(time.time()),\n"
            "                            model=request.model,\n"
            "                            index=index,\n"
            "                            content=delta,\n"
            "                            logprobs=choice_logprobs,\n"
            "                            usage=usage,\n"
            "                            token_ids=_flush_ids,\n"
            "                        )"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched regular-content branch (token_ids flush)")
        else:
            logger.warning("Could not find regular-content branch anchor in streaming code")

        # 8. Finish-reason chunks: flush remaining buffered token_ids (EOS has no delta).
        old = (
            "            for idx, finish_reason_data in finish_reasons.items():\n"
            '                finish_reason_type = finish_reason_data["type"]\n'
            "\n"
            '                # Change finish_reason to "tool_calls" if we had tool calls and stopped naturally\n'
            "                final_finish_reason = finish_reason_type\n"
            '                if has_tool_calls.get(idx, False) and finish_reason_type == "stop":\n'
            '                    final_finish_reason = "tool_calls"\n'
            "\n"
            '                matched_stop = finish_reason_data.get("matched")\n'
            "                yield _fast_sse_content(\n"
            '                    chunk_id=content["meta_info"]["id"],\n'
            "                    created=int(time.time()),\n"
            "                    model=request.model,\n"
            "                    index=idx,\n"
            "                    finish_reason=final_finish_reason,\n"
            "                    matched_stop=matched_stop,\n"
            "                )"
        )
        new = (
            "            for idx, finish_reason_data in finish_reasons.items():\n"
            '                finish_reason_type = finish_reason_data["type"]\n'
            "\n"
            '                # Change finish_reason to "tool_calls" if we had tool calls and stopped naturally\n'
            "                final_finish_reason = finish_reason_type\n"
            '                if has_tool_calls.get(idx, False) and finish_reason_type == "stop":\n'
            '                    final_finish_reason = "tool_calls"\n'
            "\n"
            "                # Flush any remaining buffered token_ids (e.g. EOS token has no text delta)\n"
            "                _final_ids = _pending_token_ids.pop(idx, None)\n"
            "\n"
            '                matched_stop = finish_reason_data.get("matched")\n'
            "                yield _fast_sse_content(\n"
            '                    chunk_id=content["meta_info"]["id"],\n'
            "                    created=int(time.time()),\n"
            "                    model=request.model,\n"
            "                    index=idx,\n"
            "                    finish_reason=final_finish_reason,\n"
            "                    matched_stop=matched_stop,\n"
            "                    token_ids=_final_ids,\n"
            "                )"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched finish-reason chunks (residual token_ids flush)")
        else:
            logger.warning("Could not find finish-reason chunk anchor in streaming code")

        # ----- Non-streaming path -----

        # 9. Inject prompt_token_ids into the non-streaming response.
        old = (
            "        response = self._build_chat_response(\n"
            "            request,\n"
            "            ret,\n"
            "            int(time.time()),\n"
            "        )\n"
            "\n"
            "        return response"
        )
        new = (
            "        response = self._build_chat_response(\n"
            "            request,\n"
            "            ret,\n"
            "            int(time.time()),\n"
            "        )\n"
            "\n"
            "        # Add prompt_token_ids for RL training trace capture\n"
            "        if isinstance(response, ChatCompletionResponse):\n"
            "            _p_ids = adapted_request.input_ids\n"
            "            if _p_ids and not isinstance(_p_ids, str):\n"
            "                _p_ids = list(_p_ids[0] if isinstance(_p_ids[0], list) else _p_ids)\n"
            "                response.prompt_token_ids = _p_ids\n"
            "\n"
            "        return response"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched non-streaming prompt_token_ids injection")
        else:
            logger.warning("Could not find non-streaming response return anchor")

        # 10. Extract completion token_ids per choice in _build_chat_response.
        old = (
            "            choice_data = ChatCompletionResponseChoice(\n"
            "                index=idx,\n"
            "                message=ChatMessage(\n"
            '                    role="assistant",\n'
            "                    content=text if text else None,\n"
            "                    tool_calls=tool_calls,\n"
            "                    reasoning_content=reasoning_text if reasoning_text else None,\n"
            "                ),\n"
            "                logprobs=choice_logprobs,\n"
            '                finish_reason=finish_reason["type"] if finish_reason else None,\n'
            "                matched_stop=(\n"
            '                    finish_reason["matched"]\n'
            '                    if finish_reason and "matched" in finish_reason\n'
            "                    else None\n"
            "                ),\n"
            "                hidden_states=hidden_states,\n"
            "            )\n"
            "            choices.append(choice_data)"
        )
        new = (
            "            # Completion token IDs — always present, no logprobs required\n"
            '            _token_ids = list(ret_item.get("output_ids") or []) or None\n'
            "\n"
            "            choice_data = ChatCompletionResponseChoice(\n"
            "                index=idx,\n"
            "                message=ChatMessage(\n"
            '                    role="assistant",\n'
            "                    content=text if text else None,\n"
            "                    tool_calls=tool_calls,\n"
            "                    reasoning_content=reasoning_text if reasoning_text else None,\n"
            "                ),\n"
            "                logprobs=choice_logprobs,\n"
            '                finish_reason=finish_reason["type"] if finish_reason else None,\n'
            "                matched_stop=(\n"
            '                    finish_reason["matched"]\n'
            '                    if finish_reason and "matched" in finish_reason\n'
            "                    else None\n"
            "                ),\n"
            "                hidden_states=hidden_states,\n"
            "                token_ids=_token_ids,\n"
            "            )\n"
            "            choices.append(choice_data)"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched non-streaming choice constructor (token_ids)")
        else:
            logger.warning("Could not find non-streaming choice constructor anchor")

        # ----- Tool-call parser: signature + token_ids attachment -----

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
            "        continuous_usage_stats: bool = False,\n"
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
            "        continuous_usage_stats: bool = False,\n"
            "        delta_token_ids=None,\n"
            "    ):"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched _process_tool_call_stream signature")
        else:
            logger.warning("Could not find _process_tool_call_stream signature anchor")

        # 12. Attach delta_token_ids to the normal-text Pydantic chunk in the tool-call parser.
        old = (
            "        # Yield normal text\n"
            "        if normal_text:\n"
            "            choice_data = ChatCompletionResponseStreamChoice(\n"
            "                index=index,\n"
            "                delta=DeltaMessage(content=normal_text),\n"
            "                finish_reason=None,\n"
            "            )\n"
            "            chunk = ChatCompletionStreamResponse(\n"
            '                id=content["meta_info"]["id"],\n'
            "                created=int(time.time()),\n"
            "                choices=[choice_data],\n"
            "                model=request.model,\n"
            "            )"
        )
        new = (
            "        # Yield normal text\n"
            "        if normal_text:\n"
            "            choice_data = ChatCompletionResponseStreamChoice(\n"
            "                index=index,\n"
            "                delta=DeltaMessage(content=normal_text),\n"
            "                finish_reason=None,\n"
            "                token_ids=delta_token_ids,\n"
            "            )\n"
            "            chunk = ChatCompletionStreamResponse(\n"
            '                id=content["meta_info"]["id"],\n'
            "                created=int(time.time()),\n"
            "                choices=[choice_data],\n"
            "                model=request.model,\n"
            "            )\n"
            "            delta_token_ids = None  # only first yielded chunk carries IDs"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched tool-call parser normal-text chunk (token_ids)")
        else:
            logger.warning("Could not find tool-call parser normal-text chunk anchor")

        # 13. Attach delta_token_ids to the tool-call Pydantic chunk in the parser.
        old = (
            "            choice_data = ChatCompletionResponseStreamChoice(\n"
            "                index=index,\n"
            "                delta=DeltaMessage(tool_calls=[tool_call]),\n"
            "                finish_reason=None,\n"
            "            )\n"
            "            chunk = ChatCompletionStreamResponse(\n"
            '                id=content["meta_info"]["id"],\n'
            "                created=int(time.time()),\n"
            "                choices=[choice_data],\n"
            "                model=request.model,\n"
            "            )"
        )
        new = (
            "            choice_data = ChatCompletionResponseStreamChoice(\n"
            "                index=index,\n"
            "                delta=DeltaMessage(tool_calls=[tool_call]),\n"
            "                finish_reason=None,\n"
            "                token_ids=delta_token_ids,\n"
            "            )\n"
            "            chunk = ChatCompletionStreamResponse(\n"
            '                id=content["meta_info"]["id"],\n'
            "                created=int(time.time()),\n"
            "                choices=[choice_data],\n"
            "                model=request.model,\n"
            "            )\n"
            "            delta_token_ids = None  # only first yielded chunk carries IDs"
        )
        if old in content:
            content = content.replace(old, new, 1)
            logger.info("Patched tool-call parser tool-call chunk (token_ids)")
        else:
            logger.warning("Could not find tool-call parser tool-call chunk anchor")

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
