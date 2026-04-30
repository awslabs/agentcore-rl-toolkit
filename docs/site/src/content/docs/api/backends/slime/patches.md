---
title: SGLang patches
description: Apply + verify the SGLang token-IDs patch used by the slime backend.
sidebar:
  order: 4
---


## `agentcore_rl_toolkit.backends.slime.patches.sglang_token_ids`

Apply SGLang v0.5.9 patch to expose prompt/completion token IDs in chat completions.

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

## `apply_patch() -> bool`

Patch the installed SGLang package.

Returns True if either file was patched, False if both were already
patched, SGLang wasn't found, or patching failed.

## `agentcore_rl_toolkit.backends.slime.patches.verify_sglang_token_ids`

Verify the SGLang v0.5.9 token_ids patch.

Launches a local SGLang server with the given model, issues greedy-decoded
chat completion requests in both streaming and non-streaming modes, and
checks that:

1. Non-streaming responses expose prompt_token_ids (on the response) and
   token_ids (per choice), and decoding token_ids matches the content.
2. Streaming responses expose prompt_token_ids on the first chunk and
   token_ids on content chunks, and decoding the concatenation matches
   the concatenated deltas.
3. Under greedy decoding, the prompt and completion token sequences are
   identical between streaming and non-streaming modes.
4. Streaming with tools active yields the same completion token sequence
   as streaming without tools, validating the _pending_token_ids buffering.

## `main(argv: Optional[list[str]] = None) -> int`
