"""vLLM plugin: per-token sampler draws (``SamplingBackend`` ``num_auxiliary_token_samples``).

With ``AGENTCORE_VLLM_TOKEN_SAMPLES=1``, a sampled-token ``logprobs=K`` request returns
``K`` i.i.d. draws from the processed sampling distribution in place of the top-``K``
head, for every request the server serves. Request ``flat_logprobs=True`` to keep
duplicate draws. vLLM loads ``register`` in each of its processes through the
``vllm.general_plugins`` entry point.
"""

from __future__ import annotations

import os
import threading

import numpy as np

SAMPLES_ENV = "AGENTCORE_VLLM_TOKEN_SAMPLES"
_MIXED_BATCH = "token-sample draws cannot share a vLLM batch with logprob_token_ids requests"
_SAMPLING = threading.local()


def samples_enabled() -> bool:
    return os.environ.get(SAMPLES_ENV, "") == "1"


def sample_rows(flat_logprobs, k: int) -> np.ndarray:
    """The ``(n, k)`` draws of a ``FlatLogprobs`` (entry 0 of each position is the sampled token)."""
    starts, ends = flat_logprobs.start_indices, flat_logprobs.end_indices
    if any(end - start != k + 1 for start, end in zip(starts, ends, strict=True)):
        raise RuntimeError(f"vLLM returned a position without exactly {k} token samples")
    return np.asarray(flat_logprobs.token_ids, dtype=np.int32).reshape(len(starts), k + 1)[:, 1:]


def _draw(logits, k: int):
    import torch

    return torch.multinomial(torch.softmax(logits.float(), dim=-1), k, replacement=True)


def install_sampler_patch() -> None:
    """Draw instead of taking the top-k head in both vLLM model runners' samplers.

    Requires ``logprobs_mode=processed_logprobs`` so the draws come from the
    distribution after temperature and top-k/top-p.
    """
    import torch
    from vllm.v1.outputs import LogprobsTensors
    from vllm.v1.sample import sampler as v1_sampler

    sampler_cls = v1_sampler.Sampler
    if getattr(sampler_cls.forward, "_agentcore_token_samples", False):
        return
    original_forward = sampler_cls.forward
    original_gather = sampler_cls.gather_logprobs

    def forward(self, logits, sampling_metadata, *args, **kwargs):
        if sampling_metadata.logprob_token_ids and sampling_metadata.max_num_logprobs:
            raise RuntimeError(_MIXED_BATCH)
        _SAMPLING.active = True  # prompt-logprob gathers outside forward keep the top-k head
        try:
            return original_forward(self, logits, sampling_metadata, *args, **kwargs)
        finally:
            _SAMPLING.active = False

    def gather_logprobs(logprobs, num_logprobs, token_ids):
        if not getattr(_SAMPLING, "active", False) or num_logprobs <= 0:
            return original_gather(logprobs, num_logprobs, token_ids)
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)
        torch._dynamo.decorators.mark_unbacked(logprobs, 0)
        torch._dynamo.decorators.mark_unbacked(token_logprobs, 0)
        ranks = v1_sampler.batched_count_greater_than(logprobs, token_logprobs)
        indices = torch.cat((token_ids, _draw(logprobs, num_logprobs)), dim=1)
        return LogprobsTensors(indices.to(torch.int32), logprobs.gather(-1, indices), ranks)

    forward._agentcore_token_samples = True
    sampler_cls.forward = forward
    sampler_cls.gather_logprobs = staticmethod(gather_logprobs)

    try:
        from vllm.v1.worker.gpu.sample import logprob as v2_logprob
        from vllm.v1.worker.gpu.sample import sampler as v2_sampler
    except ImportError:
        return
    original = v2_logprob.compute_topk_logprobs

    def compute_topk_logprobs(logits, num_logprobs, sampled_token_ids, cu_num_logits=None, **kwargs):
        if num_logprobs == 0:
            return original(logits, num_logprobs, sampled_token_ids, cu_num_logits, **kwargs)
        if kwargs.get("max_per_req_token_ids", 0):
            raise RuntimeError(_MIXED_BATCH)
        ids = torch.cat((sampled_token_ids.unsqueeze(-1), _draw(logits, num_logprobs)), dim=1)
        batch_size, vocab_size = logits.shape
        ranks = torch.empty(batch_size, dtype=torch.int64, device=logits.device)
        v2_logprob._ranks_kernel[(batch_size,)](
            ranks, logits, logits.stride(0), sampled_token_ids, vocab_size, BLOCK_SIZE=8192
        )
        return LogprobsTensors(ids, v2_logprob.compute_token_logprobs(logits, ids), ranks, cu_num_logits)

    v2_sampler.compute_topk_logprobs = compute_topk_logprobs


def install_generate_endpoint_patch() -> None:
    """Keep duplicate draws in ``/inference/v1/generate``, which keys each position's logprobs by token id."""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionLogProb,
        ChatCompletionLogProbs,
        ChatCompletionLogProbsContent,
    )
    from vllm.entrypoints.serve.disagg.serving import ServingTokens
    from vllm.logprobs import FlatLogprobs

    original = ServingTokens._create_tokens_logprobs
    if getattr(original, "_agentcore_token_samples", False):
        return

    def create_tokens_logprobs(self, token_ids, top_logprobs, num_output_top_logprobs=None):
        if not isinstance(top_logprobs, FlatLogprobs):
            return original(self, token_ids, top_logprobs, num_output_top_logprobs)
        content = []
        for i, token_id in enumerate(token_ids):
            start, end = top_logprobs.start_indices[i], top_logprobs.end_indices[i]
            entries = [
                ChatCompletionLogProb(token=f"token_id:{tid}", logprob=max(lp, -9999.0))
                for tid, lp in zip(top_logprobs.token_ids[start:end], top_logprobs.logprobs[start:end], strict=True)
            ]
            content.append(
                ChatCompletionLogProbsContent(
                    token=f"token_id:{token_id}", logprob=entries[0].logprob, top_logprobs=entries
                )
            )
        return ChatCompletionLogProbs(content=content)

    create_tokens_logprobs._agentcore_token_samples = True
    ServingTokens._create_tokens_logprobs = create_tokens_logprobs


def register() -> None:
    """``vllm.general_plugins`` entry point."""
    if samples_enabled():
        install_sampler_patch()
        install_generate_endpoint_patch()
