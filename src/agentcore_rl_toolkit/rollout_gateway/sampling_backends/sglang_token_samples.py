"""SGLang plugin for ``num_auxiliary_token_samples``, enabled by ``AGENTCORE_SGLANG_TOKEN_SAMPLES=1``.

A ``top_logprobs_num=K`` request then returns, per output position, the sampled token
followed by ``K`` draws from the temperature-scaled sampling distribution instead of the
top-``K`` tokens. Positions sampled with top-k, top-p, or min-p return only the sampled
token. Loaded in every SGLang process through the ``sglang.srt.plugins`` entry point.
"""

from __future__ import annotations

import os

SAMPLES_ENV = "AGENTCORE_SGLANG_TOKEN_SAMPLES"


def samples_enabled() -> bool:
    return os.environ.get(SAMPLES_ENV, "") == "1"


def draw_rows(logprobs, top_logprobs_nums: list[int], sampled_ids, untruncated: list[bool]):
    """Per-row ``(values, indices)``: the sampled token, then ``K`` draws from ``exp(logprobs)``."""
    import torch

    draws = torch.multinomial(logprobs.float().exp(), max(top_logprobs_nums), replacement=True)
    indices = torch.cat((sampled_ids.view(-1, 1).to(draws.dtype), draws), dim=1)
    values = logprobs.gather(-1, indices)
    widths = [(k + 1 if ok else 1) if k else 0 for k, ok in zip(top_logprobs_nums, untruncated, strict=True)]
    return [values[i, :w] for i, w in enumerate(widths)], [indices[i, :w] for i, w in enumerate(widths)]


def install_sampler_patch() -> None:
    """Draw instead of taking the top-k head in SGLang's sampler."""
    from sglang.srt.layers import sampler as sglang_sampler
    from sglang.srt.sampling.sampling_params import TOP_K_ALL

    sampler_cls = sglang_sampler.Sampler
    original = sampler_cls._attach_logprobs_to_output
    if getattr(original, "_agentcore_token_samples", False):
        return

    def _attach_logprobs_to_output(
        self, logits_output, logprobs, top_logprobs_nums, token_ids_logprobs, sampling_info, batch_next_token_ids
    ):
        # The original clamps -inf logprobs in place; zero top-k skips its head computation.
        original(
            self,
            logits_output,
            logprobs,
            [0] * len(top_logprobs_nums),
            token_ids_logprobs,
            sampling_info,
            batch_next_token_ids,
        )
        if not any(k > 0 for k in top_logprobs_nums):
            return
        untruncated = (
            (sampling_info.top_ks == TOP_K_ALL) & (sampling_info.top_ps >= 1.0) & (sampling_info.min_ps <= 0.0)
        ).tolist()
        (
            logits_output.next_token_top_logprobs_val,
            logits_output.next_token_top_logprobs_idx,
        ) = draw_rows(logprobs, top_logprobs_nums, batch_next_token_ids, untruncated)

    _attach_logprobs_to_output._agentcore_token_samples = True
    sampler_cls._attach_logprobs_to_output = _attach_logprobs_to_output


def register() -> None:
    """``sglang.srt.plugins`` entry point."""
    if samples_enabled():
        install_sampler_patch()
