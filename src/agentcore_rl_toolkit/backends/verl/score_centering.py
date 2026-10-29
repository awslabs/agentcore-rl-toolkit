"""Score function centering (SFC) for verl 0.9.0 from per-token sampler draws.

For a sampled token ``a`` with ``K`` draws ``v_k`` from the sampler ``q`` at the same
position, the loss is

    c = z_θ(a) − mean_k z_θ(v_k)
    L = −A · c  −  β_e · mean_k log p_θ(v_k),    β_e = β₀ / √e  (0 for fresh tokens)

The first term is the centered score of
https://normaluhr.github.io/en/p/score-function-centering/ (no importance ratio). The
second is a sampler-learner KL(q ‖ p_θ) as in Tinker's StableReinforce
(https://tinker-docs.thinkingmachines.ai/tinker/losses/stable-reinforce/), weighted by
token age ``e`` in optimizer steps as in
https://normaluhr.github.io/en/p/score-function-centering-experiments/.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any

import numpy as np

from agentcore_rl_toolkit.rollout_gateway.sampling_backends.vllm_token_samples import (
    SAMPLES_ENV,
    sample_rows,
    samples_enabled,
)

logger = logging.getLogger(__name__)

# Per-token fields aligned to logits rows (row t predicts token t + 1).
SAMPLE_IDS_KEY = "rollout_sample_ids"
SAMPLE_VERSIONS_KEY = "rollout_sample_versions"
# Micro-batch metadata set by ScoreCenteringMixin.
KL_BETA_KEY = "sfc_kl_beta"
TRAINER_STEP_KEY = "sfc_trainer_step"
SYNC_STEP_KEY = "sfc_parameter_sync_step"
_SCORE_KEY = "sc_score"
_SAMPLE_FIELDS = ("response_sample_ids", "response_sample_versions")

_REQUEST: contextvars.ContextVar[dict | None] = contextvars.ContextVar("agentcore_token_samples", default=None)
_SEGMENTS: contextvars.ContextVar[list | None] = contextvars.ContextVar("agentcore_sample_segments", default=None)
# Patched methods live at module level so Ray pickles them by reference.
_ORIGINAL: dict[str, Any] = {}


def _sampling_params(*args, **kwargs):
    request = _REQUEST.get()
    if request is not None and kwargs.get("logprobs") is not None:
        kwargs.update(logprobs=request["k"], flat_logprobs=True)
    return _ORIGINAL["SamplingParams"](*args, **kwargs)


async def _engine_generate(self, *args, **kwargs):
    request = _REQUEST.get()
    async for output in _ORIGINAL["AsyncLLM.generate"](self, *args, **kwargs):
        if request is not None:
            request["final"] = output
        yield output


async def _server_generate(self, prompt_ids, sampling_params, request_id, *args, **kwargs):
    generate = _ORIGINAL["vLLMHttpServer.generate"]
    k = int(sampling_params.pop("num_auxiliary_token_samples", 0) or 0)
    if k <= 0:
        return await generate(self, prompt_ids, sampling_params, request_id, *args, **kwargs)
    if not samples_enabled():
        raise RuntimeError(f"num_auxiliary_token_samples needs {SAMPLES_ENV}=1 on the rollout workers")
    request = {"k": k, "final": None}
    token = _REQUEST.set(request)
    try:
        output = await generate(self, prompt_ids, sampling_params, request_id, *args, **kwargs)
    finally:
        _REQUEST.reset(token)
    if output.token_ids:
        ids = sample_rows(request["final"].outputs[0].logprobs, k)
        version = output.extra_fields.get("global_steps")
        output.extra_fields["response_sample_ids"] = ids
        output.extra_fields["response_sample_versions"] = np.full(len(ids), -1 if version is None else version)
    return output


async def _client_generate(self, *args, **kwargs):
    output = await _ORIGINAL["LLMServerClient.generate"](self, *args, **kwargs)
    segments = _SEGMENTS.get()
    if segments is not None and "response_sample_ids" in output.extra_fields:
        segments.append({key: output.extra_fields[key] for key in _SAMPLE_FIELDS})
    return output


async def _async_client_generate(self, *args, **kwargs):
    # verl concatenates the token ids of resumed generations but drops their extra fields.
    token = _SEGMENTS.set([])
    try:
        output = await _ORIGINAL["FullyAsyncLLMServerClient.generate"](self, *args, **kwargs)
        segments = _SEGMENTS.get()
    finally:
        _SEGMENTS.reset(token)
    if segments:
        for key in _SAMPLE_FIELDS:
            output.extra_fields[key] = np.concatenate([segment[key] for segment in segments])
    return output


def install() -> None:
    """Patch verl's rollout server and async client to return and keep the draws.

    The draws themselves come from the ``vllm_token_samples`` vLLM plugin.
    """
    try:
        from verl.workers.rollout import llm_server
        from verl.workers.rollout.vllm_rollout import vllm_async_server as server
        from vllm.v1.engine.async_llm import AsyncLLM
    except ImportError as e:
        logger.debug("token-sample patches skipped: %s", e)
        return
    if not _ORIGINAL:
        _ORIGINAL.update(
            {
                "LLMServerClient.generate": llm_server.LLMServerClient.generate,
                "FullyAsyncLLMServerClient.generate": llm_server.FullyAsyncLLMServerClient.generate,
                "SamplingParams": server.SamplingParams,
                "AsyncLLM.generate": AsyncLLM.generate,
                "vLLMHttpServer.generate": server.vLLMHttpServer.generate,
            }
        )
        llm_server.LLMServerClient.generate = _client_generate
        llm_server.FullyAsyncLLMServerClient.generate = _async_client_generate
        server.SamplingParams = _sampling_params
        AsyncLLM.generate = _engine_generate
        server.vLLMHttpServer.generate = _server_generate


def sfc_settings(config) -> tuple[int, float]:
    """``(num_auxiliary_token_samples, kl_beta)`` from the trainer config's ``score_centering`` node."""
    node = config.get("score_centering") or {}
    k, beta = node.get("num_auxiliary_token_samples", 0), node.get("kl_beta", 0.0)
    if type(k) is not int or k < 0:  # noqa: E721 - reject bool
        raise ValueError(f"score_centering.num_auxiliary_token_samples must be a non-negative integer, got {k!r}")
    if isinstance(beta, bool) or not isinstance(beta, int | float) or beta < 0:
        raise ValueError(f"score_centering.kl_beta must be a non-negative number, got {beta!r}")
    return k, float(beta)


def sample_fields(auxiliary_token_samples, k: int, prompt_length: int, response_length: int) -> dict:
    """The draws and their weight versions per logits row: response token ``r`` is row ``prompt_length - 1 + r``."""
    import torch

    seq_len = prompt_length + response_length
    draws = torch.zeros((seq_len, k), dtype=torch.int32)
    versions = torch.full((seq_len,), -1, dtype=torch.int64)
    for offset, samples in auxiliary_token_samples:
        start = prompt_length - 1 + offset
        draws[start : start + len(samples)] = torch.as_tensor(np.asarray(samples.ids), dtype=torch.int32)
        if samples.versions is not None:
            versions[start : start + len(samples)] = torch.as_tensor(np.asarray(samples.versions), dtype=torch.int64)
    return {SAMPLE_IDS_KEY: draws, SAMPLE_VERSIONS_KEY: versions}


def _score_from_logits(student_logits, data) -> dict:
    import torch
    from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size

    if get_ulysses_sequence_parallel_world_size() > 1:
        raise NotImplementedError("score centering does not support Ulysses sequence parallelism")
    logits = student_logits.squeeze(0)
    labels = torch.roll(data["input_ids"].values(), shifts=-1, dims=0)
    draws = data[SAMPLE_IDS_KEY].values().to(device=logits.device, dtype=torch.int64)
    pad = logits.shape[0] - labels.shape[0]  # bucket padding, dropped by the engine
    if pad:
        labels = torch.nn.functional.pad(labels, (0, pad))
        draws = torch.nn.functional.pad(draws, (0, 0, 0, pad))
    sampled = logits.gather(-1, labels.to(logits.device).unsqueeze(-1)).squeeze(-1)
    return {_SCORE_KEY: (sampled.float() - logits.gather(-1, draws).float().mean(-1)).unsqueeze(0)}


def _token_ages(data, response_mask):
    """Age ``e`` in optimizer steps of each response token: ``(g − 1 − v) · P + k``."""
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import no_padding_2_padding

    versions = no_padding_2_padding(data[SAMPLE_VERSIONS_KEY], data).to(response_mask.device)
    if (response_mask & (versions < 0)).any():
        raise RuntimeError("the score-centering KL needs the sampler weight version of every trained token")
    trainer_step = tu.get_non_tensor_data(data=data, key=TRAINER_STEP_KEY, default=0)
    sync_step = tu.get_non_tensor_data(data=data, key=SYNC_STEP_KEY, default=1)
    return (trainer_step - versions * sync_step).clamp(min=0).float()


def score_centering_loss(config, model_output=None, data=None, dp_group=None, student_logits=None):
    """Replaces verl's ``ppo_loss``; also called as the engine's logits processor."""
    import torch
    from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
    from verl.utils import tensordict_utils as tu
    from verl.utils.metric import AggregationType, Metric
    from verl.utils.torch_functional import masked_mean
    from verl.workers.utils.padding import no_padding_2_padding

    if student_logits is not None:
        return _score_from_logits(student_logits, data)

    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    score = no_padding_2_padding(model_output[_SCORE_KEY], data)
    entropy = model_output.get("entropy")
    beta = float(tu.get_non_tensor_data(data=data, key=KL_BETA_KEY, default=0.0) or 0.0)

    config.global_batch_info.update(
        dp_size=data["dp_size"],
        batch_num_tokens=data["batch_num_tokens"],
        global_batch_size=data["global_batch_size"],
        loss_scale_factor=config.loss_scale_factor,
    )
    normalized = (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    )
    aggregation = AggregationType.SUM if normalized else AggregationType.MEAN
    padded = data.select(*[k for k in ("response_mask", "old_log_probs", "advantages", "ref_log_prob") if k in data])
    padded = padded.to_padded_tensor()
    mask = padded["response_mask"].bool()
    ages = _token_ages(data, mask) if beta else None

    def aggregate(loss_mat):
        return agg_loss(
            loss_mat=loss_mat, loss_mask=mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

    # Zero-valued surrogate whose gradient is −A·∇c.
    loss = aggregate(-padded["advantages"] * (score - score.detach()))
    metrics = {
        "actor/ppo_kl": Metric(AggregationType.MEAN, masked_mean(padded["old_log_probs"] - log_prob.detach(), mask)),
        "actor/sfc_centered_score": Metric(AggregationType.MEAN, masked_mean(score.detach(), mask)),
    }

    if ages is not None:
        # −β_e · mean_k log p(v_k), with mean_k log p(v_k) = log p(a) − c.
        weights = torch.where(ages >= 1, beta / ages.clamp(min=1).sqrt(), torch.zeros_like(ages))
        sample_kl = aggregate(weights * (score - log_prob))
        loss = loss + sample_kl
        metrics["actor/sfc_kl_loss"] = Metric(aggregation, sample_kl)
        metrics["actor/sfc_token_age"] = Metric(AggregationType.MEAN, masked_mean(ages, mask))

    if entropy is not None:
        entropy_loss = aggregate(no_padding_2_padding(entropy, data))
        loss = loss - config.entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(aggregation, entropy_loss)

    if config.use_kl_loss:
        kl_loss = aggregate(
            kl_penalty(logprob=log_prob, ref_logprob=padded["ref_log_prob"], kl_penalty=config.kl_loss_type)
        )
        loss = loss + kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(aggregation, kl_loss)
        metrics["kl_coef"] = config.kl_loss_coef

    return loss, metrics


def install_padding_patch() -> None:
    """Resize the per-token sample fields on verl's two-token batch-padding rows."""
    import torch
    from verl.trainer.ppo import padding_utils

    original = padding_utils.construct_minimal_padding_template
    if getattr(original, "_agentcore_token_samples", False):
        return

    def construct_minimal_padding_template(source_td, source_tag, eos_token_id):
        sample, tag = original(source_td, source_tag, eos_token_id)
        seq_len = sample["input_ids"].size(0)
        if SAMPLE_IDS_KEY in sample:
            sample[SAMPLE_IDS_KEY] = torch.zeros((seq_len, sample[SAMPLE_IDS_KEY].shape[-1]), dtype=torch.int32)
            sample[SAMPLE_VERSIONS_KEY] = torch.full((seq_len,), -1, dtype=torch.int64)
        return sample, tag

    construct_minimal_padding_template._agentcore_token_samples = True
    padding_utils.construct_minimal_padding_template = construct_minimal_padding_template
