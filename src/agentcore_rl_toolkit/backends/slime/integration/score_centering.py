"""Score function centering (SFC) for slime, following Tinker's StableReinforce:
https://tinker-docs.thinkingmachines.ai/tinker/losses/stable-reinforce/

    python3 -m agentcore_rl_toolkit.backends.slime.train ${SLIME_DIR}/train.py ... \
        --loss-type custom_loss \
        --custom-loss-function-path \
            agentcore_rl_toolkit.backends.slime.integration.score_centering.score_centering_loss \
        --custom-megatron-init-path \
            agentcore_rl_toolkit.backends.slime.integration.score_centering.install \
        --sfc-num-auxiliary-token-samples 8 --sfc-kl-beta 0.3

For a sampled token ``a`` with ``K`` draws ``v_k`` from the same sampler distribution:

    c = z_θ(a) − mean_k z_θ(v_k)
    L = −A · c  −  β_e · mean_k log p_θ(v_k),    β_e = β₀ / √age  (0 for fresh tokens)

where ``age`` is the optimizer steps since the sampling weights were pushed. The draws
come from the ``sglang_token_samples`` plugin and reach the trainer in
``Sample.multimodal_train_inputs``, the per-sample dict slime forwards unchanged.
"""

from __future__ import annotations

import dataclasses
from argparse import Namespace
from typing import Any

import numpy as np

from agentcore_rl_toolkit.rollout_gateway.sampling_backends.sglang_token_samples import SAMPLES_ENV, samples_enabled

SAMPLE_IDS_KEY = "agentcore_auxiliary_token_sample_ids"
SAMPLE_VERSIONS_KEY = "agentcore_auxiliary_token_sample_versions"
LOSS_PATH = f"{__name__}.score_centering_loss"


@dataclasses.dataclass
class _TrainState:
    steps: int = 0  # optimizer steps taken in this train actor
    push_steps: dict[int, int] = dataclasses.field(default_factory=dict)  # weight version -> steps at push


_STATE = _TrainState()
_ORIGINAL: dict[str, Any] = {}


def add_arguments(parser):
    """slime ``parse_args(add_custom_arguments=...)`` hook; see ``backends/slime/train.py``."""
    group = parser.add_argument_group("score function centering")
    group.add_argument("--sfc-num-auxiliary-token-samples", type=int, default=0, help="K sampler draws per token")
    group.add_argument("--sfc-kl-beta", type=float, default=0.0, help="β₀ of the token-age-weighted sampler KL")
    group.add_argument(
        "--sfc-loss-agg-mode",
        choices=("token-mean", "token-sum"),
        default="token-mean",
        help="per-rollout token mean, or token sum (verl's seq-mean-token-sum)",
    )
    return parser


def sfc_settings(args: Namespace) -> tuple[int, float]:
    """``(num_auxiliary_token_samples, kl_beta)``; zeros when the flags are not registered."""
    k, beta = getattr(args, "sfc_num_auxiliary_token_samples", 0), getattr(args, "sfc_kl_beta", 0.0)
    if k < 0 or beta < 0:
        raise ValueError("--sfc-num-auxiliary-token-samples and --sfc-kl-beta must be non-negative")
    return k, beta


def loss_reducer(args: Namespace, batch: dict, sum_of_sample_mean):
    """slime's per-rollout token mean, or a per-rollout token sum."""
    if getattr(args, "sfc_loss_agg_mode", "token-mean") == "token-mean":
        return sum_of_sample_mean
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean

    return get_sum_of_sample_mean(
        batch["total_lengths"], batch["response_lengths"], batch["loss_masks"], calculate_per_token_loss=True
    )


def validate(args: Namespace, *, rollout: bool = False) -> None:
    checks = {
        f"--loss-type custom_loss --custom-loss-function-path {LOSS_PATH}": args.loss_type == "custom_loss"
        and args.custom_loss_function_path == LOSS_PATH,
        "--rollout-top-p 1.0 and --rollout-top-k -1": args.rollout_top_p == 1.0 and args.rollout_top_k == -1,
        "no --allgather-cp": not getattr(args, "allgather_cp", False),
        "--entropy-coef 0": not getattr(args, "entropy_coef", 0.0),
    }
    if rollout:
        checks[f"{SAMPLES_ENV}=1 in the job's runtime environment"] = samples_enabled()
    if failed := [name for name, ok in checks.items() if not ok]:
        raise ValueError("score_centering requires: " + "; ".join(failed))


def train_inputs(record, k: int, *, require_versions: bool) -> dict[str, np.ndarray]:
    """``Sample.multimodal_train_inputs`` for a ``TraceRecord``: draws and weight versions per response token."""
    n = record.response_length
    ids = np.zeros((n, k), dtype=np.int32)
    versions = np.full(n, -1, dtype=np.int64)
    covered = np.zeros(n, dtype=bool)
    for offset, samples in record.auxiliary_token_samples:
        end = offset + len(samples)
        ids[offset:end] = np.asarray(samples.ids, dtype=np.int32).reshape(len(samples), k)
        if samples.versions is not None:
            versions[offset:end] = np.asarray(samples.versions, dtype=np.int64)
        covered[offset:end] = True
    trained = np.asarray(record.loss_mask, dtype=bool)
    if (trained & ~covered).any():
        raise RuntimeError("trained tokens without sampler draws")
    if require_versions and (trained & (versions < 0)).any():
        raise RuntimeError("score_centering.kl_beta needs the SGLang weight version of every trained token")
    return {SAMPLE_IDS_KEY: ids, SAMPLE_VERSIONS_KEY: versions}


def empty_train_inputs(response_length: int, k: int) -> dict[str, np.ndarray]:
    return {
        SAMPLE_IDS_KEY: np.zeros((response_length, k), dtype=np.int32),
        SAMPLE_VERSIONS_KEY: np.full(response_length, -1, dtype=np.int64),
    }


# Train actor patches. slime calls these through module attributes, so patching the
# module in the train actor process is enough.


def _get_batch(data_iterator, keys, *args, **kwargs):
    batch = _ORIGINAL["get_batch"](data_iterator, keys, *args, **kwargs)
    inputs = batch.get("multimodal_train_inputs")
    if inputs and SAMPLE_IDS_KEY in inputs:
        lengths = list(batch["response_lengths"])
        batch[SAMPLE_IDS_KEY] = list(inputs.pop(SAMPLE_IDS_KEY).split(lengths))
        batch[SAMPLE_VERSIONS_KEY] = list(inputs.pop(SAMPLE_VERSIONS_KEY).split(lengths))
        if not inputs:  # keep the draws out of the model's forward kwargs
            batch["multimodal_train_inputs"] = None
    return batch


def _train_one_step(*args, **kwargs):
    try:
        return _ORIGINAL["train_one_step"](*args, **kwargs)
    finally:
        _STATE.steps += 1


def _record_push(update_weights):
    def wrapper(self, *args, **kwargs):
        result = update_weights(self, *args, **kwargs)
        _STATE.push_steps[int(self.weight_version)] = _STATE.steps
        return result

    return wrapper


def install(args: Namespace) -> None:
    """``--custom-megatron-init-path`` hook."""
    validate(args)
    if _ORIGINAL:
        return
    from slime.backends.megatron_utils import model
    from slime.backends.megatron_utils.update_weight import update_weight_from_distributed, update_weight_from_tensor

    _ORIGINAL.update(get_batch=model.get_batch, train_one_step=model.train_one_step)
    model.get_batch = _get_batch
    model.train_one_step = _train_one_step
    for updater in (
        update_weight_from_tensor.UpdateWeightFromTensor,
        update_weight_from_distributed.UpdateWeightFromDistributed,
    ):
        updater.update_weights = _record_push(updater.update_weights)


# Loss


def _centered_scores(args: Namespace, batch: dict, logits) -> list:
    """Per-sample ``c`` from vocab-parallel logits ``[1, T, V/tp]``: ``K + 1`` gathers per row."""
    import torch
    from megatron.core import mpu
    from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
    from slime.backends.megatron_utils.loss import _build_shifted_tokens, _extract_per_sample

    logits = logits.squeeze(0)
    num_rows, vocab_local = logits.shape
    total_lengths, response_lengths = batch["total_lengths"], batch["response_lengths"]
    k = batch[SAMPLE_IDS_KEY][0].shape[-1]

    def row_targets(sequences):
        return _build_shifted_tokens(num_rows, logits.device, sequences, total_lengths, response_lengths, False)

    # Pad the draws to full sequences so they align to logits rows like the sampled tokens.
    padded = [
        torch.cat((ids.new_zeros(total - ids.shape[0], k), ids))
        for ids, total in zip(batch[SAMPLE_IDS_KEY], total_lengths, strict=True)
    ]
    targets = torch.stack(
        [row_targets(batch["unconcat_tokens"])] + [row_targets([seq[:, j] for seq in padded]) for j in range(k)],
        dim=1,
    )
    local = targets - mpu.get_tensor_model_parallel_rank() * vocab_local
    in_shard = (local >= 0) & (local < vocab_local)
    z = logits.gather(-1, local.clamp(0, vocab_local - 1)) * in_shard
    score = reduce_from_tensor_model_parallel_region(z[:, 0] - z[:, 1:].mean(-1)) / args.rollout_temperature
    return _extract_per_sample(score, None, total_lengths, response_lengths, False)[0]


def _token_ages(batch: dict, device):
    import torch
    from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

    versions = torch.cat(
        [
            slice_log_prob_with_cp(v, total, response)
            for v, total, response in zip(
                batch[SAMPLE_VERSIONS_KEY], batch["total_lengths"], batch["response_lengths"], strict=True
            )
        ]
    ).cpu()
    known = versions >= 0
    present = versions[known].unique().tolist()
    if missing := set(present) - _STATE.push_steps.keys():
        raise RuntimeError(f"no weight push recorded in this train actor for weight versions {sorted(missing)}")
    pushed = torch.full((max(present, default=0) + 1,), _STATE.steps, dtype=torch.float32)
    for version in present:
        pushed[version] = _STATE.push_steps[version]
    return torch.where(known, _STATE.steps - pushed[versions.clamp(min=0)], torch.zeros(())).to(device)


def score_centering_loss(args: Namespace, batch: dict, logits, sum_of_sample_mean):
    """``--custom-loss-function-path`` loss."""
    import torch
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy
    from slime.utils.ppo_utils import compute_approx_kl

    _, kl_beta = sfc_settings(args)
    _, out = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
    )
    log_probs = torch.cat(out["log_probs"], dim=0)
    score = torch.cat(_centered_scores(args, batch, logits), dim=0)
    advantages = torch.cat(batch["advantages"], dim=0)
    aggregate = loss_reducer(args, batch, sum_of_sample_mean)

    # Zero-valued surrogate whose gradient is −A·∇c.
    pg_loss = aggregate(-advantages * (score - score.detach()))
    loss = pg_loss
    metrics = {"pg_loss": pg_loss.detach(), "sfc_centered_score": sum_of_sample_mean(score.detach())}
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch.get("log_probs")
    if old_log_probs:
        metrics["ppo_kl"] = sum_of_sample_mean(torch.cat(old_log_probs, dim=0) - log_probs.detach())

    if kl_beta:
        # mean_k log p(v_k) = log p(a) − c
        ages = _token_ages(batch, log_probs.device)
        weights = torch.where(ages >= 1, kl_beta / ages.clamp(min=1).sqrt(), torch.zeros_like(ages))
        sample_kl = aggregate(weights * (score - log_probs))
        loss = loss + sample_kl
        metrics["sfc_kl_loss"] = sample_kl.detach()
        metrics["sfc_token_age"] = sum_of_sample_mean(ages)

    if args.use_kl_loss:
        kl = compute_approx_kl(log_probs, torch.cat(batch["ref_log_probs"], dim=0), kl_loss_type=args.kl_loss_type)
        kl_loss = aggregate(kl)
        loss = loss + args.kl_loss_coef * kl_loss
        metrics["kl_loss"] = kl_loss.detach()

    if log_probs.numel() == 0:  # keep the backward graph on ranks without response tokens
        loss = loss + 0 * logits.sum()
    return loss, {"loss": loss.detach(), **metrics}
