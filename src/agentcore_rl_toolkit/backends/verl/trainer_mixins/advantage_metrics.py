"""``critic/advantages/zero_*``: how much of a step's loss signal carries no gradient.

``zero_mean`` is the fraction of valid advantage entries that are exactly zero;
``zero_pass_mean`` says which way those collapsed groups went (all-pass vs all-fail).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import transfer_queue as tq
from transfer_queue import KVBatchMeta

from .base import TrainerMixinBase

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AdvantageZeroMetricsMixin(TrainerMixinBase):
    """Adds ``critic/advantages/zero_mean`` and ``.../zero_pass_mean``."""

    def _compute_metrics(
        self,
        batch: KVBatchMeta,
        metrics: dict[str, Any],
        timing_raw: dict[str, Any],
        global_steps: int,
        epoch: int,
    ) -> None:
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)

        # Best-effort: never abort a training step on a diagnostic.
        try:
            metrics.update(self._advantage_zero_metrics(batch))
        except Exception as e:
            logger.exception("critic/advantages/zero_* metrics failed", exc_info=e)

    def _advantage_zero_metrics(self, batch: KVBatchMeta) -> dict[str, float]:
        """Fraction of valid (response-masked) advantage entries that are exactly zero.

        The step's TransferQueue entries are still live here (``fit()`` clears them after this
        call), so the fields can be re-fetched. Computed unconditionally, so an all-masked batch
        yields ``nan`` rather than a spurious 0.0.
        """
        non_padding = [not tag.get("is_padding", False) for tag in batch.tags]
        keys = [k for k, keep in zip(batch.keys, non_padding, strict=True) if keep]
        if not keys:
            return {}

        assert batch.partition_id is not None
        data = tq.kv_batch_get(
            keys=keys,
            partition_id=batch.partition_id,
            select_fields=["advantages", "response_mask", "rm_scores"],
        ).to_padded_tensor()

        response_mask = data["response_mask"].bool()
        valid_adv = torch.masked_select(data["advantages"], response_mask)
        out = {"critic/advantages/zero_mean": (valid_adv == 0).float().mean().detach().item()}

        # Guarded separately so a key-format surprise costs only this metric, not zero_mean above.
        try:
            out.update(self._advantage_zero_pass_metrics(keys, data, response_mask))
        except Exception as e:
            logger.exception("critic/advantages/zero_pass_mean metric failed", exc_info=e)
        return out

    def _advantage_zero_pass_metrics(self, keys: list[str], data: Any, response_mask: torch.Tensor) -> dict[str, float]:
        """Fraction of fully-zero-advantage GRPO groups whose reward is 1 (all-pass vs all-fail).

        Keys are ``{uid}_{session_id}_{index}`` and a group is a ``uid``; a session's reward comes
        from its final (highest-``index``) trajectory, the only row in the group-relative
        computation. Sequence reward is the *unmasked* ``rm_scores`` row sum -- the reward sits on
        the last response token, which the loss mask need not cover. Returns ``nan`` when the step
        has no zero-advantage group.
        """
        # uid -> (rows in the group, session_id -> (index, final row))
        groups: dict[str, tuple[list[int], dict[str, tuple[int, int]]]] = {}
        for row, key in enumerate(keys):
            fields = key.rsplit("_", 2)
            assert len(fields) == 3, f"Unexpected key format: {key}"
            uid, session_id, index = fields[0], fields[1], int(fields[2])
            rows, finals = groups.setdefault(uid, ([], {}))
            rows.append(row)
            if session_id not in finals or finals[session_id][0] < index:
                finals[session_id] = (index, row)

        row_nonzero = ((data["advantages"].abs() * response_mask).sum(dim=-1) > 0).tolist()
        sequence_score = data["rm_scores"].sum(dim=-1)

        zero_groups = 0
        pass_groups = 0
        for rows, finals in groups.values():
            if any(row_nonzero[row] for row in rows):
                continue
            zero_groups += 1
            final_rows = [row for _, row in finals.values()]
            reward = sequence_score[final_rows].mean().item()
            # Tolerance so a float-sum sequence score of 0.9999... still counts as a pass.
            if abs(reward - 1.0) < 1e-6:
                pass_groups += 1

        return {"critic/advantages/zero_pass_mean": (pass_groups / zero_groups if zero_groups else float("nan"))}
