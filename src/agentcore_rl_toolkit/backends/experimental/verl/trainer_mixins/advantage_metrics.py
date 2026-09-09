"""``critic/advantages/zero_*``: how much of a step's loss signal carries no gradient.

``zero_mean`` is the fraction of valid (response-masked) advantage entries that are
exactly zero -- a diagnostic for how much of a step collapses to no gradient (e.g.
GRPO groups whose rewards are all equal). ``zero_pass_mean`` says which way it
collapsed: of the GRPO groups whose advantages are entirely zero, the fraction
whose (necessarily shared) reward is 1, i.e. how much of the collapsed signal is
all-pass rather than all-fail.

Neither depends on the agent harness, so this is a separate mixin from
:mod:`.agent_loop_metrics` even though both hook ``_compute_metrics``.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import torch
import transfer_queue as tq

from .base import TrainerMixinBase

if TYPE_CHECKING:
    from transfer_queue import KVBatchMeta

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

        # Best-effort for the same reason as every other metric here: never abort
        # a training step on a diagnostic.
        try:
            metrics.update(self._advantage_zero_metrics(batch))
        except Exception as e:
            logger.exception("critic/advantages/zero_* metrics failed", exc_info=e)

    def _advantage_zero_metrics(self, batch: KVBatchMeta) -> dict[str, float]:
        """Fraction of valid (response-masked) advantage entries that are exactly zero.

        Mirrors stock verl's ``compute_data_metrics`` calculation
        (``(masked_select(advantages, response_mask) == 0).float().mean()``) but is
        layered on here so the vendored tree stays unpatched. ``advantages`` and
        ``response_mask`` are still live in the TransferQueue when
        ``_compute_metrics`` runs (``fit()`` clears the step's entries only after
        this call), so they can be re-fetched.

        Padding samples (verl's batch-divisibility pads) are dropped *before* the
        fetch, matching the base's ``metrics_batch = batch.select_idxs(non_padding_mask)``
        so the zero fraction is measured over real trajectories only. The
        just-fetched nested fields are densified with ``to_padded_tensor`` exactly as
        ``_compute_metrics`` does before calling ``compute_data_metrics``; the padding
        positions the pad introduces are then excluded by ``response_mask`` in the
        masked_select, so they never enter the count. The value is computed
        unconditionally (like verl), so an all-masked batch yields ``nan`` rather
        than a spurious 0.0.
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

        # Grouped view of the same fetch. Guarded separately so a key-format
        # surprise costs only this metric, not zero_mean above.
        try:
            out.update(self._advantage_zero_pass_metrics(keys, data, response_mask))
        except Exception as e:
            logger.exception("critic/advantages/zero_pass_mean metric failed", exc_info=e)
        return out

    def _advantage_zero_pass_metrics(self, keys: list[str], data: Any, response_mask: torch.Tensor) -> dict[str, float]:
        """Fraction of fully-zero-advantage GRPO groups whose reward is 1.

        ``zero_mean`` says how much of the step's loss signal collapsed; this says
        which way it collapsed. A GRPO group's advantages are all zero exactly when
        every member scored the same, so such a group has one shared reward -- 1 for
        an all-pass group (nothing left to learn), 0 for an all-fail one (the task is
        out of reach). Their ratio distinguishes those two regimes, which
        ``zero_mean`` alone conflates.

        Grouping mirrors ``compute_advantage_for_multi_trajectories``: TransferQueue
        keys are ``{uid}_{session_id}_{index}``, a group is a ``uid``, and the reward
        of a session is read off its final (highest-``index``) trajectory -- the only
        row that participates in the group-relative computation. Averaging the
        group's session rewards therefore weights sessions equally rather than by
        turn count; for a zero-advantage group every session reward is identical
        anyway, so the mean is that shared value.

        A row counts as zero-advantage when its advantages are zero at every valid
        (``response_mask``) position; ``to_padded_tensor`` pad positions are excluded
        by the same mask. Sequence reward is the unmasked ``rm_scores`` row sum, as in
        verl's ``compute_data_metrics`` -- the reward sits on the last response token,
        which the loss mask does not necessarily cover. Returns ``nan`` when the step
        has no zero-advantage group, so an empty numerator is not logged as 0.0.
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
            # Rewards are 0/1 pass rates, compared with a tolerance so a float-sum
            # sequence score of 0.9999... still counts as a pass.
            if abs(reward - 1.0) < 1e-6:
                pass_groups += 1

        return {"critic/advantages/zero_pass_mean": (pass_groups / zero_groups if zero_groups else float("nan"))}
