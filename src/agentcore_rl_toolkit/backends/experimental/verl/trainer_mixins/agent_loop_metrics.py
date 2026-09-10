"""``agent_loop/*``: the per-session metrics an agent harness leaves in ``extra_fields``.

Fetches ``extra_fields`` for the step's batch and reduces every entry of its ``metrics`` dict
to ``agent_loop/<name>/{mean,min,max,sum}``. Metric names are not hardcoded, so any
agent-harness metric flows through unchanged.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import transfer_queue as tq
from transfer_queue import KVBatchMeta

from .base import TrainerMixinBase

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentLoopMetricsMixin(TrainerMixinBase):
    """Adds ``agent_loop/*`` to a PPOTrainer, reduced per session from the batch."""

    def _compute_metrics(
        self,
        batch: KVBatchMeta,
        metrics: dict[str, Any],
        timing_raw: dict[str, Any],
        global_steps: int,
        epoch: int,
    ) -> None:
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)

        # Best-effort: a failure here must never abort a training step.
        try:
            metrics.update(self._agent_loop_extra_field_metrics(batch))
        except Exception as e:
            logger.exception("agent_loop extra-field metrics failed", exc_info=e)

    def _agent_loop_extra_field_metrics(self, batch: KVBatchMeta) -> dict[str, float]:
        """Fetch ``extra_fields`` for this step's samples and reduce every metric.

        The step's TransferQueue entries are still live here (``fit()`` clears them after this
        call). Padding keys are dropped *before* the fetch, so nothing depends on synthetic
        padding rows having ``extra_fields`` ready in the queue.
        """
        non_padding = [not tag.get("is_padding", False) for tag in batch.tags]
        keys = [k for k, keep in zip(batch.keys, non_padding, strict=True) if keep]
        if not keys:
            return {}

        # .pop() materializes the NonTensorStack exposing .tolist(); indexing returns
        # tensordict's internal plain-list subclass with no .tolist().
        assert batch.partition_id is not None
        fetched = (
            tq.kv_batch_get(
                keys=keys,
                partition_id=batch.partition_id,
                select_fields=["extra_fields"],
            )
            .pop("extra_fields")
            .tolist()
        )

        efs = [(ef if isinstance(ef, dict) else {}) for ef in fetched]
        if not efs:
            return {}

        # Every trajectory of a session carries the same session-level metrics dict, so collapse
        # to one representative per session; otherwise sessions are weighted by trajectory count.
        by_session: dict[Any, dict] = {}
        for i, ef in enumerate(efs):
            sid = ef.get("request_id", None)
            by_session.setdefault(sid if sid is not None else f"_nosession_{i}", ef)
        session_efs = list(by_session.values())

        # Each metric is reduced only over the sessions that reported it (absent -> excluded,
        # never a 0.0 skew).
        out: dict[str, float] = {}
        per_metric: dict[str, list[float]] = {}
        for ef in session_efs:
            metrics = ef.get("metrics")
            if not isinstance(metrics, dict):
                continue
            for name, value in metrics.items():
                # The dict may also carry non-numeric session meta (datetimes, ids); skip those
                # rather than crashing on float(datetime).
                if not isinstance(value, (int, float)):
                    continue
                per_metric.setdefault(name, []).append(float(value))

        for name, values in per_metric.items():
            vals = np.array(values, dtype=float)
            out[f"agent_loop/{name}/mean"] = float(vals.mean())
            out[f"agent_loop/{name}/min"] = float(vals.min())
            out[f"agent_loop/{name}/max"] = float(vals.max())
            out[f"agent_loop/{name}/sum"] = float(vals.sum())
        return out
