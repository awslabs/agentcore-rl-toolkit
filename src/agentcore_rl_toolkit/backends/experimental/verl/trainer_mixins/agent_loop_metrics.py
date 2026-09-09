"""``agent_loop/*``: the per-session metrics an agent harness leaves in ``extra_fields``.

The v1 TransferQueue trainer's ``_compute_metrics`` only fetches a fixed field
list from the TransferQueue, so the ``extra_fields["metrics"]`` dict populated per
trajectory by ``RolloutSessionAgentLoop`` never reaches the tracker. This mixin fetches
``extra_fields`` for the step's batch and reduces every key/value pair in the
generic ``metrics`` dict to ``agent_loop/<name>/{mean,min,max,sum}`` -- the metric
names are not hardcoded, so any agent-harness metric flows through unchanged.
Because a session emits many trajectories that all share the same session-level
``metrics`` dict (1:many session:trajectory), the reduction first collapses to one
representative per session (keyed by ``request_id``) so sessions are not weighted
by their trajectory count.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import transfer_queue as tq

from .base import TrainerMixinBase

if TYPE_CHECKING:
    from transfer_queue import KVBatchMeta

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
        # Standard verl metrics first (compute_data_metrics, timing, throughput),
        # plus whatever other mixins sit further down the MRO.
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)

        # Best-effort: a failure here must never abort a training step.
        try:
            metrics.update(self._agent_loop_extra_field_metrics(batch))
        except Exception as e:
            # logger.exception so the traceback reaches the driver log -- a bare
            # warning here previously hid the cause of agent_loop/* going missing.
            logger.exception("agent_loop extra-field metrics failed", exc_info=e)

    def _agent_loop_extra_field_metrics(self, batch: KVBatchMeta) -> dict[str, float]:
        """Fetch ``extra_fields`` for this step's samples and reduce every metric.

        ``batch`` is the ``KVBatchMeta`` handed to ``_compute_metrics`` by
        ``fit()``; its TransferQueue entries are still live (``fit()`` clears
        them only after this call). Padding samples (``tags[i].is_padding``) are
        dropped before the fetch, the batch's trajectories are collapsed to one
        representative per session (``request_id``) so per-session metrics are not
        weighted by trajectory count. Every value is then reduced generically: each
        metric is averaged over the sessions that reported it, with no field
        special-cased -- abort/timeout arrive as ordinary 0.0/1.0 metrics from the
        loop, so their per-session rates fall out as ``agent_loop/aborted/mean`` etc.

        Padding keys are dropped *before* the fetch: this mixin is the only code
        that reads ``extra_fields`` on the post-balance batch (base's reward read
        runs pre-padding, and the spec-decode read is mtp-gated), so fetching the
        synthetic padding keys here would make us the first -- and only --
        consumer to depend on their ``extra_fields`` being ready in the queue.
        """
        non_padding = [not tag.get("is_padding", False) for tag in batch.tags]
        keys = [k for k, keep in zip(batch.keys, non_padding, strict=True) if keep]
        if not keys:
            return {}

        # Use .pop("extra_fields") (not td["extra_fields"]): indexing a single
        # non-tensor field returns tensordict's internal LinkedList (a plain list
        # subclass, no .tolist()), whereas .pop() materializes the NonTensorStack
        # that exposes .tolist() -> list[dict]. This mirrors verl's own reads in
        # trainer_base.py (reward + spec-decode paths).
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

        # 1:many session:trajectory. A session emits one trajectory per token-bearing
        # turn (or a single aborted trajectory on failure), and every one of those
        # trajectories carries the *same* session-level extra_fields["metrics"] dict
        # (llm_generated_length is folded into a session total upstream in
        # RolloutSessionAgentLoop). Reducing over trajectories would therefore weight each
        # session by its turn count, so collapse to one representative extra_fields
        # per session (request_id) before reducing. The aborted flag is uniform
        # within a session, so first-occurrence is a faithful representative.
        by_session: dict[Any, dict] = {}
        for i, ef in enumerate(efs):
            sid = ef.get("request_id", None)
            by_session.setdefault(sid if sid is not None else f"_nosession_{i}", ef)
        session_efs = list(by_session.values())

        # Reduce every metric the harness emitted -- the key set is not hardcoded
        # here, so any new agent-harness metric surfaces as agent_loop/<name>/*
        # without touching the trainer. Each metric is averaged only over the
        # sessions that actually reported it (absent -> excluded, never a 0.0 skew).
        # Abort/timeout need no special-casing: the loop emits them as 0.0/1.0
        # metrics on every session (see RolloutSessionAgentLoop.make_*_loop_output), so
        # their per-session rates fall out as agent_loop/aborted/mean etc.
        out: dict[str, float] = {}
        per_metric: dict[str, list[float]] = {}
        for ef in session_efs:
            metrics = ef.get("metrics")
            if not isinstance(metrics, dict):
                continue
            for name, value in metrics.items():
                # The metrics dict may be a copy of the whole session meta, which
                # also carries non-numeric fields (datetimes, ids, ...). The
                # reduction is float-only, so aggregate int/float values (bool
                # included, as 0.0/1.0 for abort/timeout rates) and skip the rest
                # rather than crashing on e.g. float(datetime) / float("verl_...").
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
