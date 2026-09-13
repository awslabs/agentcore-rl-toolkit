"""Keeps a failed rollout out of its own GRPO group, in every trainer mode.

A rollout that fails (container never came up, agent handler raised, run timed out) must
not decide the fate of the prompt it was sampled for. verl offers an agent loop two ways
to report that, and both are wrong at the batch level:

* **Raise.** The prompt group is marked ``failure``. The *sync* trainer tolerates that and
  trains the surviving trajectories, but the *async* trainers evict and refill the whole
  group -- so one bad rollout throws away its healthy siblings, repeatedly, and a prompt
  whose container reliably fails can stall a step.
* **Return a zero-reward row.** No group is failed anywhere, but the synthetic zero now
  sits *inside* the group's reward pool, so GRPO shifts and rescales every sibling's
  advantage by an outcome the policy never produced.

So the loop emits an inert row (one masked token, zero reward, see
``RolloutSessionAgentLoop.make_failed_loop_output``) and flags it in ``extra_fields``, and
this mixin moves that row into a group of its own before advantages are computed. Its own
advantage is zero either way -- its ``response_mask`` is all zeros -- and its siblings are
scored as if it had never been sampled, which is exactly how the sync trainer behaves when
a loop raises. Both trainer families then agree, without either giving up rows.

``uid`` is the field GRPO groups by (``_compute_advantage`` fetches it from TransferQueue),
and verl already rewrites it for its own padding rows (``pad<hex>`` in
``verl/trainer/ppo/padding_utils.py``), so this borrows the same trick with a ``fail<hex>``
prefix. It has to happen trainer-side: an agent loop cannot set ``uid`` itself, since the
worker overwrites the field from the prompt's row when it stores the output.

See ``docs/verl_agent_loop_merge.md`` §4 for the arithmetic and the residual edges.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

import transfer_queue as tq
from transfer_queue import KVBatchMeta
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

from .base import TrainerMixinBase

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

#: ``extra_fields`` entry an agent loop sets to 1.0 on a row that stands in for a failed
#: rollout. A float rather than a bool so it also reduces to a failure *rate* through the
#: ``agent_loop/*`` metrics path.
ROLLOUT_FAILED_FIELD = "rollout_failed"

#: Batch tag this mixin stamps on the rows it isolated, so the metrics mixins can tell a
#: failure stand-in apart from a real trajectory (verl's own ``is_padding`` plays the same
#: role for its padding rows).
FAILED_ROLLOUT_TAG = "is_failed_rollout"


def is_failed_rollout_row(extra_fields: Any) -> bool:
    """Whether a row's ``extra_fields`` marks it as a failed rollout's stand-in."""
    if not isinstance(extra_fields, dict):
        return False
    flag = extra_fields.get(ROLLOUT_FAILED_FIELD, 0.0)
    return bool(flag) if isinstance(flag, bool | int | float) else False


class RolloutFailureIsolationMixin(TrainerMixinBase):
    """Gives every failed-rollout row its own ``uid``, so it cannot skew a GRPO group."""

    def _balance_batch(self, batch: KVBatchMeta, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        # After the base implementation, which is where verl appends its own padding rows:
        # those are already uid-isolated, and skipping them keeps this mixin's tag off the
        # rows padding deep-copies from the batch's first sample.
        batch = super()._balance_batch(batch, metrics, logging_prefix=logging_prefix, keep_minibatch=keep_minibatch)
        metrics["training/rollout_failure/total_failed_rows"] = self._isolate_failed_rollouts(batch)
        return batch

    def _isolate_failed_rollouts(self, batch: KVBatchMeta) -> int:
        """Rewrite the ``uid`` of every failed-rollout row; return how many there were.

        Deliberately not best-effort, unlike the metrics mixins: skipping this quietly
        would train on advantages computed against rewards no policy produced, which is a
        silently wrong update rather than a missing chart.
        """
        rows = [
            (i, key)
            for i, (key, tag) in enumerate(zip(batch.keys, batch.tags, strict=True))
            if not tag.get("is_padding", False)
        ]
        if not rows:
            return 0

        assert batch.partition_id is not None
        # .pop() materializes the NonTensorStack exposing .tolist(); indexing returns
        # tensordict's internal plain-list subclass with no .tolist().
        fetched = (
            tq.kv_batch_get(
                keys=[key for _, key in rows],
                partition_id=batch.partition_id,
                select_fields=["extra_fields"],
            )
            .pop("extra_fields")
            .tolist()
        )

        failed = [row for row, ef in zip(rows, fetched, strict=True) if is_failed_rollout_row(ef)]
        if not failed:
            return 0

        # One fresh uid per row: a failed rollout emits exactly one row, and a group of one
        # gets a zero advantage from verl (singleton groups are mean 0 / std 1 by
        # construction in compute_grpo_outcome_advantage).
        uids = [f"fail{uuid.uuid4().hex}" for _ in failed]
        tq.kv_batch_put(
            keys=[key for _, key in failed],
            partition_id=batch.partition_id,
            fields=list_of_dict_to_tensordict([{"uid": uid} for uid in uids]),
        )
        for i, _ in failed:
            batch.tags[i][FAILED_ROLLOUT_TAG] = True

        logger.warning(
            "Isolated %d failed rollout(s) out of %d real rows into their own GRPO groups: %s",
            len(failed),
            len(rows),
            [key for _, key in failed],
        )
        return len(failed)


__all__ = [
    "FAILED_ROLLOUT_TAG",
    "ROLLOUT_FAILED_FIELD",
    "RolloutFailureIsolationMixin",
    "is_failed_rollout_row",
]
