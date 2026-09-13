"""``RolloutFailureIsolationMixin``: a failed rollout's row gets a GRPO group of its own.

The mixin is exercised over a stand-in base that mimics the one behaviour of verl's
``_balance_batch`` this depends on -- appending padding rows whose tags are deep-copied from
the batch's first sample -- and a fake TransferQueue, since a real one needs a live cluster.
The ordering the tests pin (isolate *after* the base call) is what keeps the mixin's tag off
those padding rows.
"""

import copy
from unittest.mock import patch

import pytest
from transfer_queue import KVBatchMeta
from verl.trainer.ppo.v1 import PPOTrainerColocateAsync, PPOTrainerSeparateAsync, PPOTrainerSync

from agentcore_rl_toolkit.backends.verl import trainer as trainer_module
from agentcore_rl_toolkit.backends.verl.trainer_mixins import rollout_failure_isolation as rfi
from agentcore_rl_toolkit.backends.verl.trainer_mixins.rollout_failure_isolation import (
    FAILED_ROLLOUT_TAG,
    ROLLOUT_FAILED_FIELD,
    RolloutFailureIsolationMixin,
    is_failed_rollout_row,
)

FAILED = {ROLLOUT_FAILED_FIELD: 1.0, "trace_index": 0}
HEALTHY = {ROLLOUT_FAILED_FIELD: 0.0, "trace_index": 0}


class _Fetched:
    """What ``tq.kv_batch_get`` returns, narrowed to the ``pop`` the mixin calls."""

    def __init__(self, extra_fields: list[dict]):
        self._extra_fields = extra_fields

    def pop(self, field: str):
        assert field == "extra_fields"
        return _NonTensorStack(self._extra_fields)


class _NonTensorStack:
    def __init__(self, values: list[dict]):
        self._values = values

    def tolist(self) -> list[dict]:
        return self._values


class _FakeTQ:
    """The TransferQueue module functions the mixin uses, recording every call."""

    def __init__(self, extra_fields: dict[str, dict]):
        self.extra_fields = extra_fields
        self.gets: list[dict] = []
        self.puts: list[dict] = []
        self.events: list[str] = []

    def kv_batch_get(self, *, keys, partition_id, select_fields):
        self.gets.append({"keys": list(keys), "partition_id": partition_id, "select_fields": list(select_fields)})
        self.events.append("get")
        return _Fetched([self.extra_fields[key] for key in keys])

    def kv_batch_put(self, *, keys, partition_id, fields):
        self.puts.append({"keys": list(keys), "partition_id": partition_id, "fields": fields})
        self.events.append("put")

    def written_uids(self) -> list[str]:
        # indexing a tensordict non-tensor field yields its own list subclass, hence list()
        return [uid for put in self.puts for uid in list(put["fields"]["uid"])]


class _BaseTrainer:
    """Stands in for ``PPOTrainer._balance_batch``, which is where padding rows appear.

    verl builds each padding row by deep-copying the batch's first sample's tags
    (``construct_minimal_padding_template``), which is why the mixin must stamp its own tag
    only after this returns.
    """

    def __init__(self, events: list[str], padding_rows: int = 0):
        self.events = events
        self.padding_rows = padding_rows

    def _balance_batch(self, batch, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        self.events.append("balance")
        metrics[f"{logging_prefix}/minmax_diff"] = 0
        for i in range(self.padding_rows):
            batch.keys.append(f"pad{i}_padsession_0")
            batch.tags.append({**copy.deepcopy(batch.tags[0]), "is_padding": True})
        return batch


class _Trainer(RolloutFailureIsolationMixin, _BaseTrainer):
    pass


def _batch(*keys: str, padding: tuple[str, ...] = ()) -> KVBatchMeta:
    tags = [{"is_padding": key in padding} for key in keys]
    return KVBatchMeta(keys=list(keys), tags=tags, partition_id="train")


def _run(batch, extra_fields: dict[str, dict], *, padding_rows: int = 0):
    """Drive ``_balance_batch`` over a fake TransferQueue; return (batch, metrics, tq)."""
    events: list[str] = []
    trainer = _Trainer(events, padding_rows=padding_rows)
    tq = _FakeTQ(extra_fields)
    tq.events = events
    metrics: dict = {}
    with patch.object(rfi, "tq", tq):
        result = trainer._balance_batch(batch, metrics)
    return result, metrics, tq


def test_failed_rows_are_moved_into_their_own_groups():
    batch = _batch("uidA_s0_0", "uidA_s1_0", "uidA_s2_0")

    result, metrics, tq = _run(batch, {"uidA_s0_0": HEALTHY, "uidA_s1_0": FAILED, "uidA_s2_0": FAILED})

    # only the failed rows are rewritten, one fresh uid each
    assert tq.puts[0]["keys"] == ["uidA_s1_0", "uidA_s2_0"]
    assert tq.puts[0]["partition_id"] == "train"
    uids = tq.written_uids()
    assert all(uid.startswith("fail") for uid in uids)
    assert len(set(uids)) == 2
    # ... and tagged, so the metrics mixins can tell them from real trajectories
    assert [tag.get(FAILED_ROLLOUT_TAG, False) for tag in result.tags] == [False, True, True]
    assert metrics["training/rollout_failure/total_failed_rows"] == 2
    # the base trainer's own metrics survive
    assert "global_seqlen/minmax_diff" in metrics


def test_a_healthy_batch_writes_nothing():
    batch = _batch("uidA_s0_0", "uidA_s1_0")

    result, metrics, tq = _run(batch, {"uidA_s0_0": HEALTHY, "uidA_s1_0": HEALTHY})

    assert tq.puts == []
    assert metrics["training/rollout_failure/total_failed_rows"] == 0
    assert not any(FAILED_ROLLOUT_TAG in tag for tag in result.tags)


def test_isolation_runs_after_the_base_and_skips_its_padding_rows():
    """verl's padding rows already have their own ``pad<hex>`` uid, and their tags are
    deep-copied from row 0 -- so tagging before the base call would mark them failed."""
    batch = _batch("uidA_s0_0", "uidA_s1_0")

    result, metrics, tq = _run(batch, {"uidA_s0_0": FAILED, "uidA_s1_0": HEALTHY}, padding_rows=2)

    assert tq.events == ["balance", "get", "put"]
    assert tq.gets[0]["keys"] == ["uidA_s0_0", "uidA_s1_0"]  # padding never fetched
    assert tq.puts[0]["keys"] == ["uidA_s0_0"]
    assert metrics["training/rollout_failure/total_failed_rows"] == 1
    padding_tags = [tag for tag in result.tags if tag["is_padding"]]
    assert len(padding_tags) == 2
    assert not any(FAILED_ROLLOUT_TAG in tag for tag in padding_tags)


def test_a_batch_of_only_padding_is_left_alone():
    batch = _batch("pad0_p_0", padding=("pad0_p_0",))

    _, metrics, tq = _run(batch, {})

    assert tq.gets == [] and tq.puts == []
    assert metrics["training/rollout_failure/total_failed_rows"] == 0


def test_the_failure_flag_survives_a_missing_or_odd_extra_fields():
    """The flag rides in a loosely-typed dict, so only a truthy scalar counts as a failure."""
    assert is_failed_rollout_row({ROLLOUT_FAILED_FIELD: 1.0})
    assert is_failed_rollout_row({ROLLOUT_FAILED_FIELD: True})
    assert not is_failed_rollout_row({ROLLOUT_FAILED_FIELD: 0.0})
    assert not is_failed_rollout_row({})
    assert not is_failed_rollout_row({ROLLOUT_FAILED_FIELD: None})
    assert not is_failed_rollout_row({ROLLOUT_FAILED_FIELD: "1.0"})
    assert not is_failed_rollout_row(None)


@pytest.mark.parametrize(
    ("trainer_cls", "verl_cls"),
    [
        (trainer_module.AgentCorePPOTrainerSync, PPOTrainerSync),
        (trainer_module.AgentCorePPOTrainerColocateAsync, PPOTrainerColocateAsync),
        (trainer_module.AgentCorePPOTrainerSeparateAsync, PPOTrainerSeparateAsync),
    ],
)
def test_every_mode_isolates_failed_rollouts(trainer_cls, verl_cls):
    """The whole point of the mixin is that sync and async agree on a failed rollout."""
    mro = trainer_cls.__mro__
    # ahead of the verl trainer, so its ``_balance_batch`` is the one being wrapped
    assert mro.index(RolloutFailureIsolationMixin) < mro.index(verl_cls)
