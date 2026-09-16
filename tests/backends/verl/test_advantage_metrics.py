"""``critic/advantages/zero_*``: what the collapsed-group diagnostics count, and what they
deliberately ignore -- verl's padding rows, whose zero ``rm_scores`` would otherwise report
an all-pass group as an all-fail one.

The TransferQueue is faked (a real one needs a live cluster); the torch reductions are real.
"""

from unittest.mock import patch

import pytest
import torch

from agentcore_rl_toolkit.backends.verl.trainer_mixins import advantage_metrics as am
from agentcore_rl_toolkit.backends.verl.trainer_mixins.advantage_metrics import AdvantageZeroMetricsMixin


class _Batch:
    def __init__(self, keys, tags):
        self.keys = keys
        self.tags = tags
        self.partition_id = "train"


class _FakeTQ:
    """Returns one fixed row per fetched key, recording which keys were asked for."""

    def __init__(self, rows: dict[str, tuple[list[float], list[int], list[float]]]):
        self.rows = rows
        self.fetched: list[str] = []

    def kv_batch_get(self, *, keys, partition_id, select_fields):
        self.fetched = list(keys)
        advantages, response_mask, rm_scores = zip(*(self.rows[key] for key in keys), strict=True)
        return _Fetched(
            {
                "advantages": torch.tensor(advantages, dtype=torch.float32),
                "response_mask": torch.tensor(response_mask, dtype=torch.float32),
                "rm_scores": torch.tensor(rm_scores, dtype=torch.float32),
            }
        )


class _Fetched:
    def __init__(self, data):
        self._data = data

    def to_padded_tensor(self):
        return self._data


def _mixin() -> AdvantageZeroMetricsMixin:
    return type("_Trainer", (AdvantageZeroMetricsMixin,), {})()


def test_padding_rows_do_not_turn_an_all_pass_group_into_a_failure():
    """Both siblings passed, so the group collapsed for the good reason; verl's synthetic
    padding row carries a zero ``rm_scores`` that is not a policy outcome at all."""
    batch = _Batch(
        keys=["uidA_s0_0", "uidA_s1_0", "padhex_s2_0"],
        tags=[{}, {}, {"is_padding": True}],
    )
    tq = _FakeTQ(
        {
            "uidA_s0_0": ([0.0, 0.0], [1, 1], [0.0, 1.0]),
            "uidA_s1_0": ([0.0, 0.0], [1, 1], [0.0, 1.0]),
        }
    )

    with patch.object(am, "tq", tq):
        metrics = _mixin()._advantage_zero_metrics(batch)

    assert tq.fetched == ["uidA_s0_0", "uidA_s1_0"]
    assert metrics["critic/advantages/zero_mean"] == 1.0
    assert metrics["critic/advantages/zero_pass_mean"] == 1.0


def test_zero_mean_counts_valid_advantage_entries_only():
    batch = _Batch(keys=["uidA_s0_0", "uidB_s1_0"], tags=[{}, {}])
    tq = _FakeTQ(
        {
            # the masked-out entries are zero, but they are not part of the loss signal
            "uidA_s0_0": ([0.5, 0.0], [1, 0], [0.0, 1.0]),
            "uidB_s1_0": ([0.0, 0.0], [1, 1], [0.0, 0.0]),
        }
    )

    with patch.object(am, "tq", tq):
        metrics = _mixin()._advantage_zero_metrics(batch)

    # 2 of the 3 unmasked entries are zero (float32 reduction, hence approx)
    assert metrics["critic/advantages/zero_mean"] == pytest.approx(2 / 3)
    # one group collapsed, and it collapsed at reward 0
    assert metrics["critic/advantages/zero_pass_mean"] == 0.0


def test_a_batch_of_only_padding_reports_nothing():
    batch = _Batch(keys=["padhex_s0_0"], tags=[{"is_padding": True}])

    with patch.object(am, "tq", _FakeTQ({})):
        assert _mixin()._advantage_zero_metrics(batch) == {}
