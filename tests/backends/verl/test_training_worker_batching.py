from contextlib import nullcontext

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.workers.engine_workers import TrainingWorker


class _Engine:
    def get_data_parallel_size(self):
        return 1

    def get_data_parallel_rank(self):
        return 0

    def train_mode(self, **kwargs):
        return nullcontext()

    def is_mp_src_rank_with_outputs(self):
        return False


class _Profiler:
    def step(self):
        pass


class _CountingWorker:
    """Stops at the boundary where TrainingWorker would run an optimizer step."""

    def __init__(self):
        self.engine = _Engine()
        self.profiler = _Profiler()
        self.flops_counter = None
        self.train_batch_calls = 0

    def train_batch(self, data):
        self.train_batch_calls += 1


def _run_worker(*, rows, **metadata):
    data = TensorDict({"dummy": torch.arange(rows)}, batch_size=[rows])
    tu.assign_non_tensor(data, **metadata)
    worker = _CountingWorker()

    TrainingWorker.train_mini_batch(worker, data)

    return worker.train_batch_calls


@pytest.mark.parametrize(
    ("rows", "num_mini_batch", "expected_calls"),
    [(256, 1, 1), (768, 1, 1), (300, 3, 3), (900, 3, 3)],
)
def test_num_mini_batch_keeps_optimizer_steps_stable(rows, num_mini_batch, expected_calls):
    assert _run_worker(rows=rows, num_mini_batch=num_mini_batch) == expected_calls


def test_fixed_mini_batch_size_scales_optimizer_steps_with_expanded_rows():
    assert _run_worker(rows=256, mini_batch_size=256) == 1
    assert _run_worker(rows=768, mini_batch_size=256) == 3
