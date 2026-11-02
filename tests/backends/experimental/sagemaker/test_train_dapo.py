"""``train_dapo`` tests: dynamic sampling on its own, and the full ``train()`` loop against fakes.

No AWS calls: the SageMaker service client, the gateway, the ACR client and the rollouts are
replaced with fakes. Dynamic sampling, advantages, datum building, the update order, evals and
teardown are the real code.
"""

import asyncio
import logging
import re
from types import SimpleNamespace

import pytest

pytest.importorskip("sagemaker.train.training_session", reason="needs sagemaker-train>=1.27 (SageMaker extras)")

import agentcore_rl_toolkit.backends.experimental.sagemaker.train_dapo as td  # noqa: E402
from agentcore_rl_toolkit.rollout_gateway import TraceRecord  # noqa: E402

MIXED = [1.0, 0.0]  # informative: rewards differ
SAME = [1.0, 1.0]  # filtered: all rewards equal


def record(loss_mask: tuple[int, ...] = (1, 1)) -> TraceRecord:
    n = len(loss_mask)
    return TraceRecord(token_ids=[1, 2, *range(10, 10 + n)], loss_mask=list(loss_mask), logprobs=[-0.5] * n)


def prompts(*rewards: list[float], untrainable: bool = False) -> list[dict]:
    return [{"id": i, "rewards": r, "untrainable": untrainable} for i, r in enumerate(rewards)]


# ---------------------------------------------------------------------------
# is_informative / sample_informative_groups
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("rewards", "expected"),
    [
        ([1.0, 0.0], True),
        ([0.0, 0.0], False),
        ([1.0, 1.0], False),
        ([0.5, 0.5, 0.6], True),
        ([1.0], False),
        ([], False),
    ],
)
def test_is_informative(rewards, expected):
    assert td.is_informative(rewards) is expected


class FakeGroups:
    """Stands in for collect_group: each prompt says its K rewards; one trainable record per rollout."""

    def __init__(self):
        self.seen: list[int] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def __call__(self, payload: dict) -> td.Group:
        self.seen.append(payload["id"])
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0)
        self.in_flight -= 1
        return [([record()], r) for r in payload["rewards"]]


def sample(queue: list[dict], batch_size: int = 2, max_num_gen_batches: int = 10, fake: FakeGroups | None = None):
    fake = fake or FakeGroups()
    result = asyncio.run(td.sample_informative_groups(queue, batch_size, max_num_gen_batches, fake))
    return result, fake


def test_sample_one_generation_enough():
    queue = prompts(MIXED, MIXED, MIXED)
    (groups, rewards, stats), fake = sample(queue)
    assert groups is not None and len(groups) == 2
    assert fake.seen == [0, 1]
    assert len(queue) == 1  # popped in place
    assert rewards == [1.0, 0.0, 1.0, 0.0]
    assert stats == {
        "num_gen_batches": 1,
        "num_groups_seen": 2,
        "num_groups_filtered": 0,
        "num_kept": 2,
        "num_empty_rollouts": 0,
    }


def test_sample_resamples_until_full():
    queue = prompts(SAME, MIXED, SAME, SAME, MIXED, SAME)
    (groups, rewards, stats), fake = sample(queue)
    assert groups is not None and len(groups) == 2
    assert fake.seen == [0, 1, 2, 3, 4, 5]
    assert len(rewards) == 12  # every rollout of every generation, before filtering
    assert (stats["num_gen_batches"], stats["num_groups_filtered"], stats["num_kept"]) == (3, 4, 2)


def test_sample_truncates_extra_groups():
    (groups, _, stats), _ = sample(prompts(SAME, MIXED, MIXED, MIXED))
    assert groups is not None and len(groups) == 2
    assert stats["num_kept"] == 3  # kept before truncation is reported


def test_sample_returns_none_when_epoch_runs_out():
    queue = prompts(SAME, SAME, MIXED)
    (groups, rewards, stats), fake = sample(queue)
    assert groups is None
    assert rewards == [1.0, 1.0, 1.0, 1.0]  # still reported for logging
    assert len(queue) == 1  # the leftover prompt is not rolled out
    assert stats["num_gen_batches"] == 1


def test_sample_raises_at_cap():
    fake = FakeGroups()
    with pytest.raises(td.InsufficientGroupsError, match="after 2 generation batches"):
        sample(prompts(*[SAME] * 10), max_num_gen_batches=2, fake=fake)
    assert len(fake.seen) == 4  # exactly 2 generations, no more


def test_sample_cap_checked_before_queue():
    # the cap is reached exactly when the queue runs out: raise, do not quietly end the epoch
    with pytest.raises(td.InsufficientGroupsError):
        sample(prompts(SAME, SAME, SAME, SAME), max_num_gen_batches=2)


def test_sample_cap_zero_is_unlimited():
    (groups, _, stats), _ = sample(prompts(*[SAME] * 8, MIXED, MIXED), max_num_gen_batches=0)
    assert groups is not None
    assert stats["num_gen_batches"] == 5


def test_sample_counts_empty_rollouts():
    async def one_failed(payload):
        return [([record()], 1.0), ([], 0.0)]  # rollout 1 failed: no records, reward 0

    queue = prompts(MIXED, MIXED)
    groups, _, stats = asyncio.run(td.sample_informative_groups(queue, 2, 10, one_failed))
    assert groups is not None
    assert stats["num_empty_rollouts"] == 2


def test_sample_one_generation_in_flight_at_a_time():
    fake = FakeGroups()
    sample(prompts(SAME, SAME, MIXED, MIXED), fake=fake)
    assert fake.max_in_flight <= 2  # never more than batch_size groups at once


# ---------------------------------------------------------------------------
# train(): fakes for everything that would talk to AWS
# ---------------------------------------------------------------------------
class Done:
    def __init__(self, value=None):
        self.value = value

    def result(self, timeout=None):
        return self.value


class FakeTrainingClient:
    training_session_name = "fake-session"

    def __init__(self):
        self.calls: list[str] = []
        self.fwd_bwd: list[dict] = []
        self.datums: list[list[dict]] = []

    def get_tokenizer(self):
        return object()

    def create_sampling_client(self):
        return object()

    def forward_backward(self, datums, loss_fn, loss_fn_config=None):
        self.calls.append("forward_backward")
        self.fwd_bwd.append({"n": len(datums), "loss_fn": loss_fn, "loss_fn_config": loss_fn_config})
        self.datums.append(datums)
        return Done(SimpleNamespace(metrics={"loss": 0.1}))

    def optim_step(self, params):
        self.calls.append("optim_step")
        return Done()

    def save_state(self):
        self.calls.append("save_state")
        return Done(SimpleNamespace(model_package_arn="ckpt"))

    def save_weights_and_get_sampling_client(self):
        self.calls.append("rebind")
        return object()

    def save_weights_for_sampler(self, description):
        self.calls.append("save_weights_for_sampler")
        return Done(SimpleNamespace(model_package_arn="weights"))

    def stop(self):
        self.calls.append("stop")


class FakeServer:
    def __init__(self, gw, host, port):
        self.base_url = "http://fake"
        self.closed = False

    def start(self):
        pass

    def shutdown(self):
        self.closed = True


class FakeHttpClient:
    def __init__(self, *args, **kwargs):
        self.closed = False

    async def aclose(self):
        self.closed = True


@pytest.fixture
def harness(monkeypatch):
    tc = FakeTrainingClient()
    state = SimpleNamespace(tc=tc, train=[], eval=[], evals=[], created=0, server=None, client=None, fail_on_call=None)
    calls = {"n": 0}

    def create_lora_training_client(**kwargs):
        state.created += 1
        return tc

    def server(*args, **kwargs):
        state.server = FakeServer(*args, **kwargs)
        return state.server

    def http_client(*args, **kwargs):
        state.client = FakeHttpClient()
        return state.client

    async def run_one_rollout(*, payload, sampling_defaults, **kwargs):
        if "eval" in payload:
            state.evals.append(sampling_defaults["temperature"])
            return [], payload["eval"], {}
        if state.fail_on_call is not None and calls["n"] == state.fail_on_call:
            raise state.fail_on_call_error
        k = payload.setdefault("_k", 0)
        payload["_k"] = (k + 1) % len(payload["rewards"])
        calls["n"] += 1
        mask = (0, 0) if payload["untrainable"] else (1, 1)
        return [record(mask)], payload["rewards"][k], {}

    def load_dataset(path):
        return [dict(p) for p in (state.eval if path == "eval.jsonl" else state.train)]

    monkeypatch.setattr(
        td, "ServiceClient", lambda **kw: SimpleNamespace(create_lora_training_client=create_lora_training_client)
    )
    monkeypatch.setattr(td, "SageMakerSdkBackend", lambda sc: SimpleNamespace(set_sampling_client=lambda new: None))
    monkeypatch.setattr(td, "RolloutGateway", lambda **kw: object())
    monkeypatch.setattr(td, "HfTemplateRenderer", lambda *a, **kw: object())
    monkeypatch.setattr(td, "ThreadedGatewayServer", server)
    monkeypatch.setattr(td, "AgentCoreHttpClient", http_client)
    monkeypatch.setattr(td, "LocalRateLimiter", lambda *a: object())
    monkeypatch.setattr(td, "local_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(td, "run_one_rollout", run_one_rollout)
    monkeypatch.setattr(td, "load_dataset", load_dataset)
    return state


def make_config(**overrides) -> td.DapoConfig:
    base = {"dataset_path": "train.jsonl", "batch_size": 2, "responses_per_prompt": 2}
    return td.DapoConfig(**{**base, **overrides})


def run(config: td.DapoConfig) -> None:
    asyncio.run(td.train(config))


def test_dapo_config_defaults_and_inherits_config():
    config = td.DapoConfig()
    assert isinstance(config, td.Config)
    assert (config.clip_low_threshold, config.clip_high_threshold, config.max_num_gen_batches) == (0.8, 1.28, 10)


def test_train_happy_path(harness):
    # step 0: [MIXED, SAME] keeps 1, resample [MIXED, MIXED] -> full; step 1: [MIXED, SAME] runs out
    harness.train = prompts(MIXED, SAME, MIXED, MIXED, MIXED, SAME)
    run(make_config())
    tc = harness.tc
    assert tc.calls.count("forward_backward") == 1
    assert tc.fwd_bwd[0] == {
        "n": 4,  # 2 groups x K=2, one record each
        "loss_fn": "ppo",
        "loss_fn_config": {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28},
    }
    assert tc.calls[-3:] == ["rebind", "save_state", "stop"]
    assert harness.server.closed and harness.client.closed


def test_train_passes_configured_clip_bounds(harness):
    harness.train = prompts(MIXED, MIXED)
    run(make_config(clip_low_threshold=0.7, clip_high_threshold=1.4, loss="cispo"))
    assert harness.tc.fwd_bwd[0]["loss_fn"] == "cispo"
    assert harness.tc.fwd_bwd[0]["loss_fn_config"] == {"clip_low_threshold": 0.7, "clip_high_threshold": 1.4}


def test_train_update_order_per_step(harness):
    harness.train = prompts(MIXED, MIXED, MIXED, MIXED)
    run(make_config())
    assert harness.tc.calls == ["forward_backward", "optim_step", "rebind"] * 2 + ["save_state", "stop"]


def test_train_advantages_follow_rewards(harness):
    harness.train = prompts([1.0, 0.0], [0.0, 1.0])
    run(make_config())
    per_datum = [
        {
            round(a, 6)
            for a, w in zip(d["lossFnInputs"]["advantages"]["data"], d["lossFnInputs"]["weights"]["data"], strict=True)
            if w
        }
        for d in harness.tc.datums[0]
    ]
    # GRPO-normalized within each group: the 1.0 rollout gets +1, the 0.0 rollout gets -1
    assert per_datum == [{1.0}, {-1.0}, {-1.0}, {1.0}]


def test_train_dataset_smaller_than_batch_never_creates_session(harness):
    harness.train = prompts(MIXED)
    with pytest.raises(ValueError, match="fewer than batch_size"):
        run(make_config())
    assert harness.created == 0  # nothing billed


def test_train_insufficient_groups_saves_then_stops(harness):
    harness.train = prompts(*[SAME] * 20)
    with pytest.raises(td.InsufficientGroupsError):
        run(make_config(max_num_gen_batches=2))
    assert harness.tc.calls == ["save_state", "stop"]  # abort checkpoint, then stop billing
    assert harness.server.closed and harness.client.closed


def test_train_other_errors_do_not_checkpoint(harness):
    harness.train = prompts(MIXED, MIXED, MIXED, MIXED)
    harness.fail_on_call = 4  # first rollout of step 1
    harness.fail_on_call_error = RuntimeError("rollout infrastructure broke")
    with pytest.raises(RuntimeError, match="infrastructure"):
        run(make_config())
    assert "save_state" not in harness.tc.calls
    assert harness.tc.calls[-1] == "stop"


def test_train_cancel_still_stops_session(harness):
    harness.train = prompts(MIXED, MIXED)
    harness.fail_on_call = 0
    harness.fail_on_call_error = asyncio.CancelledError()  # what the SIGTERM handler triggers
    with pytest.raises(asyncio.CancelledError):
        run(make_config())
    assert harness.tc.calls == ["stop"]


def test_train_skips_update_when_no_datums(harness):
    harness.train = prompts(MIXED, MIXED, untrainable=True)
    run(make_config())
    assert harness.tc.calls == ["save_state", "stop"]


# ---------------------------------------------------------------------------
# eval schedule
# ---------------------------------------------------------------------------
def eval_steps(caplog) -> list[int]:
    return [int(m) for m in re.findall(r"step=(\d+)\)", caplog.text)]


def test_eval_schedule_and_final_eval(harness, caplog):
    caplog.set_level(logging.INFO, logger=td.logger.name)
    harness.train = prompts(*[MIXED] * 6)
    harness.eval = [{"eval": 1.0}]
    run(make_config(eval_every=2, eval_dataset_path="eval.jsonl"))
    assert eval_steps(caplog) == [2, 3]  # every 2 steps, then the final step


def test_eval_uses_eval_temperature(harness):
    harness.train = prompts(MIXED, MIXED)
    harness.eval = [{"eval": 1.0}]
    run(make_config(eval_dataset_path="eval.jsonl", temperature=1.0, eval_temperature=0.0))
    assert harness.evals == [0.0]


def test_final_eval_runs_when_no_step_ran(harness, caplog):
    caplog.set_level(logging.INFO, logger=td.logger.name)
    harness.train = prompts(SAME, SAME, MIXED)  # step 0 filters everything, then runs out
    harness.eval = [{"eval": 1.0}]
    run(make_config(eval_every=2, eval_dataset_path="eval.jsonl"))
    assert "forward_backward" not in harness.tc.calls
    assert eval_steps(caplog) == [0]  # 0 % eval_every == 0 must not skip it


def test_final_eval_runs_when_last_step_was_skipped(harness, caplog):
    caplog.set_level(logging.INFO, logger=td.logger.name)
    harness.train = prompts(MIXED, MIXED) + prompts(MIXED, MIXED, untrainable=True)
    run_config = make_config(eval_every=2, eval_dataset_path="eval.jsonl")
    harness.eval = [{"eval": 1.0}]
    run(run_config)
    assert harness.tc.calls.count("forward_backward") == 1
    assert eval_steps(caplog) == [2]  # step 1 skipped (no datums) -> global_step 2, no in-loop eval


def test_final_eval_not_repeated_after_in_loop_eval(harness, caplog):
    caplog.set_level(logging.INFO, logger=td.logger.name)
    harness.train = prompts(*[MIXED] * 4)
    harness.eval = [{"eval": 1.0}]
    run(make_config(eval_every=2, eval_dataset_path="eval.jsonl"))
    assert eval_steps(caplog) == [2]


# ---------------------------------------------------------------------------
# sampling log line
# ---------------------------------------------------------------------------
def sampling_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == td.logger.name and "sampling:" in r.getMessage()]


def test_sampling_line_logged_every_step(harness, caplog):
    caplog.set_level(logging.INFO, logger=td.logger.name)
    harness.train = prompts(MIXED, SAME, MIXED, MIXED, MIXED, SAME)
    run(make_config())
    lines = sampling_lines(caplog)
    assert len(lines) == 2  # the step that ran out of prompts is logged too
    assert re.search(
        r"gen_batches=2 seen=4 filtered=1 \(25%\) kept=3 empty_rollouts=0 reward=0\.6250 rollout_time=\d+s$", lines[0]
    )
    assert re.search(
        r"gen_batches=1 seen=2 filtered=1 \(50%\) kept=1 empty_rollouts=0 reward=0\.7500 rollout_time=\d+s$", lines[1]
    )
