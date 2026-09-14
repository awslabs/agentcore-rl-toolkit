"""``RolloutSessionAgentLoop``: verl-facing contracts, and the failure path in particular.

What a failed rollout reports to verl is a configured choice (``on_rollout_failure``:
``raise`` | ``empty`` | ``inert_row``), so these tests pin two things separately: which
conditions count as a failed rollout at all, and what each setting then does with one.
Two conditions ignore the setting and always raise -- a ``RolloutContractError`` (which
every rollout of the run would hit) and a task with no ``task_id``. Also here: the
trajectory-row conversion around verl's fixed-width regions, and the key sets verl reduces
across a batch.

Most failure-cause tests run under ``inert_row``, the one setting whose row carries the
diagnosis (``failure_reason``) they assert on.
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest
import torch
from verl.workers.rollout.replica import TokenOutput

from agentcore_rl_toolkit.backends.experimental.verl import rollout_session_agent_loop as rsal
from agentcore_rl_toolkit.rollout_gateway import TraceRecord
from agentcore_rl_toolkit.rollout_session.errors import RolloutContractError

from .conftest import (
    FakeLLMServerClient,
    FakeRolloutSession,
    make_data_config,
    make_dump,
    make_loop,
    make_trainer_config,
)

# The loop names its own score inside reward_extra_info; these tests care that the key is
# one stable name on every row, not what that name is.
SCORE_KEY = rsal.REWARD_EXTRA_INFO_SCORE_KEY

pytestmark = pytest.mark.asyncio


async def run_loop(loop, sampling_params: dict | None = None, **kwargs):
    """Dispatch one rollout the way the TransferQueue runner does."""
    kwargs.setdefault("task_id", "task-1")
    kwargs.setdefault("global_steps", 3)
    # verl always resolves the whole sampling triple from its rollout config
    return await loop.run(sampling_params or {"temperature": 0.7, "top_p": 1.0, "top_k": -1}, **kwargs)


# -- construction contracts -----------------------------------------------------


async def test_use_v1_required():
    with pytest.raises(ValueError, match="use_v1"):
        make_loop(use_v1=False)


async def test_explicit_max_model_len_is_required():
    with pytest.raises(ValueError, match="explicit positive.*max_model_len"):
        make_loop(max_model_len=None)


@pytest.mark.parametrize("field", ["prompt_length", "response_length"])
async def test_padded_region_cannot_exceed_max_model_len(field):
    with pytest.raises(ValueError, match=rf"{field} \(129\) cannot exceed .*max_model_len \(128\)"):
        make_loop(max_model_len=128, **{field: 129})


@pytest.mark.parametrize("value", [0, -1, True, 129])
async def test_invalid_max_new_tokens_rejected(value):
    with pytest.raises(ValueError, match="max_new_tokens"):
        make_loop(rollout_gateway_sampling_params={"max_new_tokens": value})


async def test_trainer_chat_template_kwargs_reach_the_gateway():
    """The gateway renders the turns training will be scored on, so it must tokenize
    exactly as the trainer does -- and nothing else forwards ``data.*`` to it."""
    with patch.object(rsal, "get_or_start_gateway", wraps=rsal.get_or_start_gateway) as spy:
        make_loop(data_config=make_data_config({"enable_thinking": False}))

    assert spy.call_args.kwargs["chat_template_kwargs"] == {"enable_thinking": False}


async def test_configured_chat_template_kwargs_win():
    with patch.object(rsal, "get_or_start_gateway", wraps=rsal.get_or_start_gateway) as spy:
        make_loop(
            rollout_gateway={"host": "127.0.0.1", "public_host": "127.0.0.1", "chat_template_kwargs": {"custom": 1}},
            data_config=make_data_config({"enable_thinking": False}),
        )

    assert spy.call_args.kwargs["chat_template_kwargs"] == {"custom": 1}


# -- the successful path --------------------------------------------------------


async def test_run_end_to_end_single_trace(fake_upload):
    llm = FakeLLMServerClient()
    session = FakeRolloutSession(make_dump(reward=0.75, metrics={"agent_latency_s": 2.5}))
    loop = make_loop(llm=llm, session=session)

    outputs = await run_loop(loop, {"temperature": 0.7, "top_p": 0.9, "top_k": -1}, uid="u1")

    assert len(outputs) == 1
    out = outputs[0]
    # the response region is exactly what the fake engine generated
    assert out.response_ids == [101, 102]
    assert out.response_mask == [1, 1]
    assert out.response_logprobs == [-0.5, -0.6]
    assert out.prompt_ids
    assert out.reward_score == 0.75
    # verl counts the initial user turn, which the gateway never sees
    assert out.num_turns == 2

    assert out.extra_fields["rollout_failed"] == 0.0
    assert out.extra_fields["failure_reason"] is None
    assert out.extra_fields["request_id"] == loop.session_id
    assert out.extra_fields["trace_index"] == 0
    assert out.extra_fields["num_trace_records"] == 1
    assert out.extra_fields["reward_extra_info"] == {SCORE_KEY: 0.75, "rollout_failed": 0.0, "num_trace_records": 1.0}
    # the container's own metrics ride along in the session record
    assert out.extra_fields["metrics"]["agent_latency_s"] == 2.5
    assert out.extra_fields["metrics"]["num_records"] == 1
    assert out.extra_fields["metrics"]["reward_score"] == 0.75
    assert out.extra_fields["metrics"]["aborted"] is False

    # the session ran once, under its own lifecycle, and was torn down
    assert session.events == ["setup", "run", "shutdown"]
    task = session.tasks[0]
    assert task["task_id"] == "task-1"
    assert task["llm"]["api_key"] == loop.session_id  # the gateway capture key
    assert task["llm"]["base_url"] == f"{loop._gateway.base_url}/v1"
    assert task["llm"]["model"] == "openai/model"
    assert task["sampling_params"] == {"temperature": 0.7, "top_p": 0.9, "top_k": -1}
    # sticky routing: the engine sees the session id as its request id
    assert llm.calls[0]["request_id"] == loop.session_id

    # one S3 dump per rollout, whose uri is recorded on the session
    assert fake_upload.await_count == 1
    assert fake_upload.await_args.kwargs["s3_uri"] == loop.meta["output_s3_uri"]


async def test_multi_turn_merges_to_one_record():
    loop = make_loop(session=FakeRolloutSession(turns=3))

    outputs = await run_loop(loop)

    # CLEAN prefix extensions merge into ONE record with 3 trained turns
    assert len(outputs) == 1
    assert sum(outputs[0].response_mask) == 6
    assert outputs[0].num_turns == 4


async def test_task_carries_config_kwargs_but_no_tensors():
    session = FakeRolloutSession()
    loop = make_loop(session=session, task_kwargs={"dataset": "gsm8k"})

    await run_loop(loop, input_ids=torch.tensor([1, 2, 3]), uid="u1")

    task = session.tasks[0]
    assert task["dataset"] == "gsm8k"
    assert task["uid"] == "u1"
    assert "input_ids" not in task  # not JSON-serializable, and the container has no use for it


async def test_sampling_defaults_reach_the_engine():
    llm = FakeLLMServerClient()
    loop = make_loop(llm=llm)

    await run_loop(loop, {"temperature": 0.3, "top_p": 0.8, "top_k": 20})

    params = llm.calls[0]["sampling_params"]
    assert params["temperature"] == 0.3
    assert params["top_p"] == 0.8
    assert params["top_k"] == 20
    assert params["max_new_tokens"] == 8  # from rollout_gateway_sampling_params


async def test_session_budget_matches_response_storage():
    loop = make_loop(prompt_length=8, response_length=24, max_model_len=32)
    gateway = loop._gateway.gateway

    with patch.object(gateway, "create_session", wraps=gateway.create_session) as create_session:
        await run_loop(loop)

    max_context_tokens = create_session.call_args.kwargs["max_context_tokens"]
    assert max_context_tokens == 24
    assert max_context_tokens == loop.response_length
    assert max_context_tokens != loop.prompt_length + loop.response_length


async def test_staleness_defaults_to_the_dispatch_step():
    """An engine that reported no weight version must not inflate trajectory staleness."""
    loop = make_loop()

    outputs = await run_loop(loop, global_steps=7)

    assert outputs[0].extra_fields["min_global_steps"] == 7
    assert outputs[0].extra_fields["max_global_steps"] == 7


async def test_zero_weight_version_is_not_mistaken_for_absent():
    """Step 0 is a real weight version: the freshest possible sample, not a missing one."""
    llm = FakeLLMServerClient(
        [
            TokenOutput(
                token_ids=[101, 102],
                log_probs=[-0.5, -0.6],
                stop_reason="completed",
                extra_fields={"min_global_steps": 0, "max_global_steps": 0},
            )
        ]
    )
    loop = make_loop(llm=llm)

    outputs = await run_loop(loop, global_steps=7)

    assert outputs[0].extra_fields["min_global_steps"] == 0
    assert outputs[0].extra_fields["max_global_steps"] == 0


async def test_every_leaf_is_emitted_with_the_primary_record_last():
    """A forked session (sub-agent, compaction) trains every leaf, and verl scores and
    broadcasts from the last row -- so the most-trained leaf goes there."""
    loop = make_loop()
    records = [
        TraceRecord(token_ids=[1, 2, 3], loss_mask=[1], logprobs=[-0.1]),
        TraceRecord(token_ids=[1, 2, 4, 5, 6], loss_mask=[1, 1, 1], logprobs=[-0.2, -0.3, -0.4]),
        TraceRecord(token_ids=[1, 2, 7, 8], loss_mask=[1, 1], logprobs=[-0.5, -0.6]),
    ]
    with patch.object(loop._gateway.gateway, "finish_session", AsyncMock(return_value=records)):
        outputs = await run_loop(loop)

    assert [sum(o.response_mask) for o in outputs] == [1, 2, 3]
    assert [o.extra_fields["trace_index"] for o in outputs] == [0, 1, 2]
    assert {o.extra_fields["num_trace_records"] for o in outputs} == {3}
    # every row carries the session's reward, since GRPO scores the group from the last one
    assert {o.reward_score for o in outputs} == {1.0}
    # session-level totals are not counted once per leaf
    assert outputs[0].extra_fields["metrics"]["llm_generated_length"] == 6.0
    assert outputs[0].extra_fields["metrics"]["num_records"] == 3


async def test_records_without_tokens_are_dropped():
    loop = make_loop()
    records = [
        TraceRecord(token_ids=[], loss_mask=[], logprobs=[]),
        TraceRecord(token_ids=[1, 2, 3], loss_mask=[1], logprobs=[-0.1]),
    ]
    with patch.object(loop._gateway.gateway, "finish_session", AsyncMock(return_value=records)):
        outputs = await run_loop(loop)

    assert len(outputs) == 1
    assert outputs[0].extra_fields["num_trace_records"] == 1


# -- what each on_rollout_failure setting reports ------------------------------


async def test_raise_is_the_default():
    """Removing the trainer-side isolation mixin must not put anyone on a biased path by
    default, so the failure reaches verl unless a recipe asks for something else."""
    assert make_loop().loop_config.on_rollout_failure == "raise"


@pytest.mark.parametrize("mode", ["typo", "", None, "inert row"])
async def test_an_unknown_failure_mode_is_rejected_at_construction(mode):
    """A typo would otherwise sit unnoticed until the first failure, hours into a run."""
    with pytest.raises(ValueError, match="on_rollout_failure must be one of"):
        make_loop(on_rollout_failure=mode)


async def test_raise_reraises_the_trainer_side_exception_after_the_dump(fake_upload):
    """The original exception, not a wrapper: its traceback is the diagnosis."""
    loop = make_loop(session=FakeRolloutSession(error=RuntimeError("connection reset")), on_rollout_failure="raise")

    with pytest.raises(RuntimeError, match="connection reset"):
        await run_loop(loop)

    assert fake_upload.await_count == 1  # on record before the raise
    assert loop.meta["aborted"] is True


async def test_raise_synthesizes_an_error_for_a_container_reported_failure(fake_upload):
    """Nothing raised on this side of the wire -- the container reported the failure in its
    dump -- so the reason it gave becomes the exception verl sees."""
    loop = make_loop(
        session=FakeRolloutSession(make_dump(reward=None, exception="agent raised: boom")),
        on_rollout_failure="raise",
    )

    with pytest.raises(rsal.RolloutFailedError, match="agent raised: boom"):
        await run_loop(loop)

    assert fake_upload.await_count == 1


@pytest.mark.parametrize(
    "session",
    [
        FakeRolloutSession(make_dump(reward=None, exception="agent raised: boom")),
        FakeRolloutSession(error=RuntimeError("connection reset")),
    ],
    ids=["container_reported", "trainer_side"],
)
async def test_empty_returns_no_rows_and_does_not_raise(session, fake_upload):
    """The prompt group stays ``finished`` and trains its surviving siblings; the only
    record of the failure is the dump."""
    loop = make_loop(session=session, on_rollout_failure="empty")

    assert await run_loop(loop) == []
    assert loop.meta["aborted"] is True
    assert fake_upload.await_count == 1


async def test_inert_row_returns_one_stand_in_row(fake_upload):
    loop = make_loop(
        session=FakeRolloutSession(make_dump(reward=None, exception="agent raised: boom")),
        on_rollout_failure="inert_row",
    )

    outputs = await run_loop(loop)

    assert len(outputs) == 1
    assert_inert_row(outputs[0])
    assert outputs[0].extra_fields["failure_reason"] == "agent raised: boom"
    assert loop.meta["aborted"] is True
    # the dump is still written, so the failure is diagnosable
    assert fake_upload.await_count == 1


@pytest.mark.parametrize("mode", ["raise", "empty", "inert_row"])
async def test_a_contract_error_raises_whatever_the_failure_mode(mode, fake_upload):
    """No rollout of this run could satisfy the contract, so absorbing it would spend the
    whole job producing nothing trainable."""
    session = FakeRolloutSession(error=RolloutContractError("non-numeric reward"))
    loop = make_loop(session=session, on_rollout_failure=mode)

    with pytest.raises(RolloutContractError, match="non-numeric reward"):
        await run_loop(loop)

    # the offending task is on record before the raise
    assert fake_upload.await_count == 1


# -- what counts as a failed rollout -------------------------------------------


def failing_loop(session, **overrides):
    """A loop that reports failures as inert rows, so the diagnosis is assertable."""
    return make_loop(session=session, on_rollout_failure="inert_row", **overrides)


def assert_inert_row(output, *, dispatch_step: int = 3):
    """The shape a failed rollout's stand-in row must have: it trains nothing."""
    assert output.response_mask == [0]
    assert len(output.response_ids) == 1
    # always present, so the batch's rollout_log_probs field set is uniform
    assert output.response_logprobs == [0.0]
    assert output.reward_score == 0.0
    assert output.num_turns == 0
    assert output.extra_fields["rollout_failed"] == 1.0
    assert output.extra_fields["num_trace_records"] == 0
    assert output.extra_fields["reward_extra_info"]["rollout_failed"] == 1.0
    assert output.extra_fields["reward_extra_info"][SCORE_KEY] == 0.0
    # staleness stands in the dispatch step rather than inflating to global_steps
    assert output.extra_fields["min_global_steps"] == dispatch_step
    assert output.extra_fields["max_global_steps"] == dispatch_step


@pytest.mark.parametrize(
    ("dump", "reason"),
    [
        (make_dump(reward=None, exception=None), "no reward"),
        (make_dump(task_output=None, exception=None), "no task output"),
    ],
)
async def test_a_dump_missing_its_result_is_a_failed_rollout(dump, reason):
    loop = failing_loop(FakeRolloutSession(dump))

    outputs = await run_loop(loop)

    assert len(outputs) == 1
    assert_inert_row(outputs[0])
    assert reason in outputs[0].extra_fields["failure_reason"]


async def test_a_trainer_side_error_is_a_failed_rollout(caplog):
    """Transport, timeout, teardown: a failure on this side of the wire, logged and
    reported the same way as one the container reports."""
    loop = failing_loop(FakeRolloutSession(error=RuntimeError("connection reset")))

    with caplog.at_level(logging.ERROR):
        outputs = await run_loop(loop)

    assert len(outputs) == 1
    assert_inert_row(outputs[0])
    assert "connection reset" in outputs[0].extra_fields["failure_reason"]
    assert any("Failed rollout" in record.message for record in caplog.records)


async def test_a_failed_rollout_discards_its_partial_trace():
    """The partial trace exists because the container died, so a zero reward on it would
    describe the infrastructure, not the policy."""
    session = FakeRolloutSession(make_dump(reward=None, exception="died mid-run"), turns=2)
    loop = failing_loop(session)

    outputs = await run_loop(loop)

    assert len(outputs) == 1
    assert_inert_row(outputs[0])


async def test_a_failing_rollout_still_releases_its_gateway_session():
    """The gateway holds a session's trajectory tree until it is drained, so a failure
    that skipped ``finish_session`` would leak one tree per failed rollout."""
    loop = failing_loop(FakeRolloutSession(error=RuntimeError("boom")))

    await run_loop(loop)

    assert loop._gateway.gateway.manager.turn_count(loop.session_id) == 0
    assert loop.test_session.events == ["setup", "run", "shutdown"]


async def test_a_setup_failure_does_not_finish_an_uncreated_session():
    """A container that never came up: the gateway session was created before the run, so
    it is still drained -- and the setup error is what gets reported, not a gateway error."""
    loop = failing_loop(FakeRolloutSession(setup_error=RuntimeError("no capacity")))

    outputs = await run_loop(loop)

    assert "no capacity" in outputs[0].extra_fields["failure_reason"]
    assert loop.test_session.events == ["setup", "shutdown"]


async def test_no_trainable_trace_is_a_failed_rollout():
    """The agent never called the model (a no-op harness, an agent that crashed on start)."""
    loop = failing_loop(FakeRolloutSession(turns=0))

    outputs = await run_loop(loop)

    assert len(outputs) == 1
    assert_inert_row(outputs[0])
    assert outputs[0].extra_fields["failure_reason"] == "the rollout produced no trainable trajectory"


async def test_static_session_capture_is_diagnosed(caplog):
    """Stale agent image: the agent sends a fixed api key, so the real sid drains empty
    while turns pile up under 'EMPTY'. Without the warning this trains nothing, silently."""
    loop = failing_loop(FakeRolloutSession(session_key="EMPTY"))

    with caplog.at_level(logging.WARNING):
        outputs = await run_loop(loop)

    assert_inert_row(outputs[0])
    assert any("static session 'EMPTY'" in record.message for record in caplog.records)


async def test_a_task_without_a_task_id_raises_before_anything_runs():
    session = FakeRolloutSession()
    loop = make_loop(session=session)

    with pytest.raises(RolloutContractError, match="task_id"):
        await loop.run({"temperature": 1.0, "top_p": 1.0}, task_id=None, global_steps=1)

    assert session.events == []


async def test_run_is_one_shot():
    """One instance holds one rollout's session id, capture key and session record."""
    loop = make_loop()
    await run_loop(loop)

    with pytest.raises(RuntimeError, match="called twice"):
        await run_loop(loop)


# -- row conversion ------------------------------------------------------------


async def test_prompt_overflow_moves_into_the_response_region():
    """verl stores prompts in a fixed-width region; the overflow keeps its place in the
    token order, attended to but carrying no loss and no rollout logprob."""
    loop = make_loop(prompt_length=2, response_length=6, max_model_len=16)
    record = TraceRecord(token_ids=[1, 2, 3, 4, 5, 6], loss_mask=[1, 1], logprobs=[-0.5, -0.6])

    out = loop._record_to_output(
        record,
        index=0,
        num_records=1,
        num_turns=1,
        reward_score=1.0,
        staleness=(0, 0),
        reward_extra_info={},
    )

    assert out.prompt_ids == [1, 2]
    assert out.response_ids == [3, 4, 5, 6]
    assert out.response_mask == [0, 0, 1, 1]
    assert out.response_logprobs == [0.0, 0.0, -0.5, -0.6]


async def test_a_prompt_that_fills_the_budget_leaves_an_inert_response():
    """An empty response region crashes verl's ``AgentLoopOutput.as_dict``, so one masked
    filler token stands in -- with no loss, since the tokens were never sampled here."""
    loop = make_loop(prompt_length=4, response_length=4, max_model_len=8)
    record = TraceRecord(token_ids=[1, 2, 3, 4, 5, 6], loss_mask=[1, 1], logprobs=[-0.5, -0.6])

    out = loop._record_to_output(
        record,
        index=0,
        num_records=1,
        num_turns=1,
        reward_score=1.0,
        staleness=(0, 0),
        reward_extra_info={},
    )

    assert out.prompt_ids == [1, 2, 3, 4]
    assert out.response_mask == [0]
    assert len(out.response_ids) == 1
    assert out.response_logprobs == [0.0]


async def test_the_failed_row_is_shaped_like_a_padding_row():
    loop = make_loop()
    await loop.meta.set("step", 5)

    out = loop.make_failed_loop_output("boom")

    assert out.prompt_ids == [loop._pad_token_id()]
    assert out.response_ids == [loop._pad_token_id()]
    assert_inert_row(out, dispatch_step=5)
    assert out.extra_fields["failure_reason"] == "boom"


# -- reward_extra_info ---------------------------------------------------------


async def test_reward_extra_info_carries_declared_metrics_only():
    loop = make_loop(reward_extra_info_defaults={"f_beta": 0.0, "num_turns": 0.0, "reward": 0.0})
    dump = make_dump(reward=0.75, metrics={"f_beta": 0.6, "num_turns": 3, "undeclared": 9.0})
    loop.test_session.dump = dump

    outputs = await run_loop(loop)

    info = outputs[-1].extra_fields["reward_extra_info"]
    assert info["f_beta"] == 0.6
    assert info["num_turns"] == 3.0
    assert info[SCORE_KEY] == 0.75
    assert info["rollout_failed"] == 0.0
    assert info["num_trace_records"] == 1.0
    assert "undeclared" not in info
    assert "reward" not in info  # verl derives its own reward metric from rm_scores


async def test_reward_extra_info_key_set_is_identical_on_success_and_failure():
    """verl reduces this dict across the batch, so a key only some rows carry is averaged
    over the wrong denominator -- and the v1 replay buffer raises on a missing one."""
    defaults = {"submitted": 0.0, "num_turns": 0.0, "ok": 0.0}
    loop = make_loop(reward_extra_info_defaults=defaults)

    successful = loop._reward_extra_info(
        make_dump(reward=1.0, metrics={"submitted": 1.0, "num_turns": 4}), 1, 1.0, failed=False
    )
    failed = loop._reward_extra_info(None, 0, 0.0, failed=True)

    assert successful == {
        "submitted": 1.0,
        "num_turns": 4.0,
        "ok": 0.0,
        SCORE_KEY: 1.0,
        "rollout_failed": 0.0,
        "num_trace_records": 1.0,
    }
    assert failed == {
        "submitted": 0.0,
        "num_turns": 0.0,
        "ok": 0.0,
        SCORE_KEY: 0.0,
        "rollout_failed": 1.0,
        "num_trace_records": 0.0,
    }
    assert set(successful) == set(failed)


async def test_extra_field_key_set_is_identical_on_success_and_failure():
    """The TransferQueue stores one field set for the whole run -- which only matters for the
    setting that emits a row for a failed rollout."""
    successful_loop = make_loop()
    outputs = await run_loop(successful_loop)
    failed_loop = failing_loop(FakeRolloutSession(error=RuntimeError("boom")))
    failed_outputs = await run_loop(failed_loop)

    assert set(outputs[0].extra_fields) == set(failed_outputs[0].extra_fields)
    assert set(outputs[0].extra_fields) == set(rsal.ExtraFields.__annotations__)


async def test_trainer_config_is_not_required_to_declare_extra_info_defaults():
    loop = make_loop(trainer_config=make_trainer_config())

    assert loop._rei_defaults == {}
