"""``RolloutSessionAgentLoop``: verl-facing contracts, and the reward/abort taxonomy in particular.

Captured tokens are what decides everything. A rollout that produced any is trained on: with
the reward the container reported, or with a defaulted ``0.0`` when it reported none -- no
matter what failed on the way there. A rollout that produced none is not trainable at all, and
``on_rollout_failure`` (``raise`` | ``empty``) decides what verl is told about it.

The session record carries the taxonomy on four independent flags:

* ``aborted_by_trainer`` -- an exception reached this side of the wire (transport, timeout,
  teardown). Says nothing about whether the rollout trains.
* ``aborted_by_agent``  -- the container's own dump does not describe a finished rollout.
* ``aborted_by_gateway`` -- no trainable tokens were captured. This is the one that means the
  rollout was dropped, and it is what ``on_rollout_failure`` reacts to.
* ``reward_patched``    -- the score the rows trained on was defaulted, not reported. True
  only where something trained: a dropped rollout patches nothing.

Also here: the trajectory-row conversion around verl's fixed-width regions, and the key sets
verl reduces across a batch.
"""

import json
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
    # verl stamps one uid per prompt, shared by that prompt's n rollouts
    kwargs.setdefault("uid", "group-1")
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


async def test_raise_is_the_default():
    """A dropped rollout reaches verl unless a recipe asks for something else."""
    assert make_loop().loop_config.on_rollout_failure == "raise"


@pytest.mark.parametrize("mode", ["typo", "", None, "inert_row"])
async def test_an_unknown_failure_mode_is_rejected_at_construction(mode):
    """A typo would otherwise sit unnoticed until the first dropped rollout, hours into a
    run. ``inert_row`` is among these now: the mode is gone, and a config still asking for
    it must fail loudly rather than fall back to something else."""
    with pytest.raises(ValueError, match="on_rollout_failure must be one of"):
        make_loop(on_rollout_failure=mode)


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

    assert out.extra_fields["request_id"] == loop.session_id
    assert out.extra_fields["trace_index"] == 0
    assert out.extra_fields["num_trace_records"] == 1
    assert out.extra_fields["reward_extra_info"] == {SCORE_KEY: 0.75, "num_trace_records": 1.0}
    # the container's own metrics ride along in the session record
    assert out.extra_fields["metrics"]["agent_latency_s"] == 2.5
    assert out.extra_fields["metrics"]["num_records"] == 1
    assert out.extra_fields["metrics"]["num_turns"] == 1
    assert out.extra_fields["metrics"]["reward_score"] == 0.75
    # a reported reward on a captured trace: nothing was defaulted, nothing aborted
    assert out.extra_fields["metrics"]["reward_patched"] is False
    assert out.extra_fields["metrics"]["aborted"] is False
    assert out.extra_fields["metrics"]["aborted_by_agent"] is False
    assert out.extra_fields["metrics"]["aborted_by_trainer"] is False
    assert out.extra_fields["metrics"]["aborted_by_gateway"] is False

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


async def test_turns_are_counted_before_the_session_is_drained():
    """``finish_session`` consumes the trajectory tree *and* the turn counter, so the count
    has to be read first -- otherwise every rollout reports zero turns."""
    loop = make_loop(session=FakeRolloutSession(turns=3))

    await run_loop(loop)

    assert loop.meta["num_turns"] == 3


async def test_the_prompt_group_uid_becomes_the_group_id_and_drives_priority():
    """One uid per prompt, shared by its n rollouts. verl's name for it is `uid`; everything
    below the loop calls it `group_id`, since a session serves evaluators too. It is what the
    bounds admit rollouts by (so a group is not left half-finished), and what joins a session
    record to its group."""
    loop = make_loop()
    assigner = loop.bounds.container_priority_assigner

    with patch.object(assigner, "get_priority", AsyncMock(side_effect=assigner.get_priority)) as spy:
        await run_loop(loop, uid="group-7")

    assert spy.await_args.args == ("group-7",)
    assert loop.meta["group_id"] == "group-7"


async def test_task_carries_config_kwargs_but_no_tensors():
    session = FakeRolloutSession()
    loop = make_loop(session=session, task_kwargs={"dataset": "gsm8k"})

    await run_loop(loop, input_ids=torch.tensor([1, 2, 3]), uid="u1")

    task = session.tasks[0]
    assert task["dataset"] == "gsm8k"
    assert task["group_id"] == "u1"
    assert task["uid"] == "u1"  # the raw verl field survives too: the container sees the row
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


# -- captured tokens: what the reward becomes ----------------------------------


@pytest.mark.parametrize(
    "dump",
    [
        make_dump(reward=None, exception=None),
        make_dump(reward=None, exception="agent raised: boom"),
    ],
    ids=["no_reward", "agent_exception"],
)
async def test_a_dump_without_a_reward_trains_the_trace_at_zero(dump):
    """The tokens exist, so the trajectory is real policy behaviour and trains. Only the
    score is synthetic, and ``reward_patched`` is what says so."""
    loop = make_loop(session=FakeRolloutSession(dump))

    outputs = await run_loop(loop)

    assert len(outputs) == 1
    assert outputs[0].reward_score == 0.0
    assert outputs[0].extra_fields["reward_extra_info"][SCORE_KEY] == 0.0
    assert loop.meta["reward_patched"] is True
    assert loop.meta["aborted_by_agent"] is True
    assert loop.meta["aborted_by_trainer"] is False
    assert loop.meta["aborted"] is True


async def test_a_trainer_exception_after_the_first_token_trains_the_trace_at_zero():
    """The dominant abort: the run timed out with turns already captured. The loop does not
    let its own exception decide what trains -- the tokens do -- so the partial trajectory
    trains at a defaulted zero, and the exception is recorded as a trainer abort."""
    loop = make_loop(session=FakeRolloutSession(error=TimeoutError("agent_run timed out"), turns=2))

    outputs = await run_loop(loop)

    assert len(outputs) == 1
    assert outputs[0].reward_score == 0.0
    assert loop.meta["reward_patched"] is True
    assert loop.meta["aborted_by_trainer"] is True
    # nothing came back from the container, so it is not the one being blamed
    assert loop.meta["aborted_by_agent"] is False
    # tokens were captured, so this abort is not a drop
    assert loop.meta["aborted_by_gateway"] is False
    # the captured turns were drained out of the gateway, not abandoned in its tree
    assert not loop._gateway.gateway.manager.has_session(loop.session_id)


async def test_a_trainer_exception_is_logged_with_its_root_cause(caplog):
    loop = make_loop(session=FakeRolloutSession(error=RuntimeError("connection reset"), turns=1))

    with caplog.at_level(logging.ERROR):
        await run_loop(loop)

    assert any("connection reset" in record.message for record in caplog.records)


async def test_a_dump_with_a_reward_but_no_task_output_is_still_patched_to_zero():
    """``failure_reason`` is the single gate on the reported reward: a dump that does not
    describe a finished rollout has its score discarded even when it reported one, so on the
    trained path ``aborted_by_agent`` and ``reward_patched`` always agree."""
    loop = make_loop(session=FakeRolloutSession(make_dump(reward=0.5, task_output=None)))

    outputs = await run_loop(loop)

    assert outputs[0].reward_score == 0.0
    assert loop.meta["reward_patched"] is True
    assert loop.meta["aborted_by_agent"] is True


async def test_a_contract_error_is_classified_as_a_trainer_abort():
    """``RolloutContractError`` says no rollout of this run can satisfy the setup, but it is
    raised inside the bounded run like any other exception, so the loop treats it as one: the
    captured trace still trains at zero."""
    loop = make_loop(session=FakeRolloutSession(error=RolloutContractError("non-numeric reward")))

    outputs = await run_loop(loop)

    assert len(outputs) == 1
    assert outputs[0].reward_score == 0.0
    assert loop.meta["aborted_by_trainer"] is True


# -- no captured tokens: what verl is told --------------------------------------


def no_token_sessions():
    """Every way a rollout can finish with nothing trainable, as factories: the fakes
    accumulate lifecycle events, so each test needs its own."""
    return {
        # the harness never called the model (no-op agent, crash on start)
        "no_turns": lambda: FakeRolloutSession(turns=0),
        # the container reported its own failure, before any turn
        "container_reported": lambda: FakeRolloutSession(
            make_dump(reward=None, exception="agent raised: boom"), turns=0
        ),
        # a failure on this side of the wire, before any turn
        "trainer_side": lambda: FakeRolloutSession(error=RuntimeError("connection reset"), turns=0),
        # a stale agent image sending a fixed api key: turns pile up under 'EMPTY' while the
        # real session drains empty
        "static_session_capture": lambda: FakeRolloutSession(session_key="EMPTY"),
    }


@pytest.mark.parametrize("make_session", no_token_sessions().values(), ids=no_token_sessions().keys())
async def test_raise_reports_the_dropped_rollout_to_verl(make_session, fake_upload):
    """``_run_prompt`` marks the whole prompt group ``failure``: sync trains the surviving
    siblings, the async buffers evict and refill the group."""
    loop = make_loop(session=make_session(), on_rollout_failure="raise")

    with pytest.raises(RuntimeError, match="no token records"):
        await run_loop(loop)

    # the dump is written before the raise, so the drop is diagnosable
    assert fake_upload.await_count == 1


@pytest.mark.parametrize("make_session", no_token_sessions().values(), ids=no_token_sessions().keys())
async def test_empty_returns_no_rows_and_does_not_raise(make_session, fake_upload):
    """The prompt group stays ``finished`` and trains its surviving siblings; the only
    record of the drop is the dump and the session record."""
    loop = make_loop(session=make_session(), on_rollout_failure="empty")

    assert await run_loop(loop) == []
    assert fake_upload.await_count == 1


async def test_the_dump_of_a_discarded_rollout_carries_its_diagnosis(fake_upload):
    """A discarded rollout contributes nothing to the step's ``agent_loop/*`` metrics, so the
    S3 dump and the session record are the whole account of it."""
    loop = make_loop(
        session=FakeRolloutSession(error=RuntimeError("connection reset"), turns=0),
        on_rollout_failure="empty",
    )

    await run_loop(loop)

    dump = json.loads(fake_upload.await_args.kwargs["data"])
    assert dump["agent_loop_outputs"] is None
    assert "connection reset" in dump["trainer_exception"]
    assert dump["task"]["uid"] == "group-1"
    assert dump["meta"]["aborted_by_trainer"] is True
    assert dump["meta"]["aborted_by_gateway"] is True
    assert dump["meta"]["reward_score"] is None
    # the whole session record reaches S3, including the spans and timeout flags a discarded
    # session is diagnosed from -- a typed ``meta`` would drop every key it did not declare
    assert dump["meta"]["container_setup_timeout_exceeded"] == 0.0
    assert "agent_run_start_at" in dump["meta"]
    # every span is closed before the upload, so the dump differs from the session record by
    # exactly the uri of the dump itself
    assert dump["meta"]["rollout_session_end_at"]
    assert set(loop.meta) - set(dump["meta"]) == {"output_s3_uri"}


async def test_a_dropped_rollout_is_flagged_by_the_gateway_not_by_the_other_two():
    """``aborted_by_gateway`` is the drop, and it is independent of the other two flags: here
    the container finished cleanly and nothing failed on the trainer's side -- the harness
    simply never called the model. Nothing was patched, because nothing trained."""
    loop = make_loop(session=FakeRolloutSession(turns=0), on_rollout_failure="empty")

    assert await run_loop(loop) == []

    assert loop.meta["aborted_by_gateway"] is True
    assert loop.meta["aborted"] is True
    assert loop.meta["aborted_by_agent"] is False
    assert loop.meta["aborted_by_trainer"] is False
    assert loop.meta["reward_patched"] is False
    # the reward the container reported is still recorded, though no row carried it into a
    # training step -- aborted_by_gateway is what says this score never trained anything
    assert loop.meta["reward_score"] == 1.0


async def test_a_dropped_rollout_with_no_dump_records_no_score():
    """Nothing reported a reward and nothing trained, so the score is absent rather than a
    zero that would read as a real one."""
    loop = make_loop(session=FakeRolloutSession(make_dump(reward=None), turns=0), on_rollout_failure="empty")

    assert await run_loop(loop) == []

    assert loop.meta["aborted_by_gateway"] is True
    assert loop.meta["aborted_by_agent"] is True
    assert loop.meta["reward_score"] is None
    assert loop.meta["reward_patched"] is False


def drained_the_gateway_session(loop) -> bool:
    """Whether the drain reached the gateway: ``finish_session`` is what releases the sid's
    ``Session`` on the primary adapter.

    Scoped to the primary adapter deliberately -- ``RolloutGateway.finish_session`` drains
    every adapter but only pops the primary's store, so a non-primary adapter keeps its
    entry no matter what the loop does. That is the gateway's bug, not this loop's contract.
    """
    return loop.session_id not in loop._gateway.gateway.adapters[0].store


@pytest.mark.parametrize("make_session", no_token_sessions().values(), ids=no_token_sessions().keys())
async def test_a_dropped_rollout_still_releases_its_gateway_session(make_session):
    """Every exit from the rollout drains the session, including the ones that end in a
    dropped rollout -- otherwise the gateway accumulates per-session state per drop."""
    loop = make_loop(session=make_session(), on_rollout_failure="empty")

    await run_loop(loop)

    assert drained_the_gateway_session(loop)
    assert loop.test_session.events == ["setup", "run", "shutdown"]


async def test_a_setup_failure_still_releases_its_gateway_session():
    """A container that never came up. The gateway session is opened before the container
    is provisioned, so it exists and has to be drained even though nothing ever used it --
    and the container is never asked to run."""
    loop = make_loop(session=FakeRolloutSession(setup_error=RuntimeError("no capacity")), on_rollout_failure="empty")

    assert await run_loop(loop) == []

    assert loop.meta["aborted_by_trainer"] is True
    assert drained_the_gateway_session(loop)
    assert loop.test_session.events == ["setup", "shutdown"]


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


# -- the key sets verl reduces across a batch ----------------------------------


async def test_reward_extra_info_carries_declared_metrics_only():
    loop = make_loop(reward_extra_info_defaults={"f_beta": 0.0, "num_turns": 0.0, "reward": 0.0})
    dump = make_dump(reward=0.75, metrics={"f_beta": 0.6, "num_turns": 3, "undeclared": 9.0})
    loop.test_session.dump = dump

    outputs = await run_loop(loop)

    info = outputs[-1].extra_fields["reward_extra_info"]
    assert info["f_beta"] == 0.6
    assert info["num_turns"] == 3.0
    assert info[SCORE_KEY] == 0.75
    assert info["num_trace_records"] == 1.0
    assert "undeclared" not in info
    assert "reward" not in info  # verl derives its own reward metric from rm_scores


async def test_reward_extra_info_key_set_survives_a_missing_dump():
    """verl reduces this dict across the batch, so a key only some rows carry is averaged
    over the wrong denominator -- and the v1 replay buffer raises on a missing one. A rollout
    whose dump never arrived reports every declared key at its default and the score at 0.0."""
    defaults = {"submitted": 0.0, "num_turns": 0.0, "ok": 0.0}
    loop = make_loop(reward_extra_info_defaults=defaults)

    reported = loop._reward_extra_info(make_dump(reward=1.0, metrics={"submitted": 1.0, "num_turns": 4}), 1, 1.0)
    defaulted = loop._reward_extra_info(None, 1, 0.0)

    assert reported == {
        "submitted": 1.0,
        "num_turns": 4.0,
        "ok": 0.0,
        SCORE_KEY: 1.0,
        "num_trace_records": 1.0,
    }
    assert defaulted == {
        "submitted": 0.0,
        "num_turns": 0.0,
        "ok": 0.0,
        SCORE_KEY: 0.0,
        "num_trace_records": 1.0,
    }
    assert set(reported) == set(defaulted)


async def test_extra_field_key_set_matches_the_declared_contract():
    """The TransferQueue stores one field set for the whole run. ``metrics`` is stamped onto
    the outputs after they are built, so the declared set is only complete if that step ran
    for every row of every rollout."""
    loop = make_loop()

    reported = await run_loop(loop)
    defaulted = await run_loop(make_loop(session=FakeRolloutSession(make_dump(reward=None))))

    assert set(reported[0].extra_fields) == set(rsal.ExtraFields.__annotations__)
    assert set(defaulted[0].extra_fields) == set(rsal.ExtraFields.__annotations__)


async def test_trainer_config_is_not_required_to_declare_extra_info_defaults():
    loop = make_loop(trainer_config=make_trainer_config())

    assert loop._rei_defaults == {}
