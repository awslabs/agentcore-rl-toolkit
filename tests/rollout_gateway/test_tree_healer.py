"""Correctness tests for TreeHealer (generate-time prefix healing per tree branch).

These drive the healer + a real TrajectoryManager the way ``BaseAdapter._run_turn`` does
(heal -> generate -> record_turn), reusing the drift-producing FakeRenderer from the
LinearHealer tests: a replayed assistant message re-renders to different ids than the ones
the backend served, while user messages re-render identically.
"""

import pytest

from agentcore_rl_toolkit.rollout_gateway import BaseTrace, TrajectoryManager, TreeHealer, TurnRecord

from .test_linear_healer import A_CLOSE, A_OPEN, U_CLOSE, U_OPEN, FakeRenderer, _a, _enc, _u

TOOLS = [{"type": "function", "function": {"name": "calc", "parameters": {"type": "object"}}}]


async def _drive_turn(mgr, healer, sid, *, messages, served_ids, response_message, tools=None, use_healer=True):
    """Mirror BaseAdapter._run_turn for one turn. With ``use_healer=False`` feed the raw
    drifted render (tree mode) so tests can contrast the two paths on identical inputs."""
    if use_healer:
        prompt_ids = await healer.heal(sid, messages, tools)
    else:
        prompt_ids = await healer.renderer.render(messages, tools=tools, add_generation_prompt=True)
    turn = TurnRecord(prompt_ids=list(prompt_ids), output_ids=list(served_ids), finish_reason="stop")
    mgr.record_turn(sid, turn=turn, prompt_messages=messages, response_message=response_message, tools=tools)
    return prompt_ids


def _prompt(text):
    return [U_OPEN, *_enc(text), U_CLOSE, A_OPEN]


def _tail(text):
    return [A_CLOSE, U_OPEN, *_enc(text), U_CLOSE, A_OPEN]


async def _retry_session(*, use_healer):
    """Turn 1, turn 2, then a retry of turn 2 with the same messages: a sibling branch."""
    mgr = TrajectoryManager()
    healer = TreeHealer(FakeRenderer(), mgr)
    await _drive_turn(mgr, healer, "s", messages=[_u("q1")], served_ids=[2001], response_message=_a("r1"))
    history = [_u("q1"), _a("r1"), _u("q2")]
    first = await _drive_turn(
        mgr, healer, "s", messages=history, served_ids=[2002], response_message=_a("r2"), use_healer=use_healer
    )
    retry = await _drive_turn(
        mgr, healer, "s", messages=history, served_ids=[2003], response_message=_a("r2b"), use_healer=use_healer
    )
    return mgr, healer, first, retry


@pytest.mark.asyncio
async def test_linear_history_stays_one_sample():
    """On a linear history the tree healer behaves like the linear healer: every turn
    extends the previous turn's served ids and the session is one sample."""
    mgr = TrajectoryManager()
    healer = TreeHealer(FakeRenderer(), mgr)
    await _drive_turn(mgr, healer, "s", messages=[_u("q1")], served_ids=[2001], response_message=_a("r1"))
    p2 = await _drive_turn(
        mgr, healer, "s", messages=[_u("q1"), _a("r1"), _u("q2")], served_ids=[2002], response_message=_a("r2")
    )
    p3 = await _drive_turn(
        mgr,
        healer,
        "s",
        messages=[_u("q1"), _a("r1"), _u("q2"), _a("r2"), _u("q3")],
        served_ids=[2003],
        response_message=_a("r3"),
    )

    assert p2 == _prompt("q1") + [2001] + _tail("q2")
    assert p3 == p2 + [2002] + _tail("q3")
    recs = mgr.get_trajectory("s", base_sample=BaseTrace(index=0), reward=1.0)
    assert len(recs) == 1
    assert recs[0].token_ids == p3 + [2003]
    assert healer.pop_stats("s")["healed_turns"] == 2


@pytest.mark.asyncio
async def test_retry_branch_is_healed_on_its_anchor():
    """A retried turn anchors on the same generated turn as the first attempt, so both
    siblings extend turn 1's served ids. Each leaf is one CLEAN sample; turn 1 is trained
    once and re-emitted as context on the second leaf."""
    mgr, healer, first, retry = await _retry_session(use_healer=True)

    assert first == retry == _prompt("q1") + [2001] + _tail("q2")
    recs = mgr.get_trajectory("s", base_sample=BaseTrace(index=0), reward=1.0)
    assert [r.token_ids for r in recs] == [first + [2002], retry + [2003]]
    assert sum(r.loss_mask.count(1) for r in recs) == 3  # turn 1 once, each attempt once
    assert healer.pop_stats("s")["healed_turns"] == 2


@pytest.mark.asyncio
async def test_retry_branch_forks_without_healing():
    """Baseline for the test above: feeding the drifted full render forks each branch."""
    mgr, _, _, _ = await _retry_session(use_healer=False)
    recs = mgr.get_trajectory("s", base_sample=BaseTrace(index=0), reward=1.0)
    assert len(recs) == 3


@pytest.mark.asyncio
async def test_edit_branches_from_earlier_turn():
    """A request that edits a later message anchors on the deepest generated turn its
    messages still match, not on the latest turn."""
    mgr = TrajectoryManager()
    healer = TreeHealer(FakeRenderer(), mgr)
    await _drive_turn(mgr, healer, "s", messages=[_u("q1")], served_ids=[2001], response_message=_a("r1"))
    await _drive_turn(
        mgr, healer, "s", messages=[_u("q1"), _a("r1"), _u("q2")], served_ids=[2002], response_message=_a("r2")
    )
    edited = await _drive_turn(
        mgr, healer, "s", messages=[_u("q1"), _a("r1"), _u("q2 edited")], served_ids=[2003], response_message=_a("x")
    )

    assert edited == _prompt("q1") + [2001] + _tail("q2 edited")
    recs = mgr.get_trajectory("s", base_sample=BaseTrace(index=0), reward=1.0)
    assert len(recs) == 2


@pytest.mark.asyncio
async def test_new_root_is_rendered_in_full():
    """A request matching no generated turn (e.g. a sub-agent with its own first message)
    is rendered in full and counted as unanchored."""
    mgr = TrajectoryManager()
    healer = TreeHealer(FakeRenderer(), mgr)
    await _drive_turn(mgr, healer, "s", messages=[_u("q1")], served_ids=[2001], response_message=_a("r1"))
    sub = await _drive_turn(mgr, healer, "s", messages=[_u("sub task")], served_ids=[2002], response_message=_a("s1"))

    assert sub == _prompt("sub task")
    assert healer.pop_stats("s")["unanchored"] == 2  # both first messages


@pytest.mark.asyncio
async def test_tools_change_is_rendered_in_full():
    mgr = TrajectoryManager()
    healer = TreeHealer(FakeRenderer(), mgr)
    await _drive_turn(mgr, healer, "s", messages=[_u("q1")], served_ids=[2001], response_message=_a("r1"), tools=TOOLS)
    messages = [_u("q1"), _a("r1"), _u("q2")]
    prompt = await healer.heal("s", messages, None)

    assert prompt == await healer.renderer.render(messages, add_generation_prompt=True)
    assert healer.pop_stats("s")["tools_changed"] == 1


class PrefixChangingRenderer(FakeRenderer):
    """A template whose earlier blocks change when messages are appended (here: a leading
    message count), so no splice onto the served ids is valid."""

    async def render(self, messages, *, tools=None, add_generation_prompt=True):
        ids = await super().render(messages, tools=tools, add_generation_prompt=add_generation_prompt)
        return [len(messages), *ids]


@pytest.mark.asyncio
async def test_unhealable_template_is_rendered_in_full():
    mgr = TrajectoryManager()
    healer = TreeHealer(PrefixChangingRenderer(), mgr)
    await _drive_turn(mgr, healer, "s", messages=[_u("q1")], served_ids=[2001], response_message=_a("r1"))
    messages = [_u("q1"), _a("r1"), _u("q2")]
    prompt = await healer.heal("s", messages, None)

    assert prompt == await healer.renderer.render(messages, add_generation_prompt=True)
    stats = healer.pop_stats("s")
    assert stats["unhealable"] == 1
    assert stats["healed_turns"] == 0
