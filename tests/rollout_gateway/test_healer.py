"""Correctness tests for PrefixHealer (generate-time prefix healing over the session tree).

These drive the healer + a real TrajectoryManager the same way ``BaseAdapter._run_turn``
does (heal -> generate -> record_turn), using a FakeRenderer that reproduces the exact
failure mode healed mode targets: re-rendering a previously-generated assistant message
yields *different* token ids than the ones the backend served (cosmetic re-tokenization
drift), while system/user/tool messages re-render identically.

The FakeRenderer is deliberately tokenizer-free and template-shaped: each message renders
to ``[ROLE_OPEN] + content_ids + [ROLE_CLOSE]`` and a generation prompt appends
``[ASSISTANT_OPEN]``. That is enough to exercise the two-render suffix diff, anchoring on
the tree, and the common-suffix assistant-closer probe — including the tool-call path —
without any real tokenizer.
"""

import json

import pytest

from agentcore_rl_toolkit.rollout_gateway import (
    BaseTrace,
    PrefixHealer,
    TrajectoryManager,
    TurnRecord,
)

# special (template) tokens, kept clear of content and served ids
U_OPEN, U_CLOSE = 1000, 1001
A_OPEN, A_CLOSE = 1002, 1003
_CONTENT_BASE = 100_000  # content ids live here; served ids live in the 2000s

TOOLS = [{"type": "function", "function": {"name": "calc", "parameters": {"type": "object"}}}]


def _enc(text: str) -> list[int]:
    """Deterministic per-character encoding; distinct strings -> distinct id runs, and
    appending anything changes the tail (so the closer probe's common-suffix stops at the
    template closer, not inside content)."""
    return [_CONTENT_BASE + ord(c) for c in str(text)]


class FakeRenderer:
    """Template-shaped renderer with controllable assistant drift.

    ``render`` reconstructs assistant messages from their dict (content + tool_calls),
    which is exactly what produces drift on replay: the reconstructed ids differ from the
    ids the model actually generated (which the test feeds as ``output_ids``).
    """

    def _block(self, m: dict) -> list[int]:
        role = m.get("role")
        if role == "user":
            return [U_OPEN, *_enc(m.get("content") or ""), U_CLOSE]
        if role == "assistant":
            ids = [A_OPEN]
            content = m.get("content")
            if isinstance(content, list):  # block content -> flatten text parts
                content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
            if content:
                ids += _enc(content)
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                ids += _enc(fn.get("name") or "")
                ids += _enc(json.dumps(fn.get("arguments") or {}, sort_keys=True))
            ids.append(A_CLOSE)
            return ids
        raise AssertionError(f"unexpected role {role!r}")

    async def render(self, messages, *, tools=None, add_generation_prompt=True) -> list[int]:
        out: list[int] = []
        for m in messages:
            out += self._block(m)
        if add_generation_prompt:
            out.append(A_OPEN)
        return out


def _u(text):
    return {"role": "user", "content": text}


def _a(text):
    return {"role": "assistant", "content": text}


def _prompt(text):
    """The full render of a single-user-message conversation."""
    return [U_OPEN, *_enc(text), U_CLOSE, A_OPEN]


def _tail(text):
    """The closer after an assistant turn, a new user message, and a generation prompt."""
    return [A_CLOSE, U_OPEN, *_enc(text), U_CLOSE, A_OPEN]


def _make(renderer=None, **kw):
    mgr = TrajectoryManager()
    return mgr, PrefixHealer(renderer or FakeRenderer(), mgr, **kw)


async def _drive_turn(
    mgr, healer, sid, *, messages, served_ids, response_message, tools=None, logprobs=None, use_healer=True
):
    """Mirror BaseAdapter._run_turn for one turn: heal -> (fake generate) -> record_turn.
    When ``use_healer`` is False, feed the raw drifted render instead (tree mode) so tests
    can contrast the two paths on identical inputs."""
    if use_healer:
        prompt_ids = await healer.heal(sid, messages, tools)
    else:
        prompt_ids = await healer.renderer.render(messages, tools=tools, add_generation_prompt=True)
    turn = TurnRecord(
        prompt_ids=list(prompt_ids),
        output_ids=list(served_ids),
        finish_reason="stop",
        output_log_probs=list(logprobs) if logprobs is not None else None,
    )
    mgr.record_turn(sid, turn=turn, prompt_messages=messages, response_message=response_message, tools=tools)
    return prompt_ids


def _records(mgr, sid="s"):
    return mgr.get_trajectory(sid, base_sample=BaseTrace(index=0), reward=1.0)


# ---------------------------------------------------------------------------
# The headline: renderer-driven drift forks without healing, collapses with it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unhealed_drift_forks():
    """Baseline: served ids [2001,2002] for turn 1 differ from the renderer's re-render of
    the same assistant message. Feeding the raw drifted render forks the append-only
    rollout into two samples."""
    mgr, healer = _make()
    await _drive_turn(
        mgr, healer, "s", messages=[_u("q1")], served_ids=[2001, 2002], response_message=_a("r1"), use_healer=False
    )
    await _drive_turn(
        mgr,
        healer,
        "s",
        messages=[_u("q1"), _a("r1"), _u("q2")],
        served_ids=[2003, 2004, 2005],
        response_message=_a("r2"),
        use_healer=False,
    )
    assert len(_records(mgr)) == 2  # append-only rollout shattered by cosmetic drift


@pytest.mark.asyncio
async def test_healed_drift_stays_one_sample():
    """Same inputs, healed: turn 2 generates on the served prefix, the manager stays
    CLEAN, and the whole session is one training sample with both turns trained."""
    mgr, healer = _make()
    await _drive_turn(
        mgr, healer, "s", messages=[_u("q1")], served_ids=[2001, 2002], response_message=_a("r1"), logprobs=[-0.1, -0.2]
    )
    healed = await _drive_turn(
        mgr,
        healer,
        "s",
        messages=[_u("q1"), _a("r1"), _u("q2")],
        served_ids=[2003, 2004, 2005],
        response_message=_a("r2"),
        logprobs=[-0.3, -0.4, -0.5],
    )

    recs = _records(mgr)
    assert len(recs) == 1
    rec = recs[0]

    # served prefix = turn-1 fed prompt + served [2001,2002]; then the template closer
    # [A_CLOSE], the new user turn, a generation prompt, and turn-2 served ids.
    served_prefix = _prompt("q1") + [2001, 2002]
    tail = _tail("q2")
    assert healed == served_prefix + tail
    assert rec.token_ids == served_prefix + tail + [2003, 2004, 2005]

    # response region (leading turn-1 prompt stripped): o1 trained, glue masked, o2 trained
    o1, glue, o2 = [1, 1], [0] * len(tail), [1, 1, 1]
    assert rec.loss_mask == o1 + glue + o2
    # served logprobs preserved for both generated turns; glue zeroed
    assert rec.logprobs == [-0.1, -0.2] + [0.0] * len(tail) + [-0.3, -0.4, -0.5]
    assert healer.counters["healed_turns"] == 1
    assert healer.counters["branches"] == 0


@pytest.mark.asyncio
async def test_healed_does_not_duplicate_closer_in_served_output():
    """When the served output already ends with the template closer (no_stop_trim keeps the
    stop token), the overlap-trim must drop it from the reinserted glue so it is not
    doubled -- otherwise the manager would see a two-closer sequence and drift."""
    mgr, healer = _make()
    # turn 1's served ids END with the closer token (A_CLOSE == the glue the probe finds).
    await _drive_turn(mgr, healer, "s", messages=[_u("q1")], served_ids=[2001, A_CLOSE], response_message=_a("r1"))
    healed = await healer.heal("s", [_u("q1"), _a("r1"), _u("q2")], None)

    served_prefix = _prompt("q1") + [2001, A_CLOSE]
    tail = [U_OPEN, *_enc("q2"), U_CLOSE, A_OPEN]
    # closer already present at the tail of served_prefix -> glue trimmed to nothing.
    assert healed == served_prefix + tail
    assert healed.count(A_CLOSE) == 1  # not doubled


@pytest.mark.asyncio
async def test_healed_three_turns_one_sample():
    """Drift on every replayed assistant turn across a 3-turn chain still yields exactly
    one sample, each turn extending the previous turn's served ids."""
    mgr, healer = _make()
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
    recs = _records(mgr)
    assert len(recs) == 1
    assert recs[0].token_ids == p3 + [2003]
    assert sum(recs[0].loss_mask) == 3  # one trained token per generated turn
    assert healer.pop_stats("s")["healed_turns"] == 2


# ---------------------------------------------------------------------------
# Message-level anchoring: a re-keyed tool-call replay is dict-equal but renders
# to different tokens under an order-sensitive template. It must stay healed.
# ---------------------------------------------------------------------------


class OrderSensitiveRenderer(FakeRenderer):
    """Renders tool-call arguments in INSERTION order (like a real chat template), so a
    replayed assistant tool-call with re-keyed arguments produces different tokens even
    though the message dict is ``==`` (the wire form is sorted, the model's emission order
    is not)."""

    def _block(self, m: dict) -> list[int]:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            ids = [A_OPEN]
            content = m.get("content")
            if content:
                ids += _enc(content)
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                ids += _enc(fn.get("name") or "")
                # sort_keys=False -> argument key order changes the token render
                ids += _enc(json.dumps(fn.get("arguments") or {}, sort_keys=False))
            ids.append(A_CLOSE)
            return ids
        return super()._block(m)


def _a_tool(name, arguments, content=""):
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
    }


@pytest.mark.asyncio
async def test_reordered_tool_call_args_stay_healed():
    """A replayed tool-call assistant whose arguments are re-keyed (dict-equal, but a
    different token render under an order-sensitive template) must still anchor -- the
    drift a token-prefix check would wrongly treat as a branch."""
    r = OrderSensitiveRenderer()
    mgr, healer = _make(r)

    emitted = {"path": "/f.py", "text": "x"}  # the model's emission order
    reordered = {"text": "x", "path": "/f.py"}  # the client's sorted-wire round-trip
    assert _a_tool("edit", emitted) == _a_tool("edit", reordered)  # manager sees them equal
    # ... but the order-sensitive renderer does not:
    assert await r.render([_a_tool("edit", emitted)]) != await r.render([_a_tool("edit", reordered)])

    await _drive_turn(
        mgr, healer, "s", messages=[_u("q1")], served_ids=[2001, 2002], response_message=_a_tool("edit", emitted)
    )
    healed = await _drive_turn(
        mgr,
        healer,
        "s",
        messages=[_u("q1"), _a_tool("edit", reordered), _u("obs")],
        served_ids=[2003],
        response_message=_a("done"),
    )

    # the fed prompt reuses the served prefix (the drifted replay order never appears),
    # and the whole session stays one training sample.
    assert healer.counters["branches"] == 0
    assert healer.counters["healed_turns"] == 1
    served_prefix = _prompt("q1") + [2001, 2002]
    assert healed[: len(served_prefix)] == served_prefix
    assert len(_records(mgr)) == 1


# ---------------------------------------------------------------------------
# Assistant-closer probe, including the tool-call path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_probe_isolates_closer_text_turn():
    r = FakeRenderer()
    _, healer = _make(r)
    prev = [_u("q1"), _a("hello world")]
    r_prev = await r.render(prev, add_generation_prompt=False)
    assert await healer._assistant_close(prev, None, r_prev) == [A_CLOSE]


@pytest.mark.asyncio
async def test_close_probe_isolates_closer_tool_call_turn():
    """A content-less tool-call assistant message: the closer probe uses text bodies and
    still recovers exactly the template closer."""
    r = FakeRenderer()
    _, healer = _make(r)
    tool_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": {"q": "cats"}}}],
    }
    prev = [_u("q1"), tool_msg]
    r_prev = await r.render(prev, add_generation_prompt=False)
    assert await healer._assistant_close(prev, None, r_prev) == [A_CLOSE]


# ---------------------------------------------------------------------------
# Branches: requests that do not extend the session's latest turn.
# ---------------------------------------------------------------------------


async def _retry_session(*, use_healer):
    """Turn 1, turn 2, then a retry of turn 2 with the same messages: a sibling branch."""
    mgr, healer = _make()
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
async def test_retry_branch_is_healed_on_its_anchor():
    """A retried turn anchors on the same generated turn as the first attempt, so both
    siblings extend turn 1's served ids. Each leaf is one CLEAN sample; turn 1 is trained
    once and re-emitted as context on the second leaf."""
    mgr, healer, first, retry = await _retry_session(use_healer=True)

    assert first == retry == _prompt("q1") + [2001] + _tail("q2")
    recs = _records(mgr)
    assert [r.token_ids for r in recs] == [first + [2002], retry + [2003]]
    assert sum(r.loss_mask.count(1) for r in recs) == 3  # turn 1 once, each attempt once
    stats = healer.pop_stats("s")
    assert stats["healed_turns"] == 2
    assert stats["branches"] == 1  # the retry


@pytest.mark.asyncio
async def test_retry_branch_forks_without_healing():
    """Baseline for the test above: feeding the drifted full render forks each branch."""
    mgr, _, _, _ = await _retry_session(use_healer=False)
    assert len(_records(mgr)) == 3


@pytest.mark.asyncio
async def test_continuing_an_older_branch_is_healed():
    """After a retry, the agent continues the FIRST attempt. The request anchors on that
    attempt even though it is not the latest turn, so the first branch stays one sample."""
    mgr, healer, first, _ = await _retry_session(use_healer=True)
    cont = await _drive_turn(
        mgr,
        healer,
        "s",
        messages=[_u("q1"), _a("r1"), _u("q2"), _a("r2"), _u("q3")],
        served_ids=[2004],
        response_message=_a("r3"),
    )

    assert cont == first + [2002] + _tail("q3")
    recs = _records(mgr)
    assert len(recs) == 2  # one per branch
    assert healer.pop_stats("s")["branches"] == 2  # the retry, then the return to attempt 1


@pytest.mark.asyncio
async def test_edit_branches_from_earlier_turn():
    """A request that edits a later message anchors on the deepest generated turn its
    messages still match, not on the latest turn."""
    mgr, healer = _make()
    await _drive_turn(mgr, healer, "s", messages=[_u("q1")], served_ids=[2001], response_message=_a("r1"))
    await _drive_turn(
        mgr, healer, "s", messages=[_u("q1"), _a("r1"), _u("q2")], served_ids=[2002], response_message=_a("r2")
    )
    edited = await _drive_turn(
        mgr, healer, "s", messages=[_u("q1"), _a("r1"), _u("q2 edited")], served_ids=[2003], response_message=_a("x")
    )

    assert edited == _prompt("q1") + [2001] + _tail("q2 edited")
    assert len(_records(mgr)) == 2
    assert healer.pop_stats("s")["branches"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("edited", ["user", "assistant"])
async def test_edit_with_no_generated_anchor_is_rendered_in_full(edited):
    """Editing the first user message, or the only generated assistant turn, leaves no
    generated turn to anchor on: the request is rendered in full and both generations
    are preserved."""
    mgr, healer = _make()
    await _drive_turn(mgr, healer, "s", messages=[_u("q1")], served_ids=[2001], response_message=_a("r1"))
    messages = [
        _u("Q1!" if edited == "user" else "q1"),
        _a("R1!" if edited == "assistant" else "r1"),
        _u("q2"),
    ]
    prompt = await _drive_turn(mgr, healer, "s", messages=messages, served_ids=[2002], response_message=_a("r2"))

    assert prompt == await healer.renderer.render(messages, add_generation_prompt=True)
    assert healer.counters["branches"] == 1
    recs = _records(mgr)
    assert [rec.token_ids[-1] for rec in recs] == [2001, 2002]
    assert [rec.loss_mask for rec in recs] == [[1], [1]]


@pytest.mark.asyncio
async def test_new_root_is_rendered_in_full():
    """A request matching no generated turn (e.g. a sub-agent with its own first message)
    is rendered in full. The first turn of a session is not a branch; the new root is."""
    mgr, healer = _make()
    await _drive_turn(mgr, healer, "s", messages=[_u("q1")], served_ids=[2001], response_message=_a("r1"))
    assert healer.pop_stats("s")["branches"] == 0
    sub = await _drive_turn(mgr, healer, "s", messages=[_u("sub task")], served_ids=[2002], response_message=_a("s1"))

    assert sub == _prompt("sub task")
    assert healer.pop_stats("s")["branches"] == 1


@pytest.mark.asyncio
async def test_tools_change_is_rendered_in_full():
    """A changed tools schema is a branch even when the messages extend cleanly; the
    anchor's served ids were generated under other tools, so the request is rendered in full."""
    mgr, healer = _make()
    await _drive_turn(mgr, healer, "s", messages=[_u("q1")], served_ids=[2001], response_message=_a("r1"), tools=TOOLS)
    messages = [_u("q1"), _a("r1"), _u("q2")]
    prompt = await healer.heal("s", messages, None)

    assert prompt == await healer.renderer.render(messages, add_generation_prompt=True)
    stats = healer.pop_stats("s")
    assert stats["branches"] == 1
    assert stats["healed_turns"] == 0


@pytest.mark.asyncio
async def test_branch_error_raises_when_configured():
    mgr, healer = _make(on_branch="error")
    await _drive_turn(mgr, healer, "s", messages=[_u("q1")], served_ids=[2001], response_message=_a("r1"))
    # extending the latest turn is fine
    await _drive_turn(
        mgr, healer, "s", messages=[_u("q1"), _a("r1"), _u("q2")], served_ids=[2002], response_message=_a("r2")
    )
    with pytest.raises(RuntimeError, match="branched"):
        await healer.heal("s", [_u("q1"), _a("r1"), _u("EDITED")], None)


def test_bad_on_branch_rejected():
    with pytest.raises(ValueError):
        PrefixHealer(FakeRenderer(), TrajectoryManager(), on_branch="reset")


class PrefixChangingRenderer(FakeRenderer):
    """A template whose earlier blocks change when messages are appended (here: a leading
    message count), so no splice onto the served ids is valid."""

    async def render(self, messages, *, tools=None, add_generation_prompt=True):
        ids = await super().render(messages, tools=tools, add_generation_prompt=add_generation_prompt)
        return [len(messages), *ids]


@pytest.mark.asyncio
async def test_unhealable_template_is_rendered_in_full():
    mgr, healer = _make(PrefixChangingRenderer())
    await _drive_turn(mgr, healer, "s", messages=[_u("q1")], served_ids=[2001], response_message=_a("r1"))
    messages = [_u("q1"), _a("r1"), _u("q2")]
    prompt = await healer.heal("s", messages, None)

    assert prompt == await healer.renderer.render(messages, add_generation_prompt=True)
    stats = healer.pop_stats("s")
    assert stats["unhealable"] == 1
    assert stats["healed_turns"] == 0
    assert stats["branches"] == 0  # the history is append-only; the template is the issue


# ---------------------------------------------------------------------------
# Per-session counters (the metadata seam).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pop_stats_is_per_session_and_isolated():
    """pop_stats returns just this sid's counters (not the run-cumulative total), keeps
    sessions isolated, and clears the sid so a second pop is empty."""
    mgr, healer = _make()
    for sid, (q, r) in {"a": ("q", "r"), "b": ("p", "s")}.items():
        await _drive_turn(mgr, healer, sid, messages=[_u(q + "1")], served_ids=[2001], response_message=_a(r + "1"))
        await _drive_turn(
            mgr,
            healer,
            sid,
            messages=[_u(q + "1"), _a(r + "1"), _u(q + "2")],
            served_ids=[2002],
            response_message=_a(r + "2"),
        )

    # run-cumulative counter sees both sessions; per-sid sees only its own.
    assert healer.counters["healed_turns"] == 2
    stats_a = healer.pop_stats("a")
    assert stats_a["healed_turns"] == 1
    assert set(stats_a) == set(PrefixHealer.COUNTER_KEYS)  # fixed schema
    assert healer.pop_stats("a")["healed_turns"] == 0  # cleared after pop
    assert healer.counters["healed_turns"] == 2  # run-cumulative untouched
    assert healer.pop_stats("b")["healed_turns"] == 1


@pytest.mark.asyncio
async def test_drop_clears_per_sid_stats():
    mgr, healer = _make()
    await _drive_turn(mgr, healer, "s", messages=[_u("q1")], served_ids=[2001], response_message=_a("r1"))
    await _drive_turn(
        mgr, healer, "s", messages=[_u("q1"), _a("r1"), _u("q2")], served_ids=[2002], response_message=_a("r2")
    )
    healer.drop("s")
    assert healer.pop_stats("s") == dict.fromkeys(PrefixHealer.COUNTER_KEYS, 0)
