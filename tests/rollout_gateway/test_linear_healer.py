"""Correctness tests for LinearHealer (generate-time prefix healing).

These drive the healer + a real TrajectoryManager the same way ``BaseAdapter._run_turn``
does (heal -> generate -> commit -> record_turn), using a FakeRenderer that reproduces the
exact failure mode linear mode targets: re-rendering a previously-generated assistant
message yields *different* token ids than the ones the backend served (cosmetic
re-tokenization drift), while system/user/tool messages re-render identically.

The FakeRenderer is deliberately tokenizer-free and template-shaped: each message renders
to ``[ROLE_OPEN] + content_ids + [ROLE_CLOSE]`` and a generation prompt appends
``[ASSISTANT_OPEN]``. That is enough to exercise the two-render suffix diff, the
append-only linearity check, and the common-suffix assistant-closer probe — including the
tool-call path — without any real tokenizer.
"""

import json

import pytest

from agentcore_rl_toolkit.rollout_gateway import (
    BaseTrace,
    LinearHealer,
    TrajectoryManager,
    TurnRecord,
)

# special (template) tokens, kept clear of content and served ids
U_OPEN, U_CLOSE = 1000, 1001
A_OPEN, A_CLOSE = 1002, 1003
_CONTENT_BASE = 100_000  # content ids live here; served ids live in the 2000s


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

    def render(self, messages, *, tools=None, add_generation_prompt=True) -> list[int]:
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


def _drive_turn(mgr, healer, sid, *, messages, served_ids, response_message, logprobs=None, use_healer=True):
    """Mirror BaseAdapter._run_turn for one turn: heal -> (fake generate) -> commit ->
    record_turn. When ``use_healer`` is False, feed the raw drifted render instead (today's
    behavior) so tests can contrast the two paths on identical inputs."""
    if use_healer:
        prompt_ids = healer.heal(sid, messages, None)
    else:
        prompt_ids = healer.renderer.render(messages, tools=None, add_generation_prompt=True)
    turn = TurnRecord(
        prompt_ids=list(prompt_ids),
        output_ids=list(served_ids),
        finish_reason="stop",
        output_log_probs=list(logprobs) if logprobs is not None else None,
    )
    if use_healer:
        healer.commit(
            sid,
            fed_prompt_ids=turn.prompt_ids,
            output_ids=turn.output_ids,
            messages=messages,
            response_message=response_message,
            tools=None,
        )
    mgr.record_turn(sid, turn=turn, prompt_messages=messages, response_message=response_message)
    return prompt_ids


# ---------------------------------------------------------------------------
# The headline: renderer-driven drift forks without healing, collapses with it.
# ---------------------------------------------------------------------------


def test_unhealed_drift_forks():
    """Baseline: served ids [2001,2002] for turn 1 differ from the renderer's re-render of
    the same assistant message, and turn 2's response is long. Feeding the raw drifted
    render forks the linear rollout into two samples."""
    r = FakeRenderer()
    mgr = TrajectoryManager(fork_threshold_tokens=1)  # any drift + response -> fork
    healer = LinearHealer(r)
    _drive_turn(
        mgr, healer, "s", messages=[_u("q1")], served_ids=[2001, 2002], response_message=_a("r1"), use_healer=False
    )
    _drive_turn(
        mgr,
        healer,
        "s",
        messages=[_u("q1"), _a("r1"), _u("q2")],
        served_ids=[2003, 2004, 2005],
        response_message=_a("r2"),
        use_healer=False,
    )
    recs = mgr.get_trajectory("s", base_sample=BaseTrace(index=0), reward=1.0)
    assert len(recs) == 2  # linear rollout shattered by cosmetic drift


def test_healed_drift_stays_one_sample():
    """Same inputs, healed: turn 2 generates on the canonical served prefix, the manager
    stays CLEAN, and the whole session is one training sample with both turns trained."""
    r = FakeRenderer()
    mgr = TrajectoryManager(fork_threshold_tokens=1)
    healer = LinearHealer(r)
    _drive_turn(
        mgr, healer, "s", messages=[_u("q1")], served_ids=[2001, 2002], response_message=_a("r1"), logprobs=[-0.1, -0.2]
    )
    healed = _drive_turn(
        mgr,
        healer,
        "s",
        messages=[_u("q1"), _a("r1"), _u("q2")],
        served_ids=[2003, 2004, 2005],
        response_message=_a("r2"),
        logprobs=[-0.3, -0.4, -0.5],
    )

    recs = mgr.get_trajectory("s", base_sample=BaseTrace(index=0), reward=1.0)
    assert len(recs) == 1
    rec = recs[0]

    # canonical prefix = turn-1 fed prompt + served [2001,2002]; then the template closer
    # [A_CLOSE], the new user turn, a generation prompt, and turn-2 served ids.
    turn1_prompt = [U_OPEN, *_enc("q1"), U_CLOSE, A_OPEN]
    served_prefix = turn1_prompt + [2001, 2002]
    tail = [A_CLOSE, U_OPEN, *_enc("q2"), U_CLOSE, A_OPEN]
    assert healed == served_prefix + tail
    assert rec.token_ids == served_prefix + tail + [2003, 2004, 2005]

    # response region (leading turn-1 prompt stripped): o1 trained, glue masked, o2 trained
    o1, glue, o2 = [1, 1], [0] * len(tail), [1, 1, 1]
    assert rec.loss_mask == o1 + glue + o2
    # served logprobs preserved for both generated turns; glue zeroed
    assert rec.logprobs == [-0.1, -0.2] + [0.0] * len(tail) + [-0.3, -0.4, -0.5]
    assert healer.counters["healed_turns"] == 1
    assert healer.counters["nonlinear"] == 0


def test_healed_does_not_duplicate_closer_in_served_output():
    """When the served output already ends with the template closer (no_stop_trim keeps the
    stop token), the overlap-trim must drop it from the reinserted glue so it is not
    doubled -- otherwise the manager would see a two-closer sequence and drift."""
    r = FakeRenderer()
    mgr = TrajectoryManager(fork_threshold_tokens=1)
    healer = LinearHealer(r)
    # turn 1's served ids END with the closer token (A_CLOSE == the glue the probe finds).
    _drive_turn(mgr, healer, "s", messages=[_u("q1")], served_ids=[2001, A_CLOSE], response_message=_a("r1"))
    healed = healer.heal("s", [_u("q1"), _a("r1"), _u("q2")], None)

    turn1_prompt = [U_OPEN, *_enc("q1"), U_CLOSE, A_OPEN]
    served_prefix = turn1_prompt + [2001, A_CLOSE]
    tail = [U_OPEN, *_enc("q2"), U_CLOSE, A_OPEN]
    # closer already present at the tail of served_prefix -> glue trimmed to nothing.
    assert healed == served_prefix + tail
    assert healed.count(A_CLOSE) == 1  # not doubled


def test_healed_three_turns_one_sample():
    """Drift on every replayed assistant turn across a 3-turn chain still yields exactly
    one sample (the case that produced ~6 samples/session on the real run)."""
    r = FakeRenderer()
    mgr = TrajectoryManager(fork_threshold_tokens=1)
    healer = LinearHealer(r)
    _drive_turn(mgr, healer, "s", messages=[_u("q1")], served_ids=[2001], response_message=_a("r1"))
    _drive_turn(mgr, healer, "s", messages=[_u("q1"), _a("r1"), _u("q2")], served_ids=[2002], response_message=_a("r2"))
    _drive_turn(
        mgr,
        healer,
        "s",
        messages=[_u("q1"), _a("r1"), _u("q2"), _a("r2"), _u("q3")],
        served_ids=[2003],
        response_message=_a("r3"),
    )
    recs = mgr.get_trajectory("s", base_sample=BaseTrace(index=0), reward=1.0)
    assert len(recs) == 1
    # three trained tokens (2001, 2002, 2003), one per generated turn
    assert sum(recs[0].loss_mask) == 3
    assert healer.counters["healed_turns"] == 2


# ---------------------------------------------------------------------------
# Assistant-closer probe, including the tool-call path (the doc's caveat).
# ---------------------------------------------------------------------------


def test_close_probe_isolates_closer_text_turn():
    r = FakeRenderer()
    healer = LinearHealer(r)
    prev = [_u("q1"), _a("hello world")]
    r_prev = r.render(prev, add_generation_prompt=False)
    assert healer._assistant_close(prev, None, r_prev) == [A_CLOSE]


def test_close_probe_isolates_closer_tool_call_turn():
    """A content-less tool-call assistant message: the closer probe perturbs the tool
    arguments (not text) and still recovers exactly the template closer."""
    r = FakeRenderer()
    healer = LinearHealer(r)
    tool_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": {"q": "cats"}}}],
    }
    prev = [_u("q1"), tool_msg]
    r_prev = r.render(prev, add_generation_prompt=False)
    assert healer._assistant_close(prev, None, r_prev) == [A_CLOSE]


# ---------------------------------------------------------------------------
# Non-linear history handling.
# ---------------------------------------------------------------------------


def test_nonlinear_reset_reanchors_and_continues():
    """If a turn's render is not an append-only extension (here: a prior user message is
    edited), reset re-anchors to the incoming render and keeps healing forward."""
    r = FakeRenderer()
    healer = LinearHealer(r, on_nonlinear="reset")
    mgr = TrajectoryManager(fork_threshold_tokens=1)
    _drive_turn(mgr, healer, "s", messages=[_u("q1")], served_ids=[2001], response_message=_a("r1"))
    # turn 2 edits the prior user message ("q1" -> "Q1!"): r_prev is no longer a prefix.
    healed = healer.heal("s", [_u("Q1!"), _a("r1"), _u("q2")], None)
    assert healed == r.render([_u("Q1!"), _a("r1"), _u("q2")], add_generation_prompt=True)
    assert healer.counters["nonlinear"] == 1


def test_nonlinear_error_raises_when_configured():
    r = FakeRenderer()
    healer = LinearHealer(r, on_nonlinear="error")
    healer.commit(
        "s",
        fed_prompt_ids=r.render([_u("q1")], add_generation_prompt=True),
        output_ids=[2001],
        messages=[_u("q1")],
        response_message=_a("r1"),
        tools=None,
    )
    with pytest.raises(RuntimeError, match="append-only"):
        healer.heal("s", [_u("EDITED"), _a("r1"), _u("q2")], None)


def test_nonlinear_passthrough_disables_healing_for_session():
    r = FakeRenderer()
    healer = LinearHealer(r, on_nonlinear="passthrough")
    healer.commit(
        "s",
        fed_prompt_ids=r.render([_u("q1")], add_generation_prompt=True),
        output_ids=[2001],
        messages=[_u("q1")],
        response_message=_a("r1"),
        tools=None,
    )
    healer.heal("s", [_u("EDITED"), _a("r1"), _u("q2")], None)  # trips passthrough
    # subsequent turns are no-ops (raw render), even if they look linear again
    out = healer.heal("s", [_u("EDITED"), _a("r1"), _u("q2"), _a("r2"), _u("q3")], None)
    assert out == r.render([_u("EDITED"), _a("r1"), _u("q2"), _a("r2"), _u("q3")], add_generation_prompt=True)
    assert healer.counters["nonlinear"] == 1


def test_bad_on_nonlinear_rejected():
    with pytest.raises(ValueError):
        LinearHealer(FakeRenderer(), on_nonlinear="nope")


# ---------------------------------------------------------------------------
# Per-session counters (the metadata seam).
# ---------------------------------------------------------------------------


def test_pop_stats_is_per_session_and_isolated():
    """pop_stats returns just this sid's counters (not the run-cumulative total), keeps
    sessions isolated, and clears the sid so a second pop is empty."""
    r = FakeRenderer()
    mgr = TrajectoryManager(fork_threshold_tokens=1)
    healer = LinearHealer(r)
    # session A: two turns -> one healed turn.
    _drive_turn(mgr, healer, "a", messages=[_u("q1")], served_ids=[2001], response_message=_a("r1"))
    _drive_turn(mgr, healer, "a", messages=[_u("q1"), _a("r1"), _u("q2")], served_ids=[2002], response_message=_a("r2"))
    # session B: two turns -> one healed turn.
    _drive_turn(mgr, healer, "b", messages=[_u("p1")], served_ids=[3001], response_message=_a("s1"))
    _drive_turn(mgr, healer, "b", messages=[_u("p1"), _a("s1"), _u("p2")], served_ids=[3002], response_message=_a("s2"))

    # run-cumulative counter sees both sessions; per-sid sees only its own.
    assert healer.counters["healed_turns"] == 2
    stats_a = healer.pop_stats("a")
    assert stats_a["healed_turns"] == 1
    assert set(stats_a) == set(LinearHealer.COUNTER_KEYS)  # fixed schema
    assert healer.pop_stats("a")["healed_turns"] == 0  # cleared after pop
    assert healer.counters["healed_turns"] == 2  # run-cumulative untouched
    assert healer.pop_stats("b")["healed_turns"] == 1


def test_pop_stats_counts_nonlinear_for_session():
    r = FakeRenderer()
    healer = LinearHealer(r, on_nonlinear="reset")
    healer.commit(
        "s",
        fed_prompt_ids=r.render([_u("q1")], add_generation_prompt=True),
        output_ids=[2001],
        messages=[_u("q1")],
        response_message=_a("r1"),
        tools=None,
    )
    healer.heal("s", [_u("EDITED"), _a("r1"), _u("q2")], None)  # non-linear -> reset
    stats = healer.pop_stats("s")
    assert stats["nonlinear"] == 1


def test_drop_clears_per_sid_stats():
    r = FakeRenderer()
    mgr = TrajectoryManager(fork_threshold_tokens=1)
    healer = LinearHealer(r)
    _drive_turn(mgr, healer, "s", messages=[_u("q1")], served_ids=[2001], response_message=_a("r1"))
    _drive_turn(mgr, healer, "s", messages=[_u("q1"), _a("r1"), _u("q2")], served_ids=[2002], response_message=_a("r2"))
    healer.drop("s")
    # per-sid counters cleared -> fixed schema, all zero.
    assert healer.pop_stats("s") == dict.fromkeys(LinearHealer.COUNTER_KEYS, 0)
