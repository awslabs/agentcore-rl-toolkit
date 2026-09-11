"""Generate-time prefix healing for strictly-linear agent histories.

This is the engine behind the gateway's ``history_mode="linear"``. It targets agents whose conversation
is **append-only** and where cosmetic re-tokenization drift of a previously-generated
assistant turn is considered insignificant (OOD-negligible).

The problem it removes: each turn the client replays the whole conversation as messages;
re-rendering a prior assistant message from its parsed dict rarely reproduces the exact
token ids the model actually generated (whitespace, tool-call JSON spacing, reasoning
re-keying). The :class:`~.trajectory.TrajectoryManager` then sees that drift and FORKs the
single rollout into multiple training samples (or REALIGNs and drops a turn's signal).

The fix: because the gateway owns the tokens handed to the backend, it substitutes the
exact ids it already served for the earlier turns **before** generation, so the model
always generates on the canonical, drift-free context. The recorded prompt is then an
exact prefix extension of what the manager holds, so the manager stays permanently CLEAN
— one sample per session, every generated turn trained on its true multi-turn context.

Per turn the healer holds, per sid:

* ``served_prefix`` — the canonical token ids of the conversation through the end of the
  last turn's *generated content* (i.e. what we fed to generate last turn, plus the ids
  the backend actually served). This is exactly what the manager holds after that turn.
* ``prev_messages`` / ``prev_tools`` — the message list (including the generated assistant)
  and tools schema last rendered, used to re-derive the drifted skeleton.

Healing ``messages`` for the next turn (:meth:`heal`):

1. Linearity check at the **message level**: ``prev_messages`` must be a prefix of the
   incoming ``messages`` under dict equality (``==``, which is order-independent), and the
   tools schema must be unchanged. This is the SAME notion the :class:`TrajectoryManager`
   uses to match replayed history, so a prior turn the client re-serialized cosmetically
   (e.g. tool-call ``arguments`` re-keyed, or JSON re-spaced — the wire form is emitted
   ``sort_keys=True`` while the manager kept the model's emission order) still compares
   equal and stays linear. Only a genuine edit/drop/branch/compaction (or a tools change)
   is non-linear — handled per ``on_nonlinear``. ``new = messages[len(prev_messages):]``
   is the genuinely-new observation(s).
2. ``r_prev = render(prev_messages, add_generation_prompt=False)`` — the prior conversation
   rendered from the gateway's OWN stored messages (``= served_prompt' + O_n' + close_n``).
3. ``r_ext = render(prev_messages + new, add_generation_prompt=True)`` — the same stored
   prefix extended by the new observation(s). ``r_ext`` starts with ``r_prev`` **by
   construction** (same prefix list, same tools), so the client's re-serialization of prior
   turns never enters the fed tokens.
4. ``close_n`` — the template's assistant-closer after the last turn, extracted exactly by
   a common-suffix probe (:func:`_assistant_close`), independent of template internals.
5. ``healed = served_prefix + close_n + r_ext[len(r_prev):]`` — canonical prefix, the
   template close, then the genuinely-new observation + generation-prompt tail.

:meth:`commit` (after a successful turn) advances ``served_prefix`` to
``healed + output_ids`` and ``prev_messages`` to ``messages + [response_message]``.

The healer is tokenizer-free and torch-free: it only calls the injected ``Renderer``.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import Counter

from .render import Renderer

logger = logging.getLogger(__name__)

__all__ = ["LinearHealer"]

# two clearly-distinct assistant bodies used to probe the template's between-message glue
# (see _assistant_close). They must tokenize to different trailing tokens so the common
# suffix of the two renders is exactly the closer, nothing of the body.
_PROBE_A = "linprobe body alpha 00000"
_PROBE_B = "linprobe body omega 99999"


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def _common_suffix(a: list[int], b: list[int]) -> list[int]:
    """The longest common suffix of two id lists (as a fresh list)."""
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[-1 - i] == b[-1 - i]:
        i += 1
    return a[len(a) - i :] if i else []


@dataclasses.dataclass
class _State:
    served_prefix: list[int]
    prev_messages: list[dict]
    prev_tools: list[dict] | None


class LinearHealer:
    """Per-sid generate-time prefix healing for linear histories.

    ``on_nonlinear`` picks the behaviour when the message-level linearity assumption breaks
    (the stored history is not a dict-equality prefix of the incoming history, or the tools
    schema changed):

    * ``"reset"`` (default) — re-anchor to the incoming render and keep healing forward,
      treating the jump as the start of a fresh linear segment. Cheapest; correct when the
      jump is a benign, agent-controlled compaction. The single jump turn conditions on the
      client's (drifted-but-linear-from-here) render.
    * ``"error"`` — raise, failing the rollout, for runs that want a hard guarantee.
    * ``"passthrough"`` — stop healing this sid and hand back the raw incoming render, so
      the caller records the drifted prompt and the manager's tree/FORK path takes over.
    """

    def __init__(self, renderer: Renderer, *, on_nonlinear: str = "reset") -> None:
        if on_nonlinear not in ("reset", "error", "passthrough"):
            raise ValueError(f"on_nonlinear must be reset|error|passthrough, got {on_nonlinear!r}")
        self.renderer = renderer
        self.on_nonlinear = on_nonlinear
        self._state: dict[str, _State] = {}
        # sids we have stopped healing (passthrough after a non-linear jump)
        self._disabled: set[str] = set()
        # run-cumulative totals across every sid this healer instance has seen
        self.counters: Counter[str] = Counter()
        # per-sid totals, popped at session end so they can ride out on the record's
        # metadata (a session-level view, distinct from the run-cumulative ``counters``)
        self._per_sid: dict[str, Counter[str]] = {}

    def _bump(self, sid: str, key: str, n: int = 1) -> None:
        self.counters[key] += n
        self._per_sid.setdefault(sid, Counter())[key] += n

    # -- public -------------------------------------------------------------

    def heal(self, sid: str, messages: list[dict], tools: list[dict] | None) -> list[int]:
        """Return the token ids to feed the backend for this turn.

        For the first turn of a sid (or a passthrough sid) this is just the plain render;
        otherwise the drift-healed canonical prompt described in the module docstring.
        ``commit`` must be called afterwards on the turns that are actually recorded.
        """
        # canonical tokens = assistant message in token space as sampled from the llm
        # drifted tokens = same message but with a different token-space representation
        # some messages render into canonical tokens, some into drifted
        # given canonical tokens, we can't compute a message that will render into canonical tokens
        # but if two messages are equal, then we can use canonical tokens as context for the llm

        st = self._state.get(sid)
        if st is None or sid in self._disabled:
            # first turn, or healing disabled for this sid: feed the plain render
            return self.renderer.render(messages, tools=tools, add_generation_prompt=True)

        # Linearity is decided in MESSAGE space, not token space. st.prev_messages is this
        # session's history before this request (including the last assistant message). The
        # history is linear iff those messages are a prefix of the incoming messages under
        # dict equality -- the same order-independent match the TrajectoryManager uses -- and
        # the tools schema is unchanged. A prior assistant turn the client re-serialized
        # cosmetically (tool-call arguments re-keyed, or JSON re-spaced: the wire form is
        # emitted sort_keys=True while the manager keeps the model's emission order) still
        # compares equal, so it stays linear instead of tripping a token-level prefix check.
        new = self._linear_delta(st, messages, tools)
        if new is None:
            # genuine edit/drop/branch/compaction (or a tools change): re-anchor
            r_full = self.renderer.render(messages, tools=tools, add_generation_prompt=True)
            return self._handle_nonlinear(sid, r_full)

        # r_prev is the token-space render of the gateway's OWN stored history. r_ext extends
        # that same stored prefix by only the genuinely-new (non-assistant, drift-free)
        # messages, so r_ext starts with r_prev by construction (identical prefix list and
        # tools) -- the client's re-serialized prior turns never enter the fed tokens.
        r_prev = self.renderer.render(st.prev_messages, tools=st.prev_tools, add_generation_prompt=False)
        r_ext = self.renderer.render(st.prev_messages + new, tools=st.prev_tools, add_generation_prompt=True)
        if r_ext[: len(r_prev)] != r_prev:
            # template glue for the prior turns depends on the appended messages, so we
            # cannot splice the canonical prefix safely; re-anchor rather than emit garbage.
            return self._handle_nonlinear(sid, r_ext)

        # close is the token-space marker of the end of the assistant turn
        close = self._assistant_close(st.prev_messages, st.prev_tools, r_prev)
        if close is None:
            # couldn't isolate the assistant-closer (e.g. a template whose glue depends on
            # message type); don't emit a subtly-wrong sequence — fall back.
            self._bump(sid, "close_unresolved")
            return self._handle_nonlinear(sid, r_ext)

        # don't duplicate close tokens the model already emitted at the end of its served
        # output (e.g. a trailing stop token kept by no_stop_trim): drop the prefix of
        # `close` that the served prefix already ends with.
        close = self._trim_overlap(st.served_prefix, close)

        # tail is the token representation of the new messages in the incoming request
        # these are not assistant messages, so they are masked and drift-free
        tail = r_ext[len(r_prev) :]  # new observation messages + generation prompt

        self._bump(sid, "healed_turns")
        self._bump(sid, "healed_prefix_tokens", len(st.served_prefix) + len(close))

        # st.served_prefix is the accumulation of rendered non-assistant messages
        # and canonical tokens of assistant messsages, therefore the concatenated
        # sequence contains zero drifted tokens
        return list(st.served_prefix) + close + tail

    @staticmethod
    def _linear_delta(st: _State, messages: list[dict], tools: list[dict] | None) -> list[dict] | None:
        """The genuinely-new messages appended since the last turn, or ``None`` if the
        incoming history is not a linear extension of the stored history.

        Linear iff ``st.prev_messages`` is a prefix of ``messages`` under dict equality
        (order-independent, matching :class:`TrajectoryManager`) AND the tools schema is
        unchanged. Returns ``messages[len(prev_messages):]`` when linear (possibly empty:
        a bare re-generation of the same turn).
        """
        prev = st.prev_messages
        k = len(prev)
        if len(messages) < k or messages[:k] != prev:
            return None
        if (list(tools) if tools else None) != st.prev_tools:
            return None
        return messages[k:]

    def commit(
        self,
        sid: str,
        *,
        fed_prompt_ids: list[int],
        output_ids: list[int],
        messages: list[dict],
        response_message: dict | None,
        tools: list[dict] | None,
    ) -> None:
        """Advance per-sid state after a turn that was actually recorded.

        ``fed_prompt_ids`` is what :meth:`heal` returned (and what the backend generated
        against); ``output_ids`` is what the backend served.
        """
        if sid in self._disabled:
            return
        asst = response_message if response_message is not None else {"role": "assistant", "content": ""}
        self._state[sid] = _State(
            served_prefix=list(fed_prompt_ids) + list(output_ids),
            prev_messages=list(messages) + [asst],
            prev_tools=list(tools) if tools else None,
        )

    #: every counter key this healer can emit; ``pop_stats`` returns all of them (with
    #: zeros for counters that never fired) so per-session stats have a stable schema.
    COUNTER_KEYS = ("healed_turns", "healed_prefix_tokens", "nonlinear", "close_unresolved")

    def pop_stats(self, sid: str) -> dict[str, int]:
        """Remove and return this sid's healing counters as a fixed-schema dict.

        Always includes every key in :attr:`COUNTER_KEYS` (counters that never fired are
        zeroed), so a downstream per-session metric reduction sees the same columns for
        every session. Called at session teardown so the per-session view can be attached
        to the session's record metadata; the run-cumulative ``counters`` is untouched.
        """
        c = self._per_sid.pop(sid, None) or Counter()
        return {k: c[k] for k in self.COUNTER_KEYS}

    def drop(self, sid: str) -> None:
        self._state.pop(sid, None)
        self._disabled.discard(sid)
        self._per_sid.pop(sid, None)

    # -- internals ----------------------------------------------------------

    def _handle_nonlinear(self, sid: str, r_full: list[int]) -> list[int]:
        self._bump(sid, "nonlinear")
        if self.on_nonlinear == "error":
            raise RuntimeError(
                f"linear history mode: sid {sid!r} broke the append-only assumption "
                "(stored history is not a message-level prefix of the incoming history)"
            )
        if self.on_nonlinear == "passthrough":
            self._disabled.add(sid)
            self._state.pop(sid, None)
            logger.warning("linear healer: sid=%s non-linear jump; disabling healing for this session", sid)
            return r_full
        # reset: re-anchor to the incoming render as a fresh linear segment. commit() will
        # overwrite _state; here we just clear it so this turn is treated first-turn-like.
        # (No separate reset counter: with on_nonlinear="reset" every ``nonlinear`` jump is
        # re-anchored, so it would just duplicate ``nonlinear``; the mode is known config.)
        self._state.pop(sid, None)
        logger.info("linear healer: sid=%s non-linear jump; re-anchoring", sid)
        return r_full

    def _assistant_close(
        self, prev_messages: list[dict], tools: list[dict] | None, r_prev: list[int]
    ) -> list[int] | None:
        """The template's between-message glue that follows the last assistant message
        (e.g. Qwen's ``<|im_end|>\\n``), returned as token ids (possibly empty).

        The glue is **type-independent** — templates append the same tokens after a text
        or a tool-call assistant message — and it is what re-tokenization drift never
        touches. So we probe it with two *text* assistant bodies (no tool-call wrapper to
        share a suffix with) appended after the same prior turns, and take the common
        suffix of the two renders: the bodies differ in every trailing token, the glue is
        identical.

        Returns ``None`` if the probed glue is not actually a suffix of the real ``r_prev``
        — which happens iff this template's glue depends on message type (so the text probe
        doesn't describe the actual tool-call turn). The caller then falls back rather than
        emit a wrong boundary.
        """
        before = prev_messages[:-1]
        ra = self.renderer.render(
            before + [{"role": "assistant", "content": _PROBE_A}], tools=tools, add_generation_prompt=False
        )
        rb = self.renderer.render(
            before + [{"role": "assistant", "content": _PROBE_B}], tools=tools, add_generation_prompt=False
        )
        close = _common_suffix(ra, rb)
        if close and r_prev[-len(close) :] != close:
            return None  # glue is message-type-dependent for this template; don't guess
        return close

    @staticmethod
    def _trim_overlap(served_prefix: list[int], close: list[int]) -> list[int]:
        """Drop the longest prefix of ``close`` that ``served_prefix`` already ends with, so
        a closer the model emitted in its served output isn't duplicated."""
        for k in range(min(len(close), len(served_prefix)), 0, -1):
            if served_prefix[-k:] == close[:k]:
                return close[k:]
        return close
