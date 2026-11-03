"""Generate-time prefix healing over the session tree.

This is the engine behind the gateway's ``history_mode="linear"``. It targets harnesses
whose cosmetic re-tokenization drift of a previously-generated assistant turn is
considered insignificant (OOD-negligible).

The problem it removes: each turn the client replays the whole conversation as messages;
re-rendering a prior assistant message from its parsed dict rarely reproduces the exact
token ids the model actually generated (whitespace, tool-call JSON spacing, reasoning
re-keying). The :class:`~.trajectory.TrajectoryManager` then sees that drift and FORKs the
rollout into multiple training samples.

The fix: because the gateway owns the tokens handed to the backend, it substitutes the
exact ids it already served for the earlier turns **before** generation, so the model
always generates on the canonical, drift-free context. Each recorded prompt is then an
exact prefix extension of the turn it continues, so every branch of the manager's tree
stays CLEAN: an append-only session is one sample, and retries, edits, parallel branches,
and sub-agent forks that reuse the session id add one sample per branch.

Healing ``messages`` for the next turn (``await heal(...)``):

1. Anchor at the **message level**: :meth:`TrajectoryManager.heal_anchor` returns the
   deepest generated turn whose root path is a prefix of ``messages`` under the dict
   equality ``record_turn`` mounts by, so a prior turn the client re-serialized
   cosmetically (tool-call ``arguments`` re-keyed, JSON re-spaced) still matches. The
   anchor carries the turn's served ids (``prompt_ids + output_ids``), its root-path
   messages, and its tools schema. ``new = messages[len(anchor.messages):]`` is the
   genuinely-new observation(s).
2. If the renderer supports ``render_delta``, render the stored history and its extension
   to text, probe the assistant closer in text, and encode only the closer and the new
   tail. If that path is unavailable or declines, use steps 3–5 below.
3. ``r_prev = render(anchor.messages, add_generation_prompt=False)`` — the prior
   conversation rendered from the gateway's OWN stored messages.
4. ``r_ext = render(anchor.messages + new, add_generation_prompt=True)`` — the same stored
   prefix extended by the new observation(s). Check that ``r_ext`` starts with ``r_prev``;
   the client's re-serialization of prior turns never enters the fed tokens.
5. ``close`` — the template's assistant-closer after the anchor turn, extracted by a
   common-suffix probe (:meth:`LinearHealer._assistant_close`).
6. Splice the closer and new tail onto the anchor's served ids, omitting any closer tokens
   already present at their end.

A request with no anchor (first turn, or a new root such as a sub-agent with its own first
message), a changed tools schema, or a template the splice cannot handle is rendered in
full, and the manager handles any drift as in tree mode.

The healer keeps no per-session history: the anchor comes from the manager's tree, which
``record_turn`` advances. It is tokenizer-free and torch-free; it only calls the injected
``Renderer``.
"""

from __future__ import annotations

import logging
from collections import Counter

from .render import _PROBE_A, _PROBE_B, Renderer, _common_suffix
from .trajectory import HealAnchor, TrajectoryManager

logger = logging.getLogger(__name__)

__all__ = ["LinearHealer", "make_healer"]

HISTORY_MODES = ("tree", "linear")


def make_healer(
    history_mode: str, renderer: Renderer, manager: TrajectoryManager, on_nonlinear: str = "reset"
) -> LinearHealer | None:
    """The healer for ``history_mode``: a ``LinearHealer`` for ``"linear"``, ``None`` for
    ``"tree"``."""
    if history_mode not in HISTORY_MODES:
        raise ValueError(f"history_mode must be one of {HISTORY_MODES}, got {history_mode!r}")
    return LinearHealer(renderer, manager, on_nonlinear=on_nonlinear) if history_mode == "linear" else None


class LinearHealer:
    """Per-branch generate-time prefix healing anchored on the manager's tree.

    A request is *non-linear* when it does not extend the session's latest generated
    turn: a retry, an edit or drop of a prior message, a parallel branch, a sub-agent with
    its own first message, a compaction, or a tools change. ``on_nonlinear`` picks the
    behavior:

    * ``"reset"`` (default) — heal from the deepest generated turn the request still
      matches, or render in full when none matches or its tools schema differs.
    * ``"error"`` — raise, failing the rollout, for runs that require an append-only
      history.
    * ``"passthrough"`` — stop healing this sid and hand back the full render, so the
      manager's tree/FORK path handles the rest of the session.

    Every non-linear request is counted under ``nonlinear``.
    """

    #: every counter key this healer can emit; ``pop_stats`` returns all of them (with
    #: zeros for counters that never fired) so per-session stats have a stable schema.
    COUNTER_KEYS = ("healed_turns", "healed_prefix_tokens", "nonlinear", "close_unresolved", "unhealable")

    def __init__(self, renderer: Renderer, manager: TrajectoryManager, *, on_nonlinear: str = "reset") -> None:
        if on_nonlinear not in ("reset", "error", "passthrough"):
            raise ValueError(f"on_nonlinear must be reset|error|passthrough, got {on_nonlinear!r}")
        self.renderer = renderer
        self.manager = manager
        self.on_nonlinear = on_nonlinear
        # sids we have stopped healing (passthrough after a non-linear request)
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

    async def heal(self, sid: str, messages: list[dict], tools: list[dict] | None) -> list[int]:
        """Return the token ids to feed the backend for this turn: the drift-healed prompt
        described in the module docstring, or the full render when there is nothing to
        heal from."""
        tools = list(tools) if tools else None
        if sid in self._disabled:
            return await self._render_full(messages, tools)

        anchor = self.manager.heal_anchor(sid, messages)
        latest = self.manager.turn_count(sid)
        extends_latest = anchor is not None and anchor.turn_index == latest and anchor.tools == tools
        if latest and not extends_latest:
            self._bump(sid, "nonlinear")
            if self.on_nonlinear == "error":
                raise RuntimeError(
                    f"linear history mode: sid {sid!r} broke the append-only assumption "
                    "(the request does not extend the session's latest turn)"
                )
            if self.on_nonlinear == "passthrough":
                self._disabled.add(sid)
                logger.warning("linear healer: sid=%s non-linear request; disabling healing for this session", sid)
                return await self._render_full(messages, tools)

        if anchor is None or anchor.tools != tools:
            return await self._render_full(messages, tools)
        return await self._extend(sid, anchor, messages[len(anchor.messages) :])

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
        self._disabled.discard(sid)
        self._per_sid.pop(sid, None)

    # -- internals ----------------------------------------------------------

    async def _render_full(self, messages: list[dict], tools: list[dict] | None) -> list[int]:
        return await self.renderer.render(messages, tools=tools, add_generation_prompt=True)

    async def _extend(self, sid: str, anchor: HealAnchor, new: list[dict]) -> list[int]:
        """Splice ``anchor.served_ids`` with the closer and the render of ``new``, or
        return the full render when the template cannot be spliced."""
        render_delta = getattr(self.renderer, "render_delta", None)
        if render_delta is not None:
            delta = await render_delta(anchor.messages, new, tools=anchor.tools)
            if delta is not None:
                close, tail = delta
                return self._splice(sid, anchor.served_ids, close, tail)

        # r_prev is the token-space render of the gateway's OWN stored history; r_ext
        # extends that same stored prefix by only the genuinely-new messages, so the
        # client's re-serialized prior turns never enter the fed tokens.
        r_prev = await self.renderer.render(anchor.messages, tools=anchor.tools, add_generation_prompt=False)
        r_ext = await self.renderer.render(anchor.messages + new, tools=anchor.tools, add_generation_prompt=True)
        if r_ext[: len(r_prev)] != r_prev:
            # template glue for the prior turns depends on the appended messages, so the
            # served ids cannot be spliced safely.
            self._bump(sid, "unhealable")
            return r_ext

        close = await self._assistant_close(anchor.messages, anchor.tools, r_prev)
        if close is None:
            # couldn't isolate the assistant-closer (a template whose glue depends on
            # message type); don't emit a subtly-wrong sequence.
            self._bump(sid, "close_unresolved")
            self._bump(sid, "unhealable")
            return r_ext

        return self._splice(sid, anchor.served_ids, close, r_ext[len(r_prev) :])

    def _splice(self, sid: str, served_ids: list[int], close: list[int], tail: list[int]) -> list[int]:
        # don't duplicate close tokens the model already emitted at the end of its served
        # output (e.g. a trailing stop token kept by no_stop_trim).
        close = self._trim_overlap(served_ids, close)
        self._bump(sid, "healed_turns")
        self._bump(sid, "healed_prefix_tokens", len(served_ids) + len(close))
        # served_ids interleave rendered non-assistant messages with the served tokens of
        # every generated turn on the branch, so the result contains no drifted tokens.
        return list(served_ids) + close + tail

    async def _assistant_close(
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
        ra = await self.renderer.render(
            before + [{"role": "assistant", "content": _PROBE_A}], tools=tools, add_generation_prompt=False
        )
        rb = await self.renderer.render(
            before + [{"role": "assistant", "content": _PROBE_B}], tools=tools, add_generation_prompt=False
        )
        close = _common_suffix(ra, rb)
        if close and r_prev[-len(close) :] != close:
            return None  # glue is message-type-dependent for this template; don't guess
        return close

    @staticmethod
    def _trim_overlap(served_ids: list[int], close: list[int]) -> list[int]:
        """Drop the longest prefix of ``close`` that ``served_ids`` already ends with, so
        a closer the model emitted in its served output isn't duplicated."""
        for k in range(min(len(close), len(served_ids)), 0, -1):
            if served_ids[-k:] == close[:k]:
                return close[k:]
        return close
