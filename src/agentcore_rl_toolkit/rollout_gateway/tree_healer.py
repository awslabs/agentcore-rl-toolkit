"""Generate-time prefix healing over the session tree.

This is the engine behind the gateway's ``history_mode="tree_healed"``. It applies the
:class:`~.linear.LinearHealer` splice to every branch of the
:class:`~.trajectory.TrajectoryManager` tree instead of to one linear history.

For each request the healer asks the manager for the deepest generated turn whose root
path is a message-level prefix of the request (:meth:`TrajectoryManager.heal_anchor`).
The prompt reuses that turn's served token ids and renders only the messages after it:

    prompt_ids = anchor.served_ids + close + render(messages[len(anchor.messages):])

Retries, parallel tool-call branches, and sub-agent forks that reuse the session id
each extend their own anchor, so every branch reaches the manager CLEAN. A request
with no anchor (first turn, new root), a changed tools schema, or a template the
splice cannot handle is rendered in full, and the manager handles any drift as in
tree mode.

The healer holds no per-session history: the anchor comes from the manager's tree, which
``record_turn`` advances, so :meth:`commit` is a no-op.
"""

from __future__ import annotations

from .linear import LinearHealer, _State
from .render import Renderer
from .trajectory import TrajectoryManager

__all__ = ["TreeHealer", "make_healer"]


def make_healer(
    history_mode: str, renderer: Renderer, manager: TrajectoryManager, linear_on_nonlinear: str = "reset"
) -> LinearHealer | None:
    """The healer for ``history_mode``: ``LinearHealer`` for ``"linear"``,
    ``TreeHealer`` for ``"tree_healed"``, and ``None`` for ``"tree"``."""
    if history_mode == "linear":
        return LinearHealer(renderer, on_nonlinear=linear_on_nonlinear)
    if history_mode == "tree_healed":
        return TreeHealer(renderer, manager)
    return None


class TreeHealer(LinearHealer):
    """Per-branch generate-time prefix healing anchored on the manager's tree."""

    STATS_KEY = "tree_healer"
    COUNTER_KEYS = (
        "healed_turns",
        "healed_prefix_tokens",
        "unanchored",
        "tools_changed",
        "unhealable",
        "close_unresolved",
    )

    def __init__(self, renderer: Renderer, manager: TrajectoryManager) -> None:
        super().__init__(renderer)
        self.manager = manager

    async def heal(self, sid: str, messages: list[dict], tools: list[dict] | None) -> list[int]:
        anchor = self.manager.heal_anchor(sid, messages)
        tools = list(tools) if tools else None
        if anchor is None or anchor.tools != tools:
            self._bump(sid, "unanchored" if anchor is None else "tools_changed")
            return await self.renderer.render(messages, tools=tools, add_generation_prompt=True)
        st = _State(served_prefix=anchor.served_ids, prev_messages=anchor.messages, prev_tools=tools)
        return await self._extend(sid, st, messages[len(anchor.messages) :])

    def commit(self, sid: str, **_: object) -> None:
        """No-op: ``record_turn`` stores the turn the next anchor is read from."""

    def _handle_nonlinear(self, sid: str, r_full: list[int]) -> list[int]:
        # The splice failed for this template; feed the full render and let the
        # manager classify the drift.
        self._bump(sid, "unhealable")
        return r_full
