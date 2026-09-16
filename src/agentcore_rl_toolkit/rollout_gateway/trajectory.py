"""Build a per-session training trajectory from multi-turn conversation data.

The :class:`TrajectoryManager` builds one trajectory per session. ``record_turn``
feeds in each turn (prompt messages + the served model's token snapshot), routing it
into a per-sid message tree; ``get_trajectory`` then linearizes that tree into a
``list[TraceRecord]`` of loss-masked training rows, tolerating re-tokenization drift
between turns via a CLEAN / FORK classification (see :class:`DriftKind`).

The manager is tokenizer-free and torch-free: it operates purely on the token ids in
each :class:`TurnRecord` and emits plain :class:`TraceRecord` rows.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
from collections.abc import Iterator
from typing import Any

from .trace import BaseTrace, Status, TraceRecord

logger = logging.getLogger(__name__)


# ===========================================================================
# TurnRecord
# ===========================================================================


@dataclasses.dataclass(frozen=True)
class TurnRecord:
    """One backend ``generate`` snapshot: the contract between an adapter and the
    manager. Adapters build it from a turn's prompt/output token ids; ``record_turn``
    consumes it. Sampling backends return this type directly."""

    prompt_ids: list[int]
    output_ids: list[int]
    finish_reason: str
    output_log_probs: list[float] = dataclasses.field(default_factory=list)
    ill_formed: bool = False


# ===========================================================================
# MessageNode
# ===========================================================================


class MessageNode:
    """One node in a session's routing tree, carrying a single chat message
    (``None`` for the dummy root and for an assistant leaf we generated but
    whose ``response_message`` was empty).

    The two kinds are distinguished by whether ``turn`` is set, which reflects
    WHERE the message came from:

    * **generated** (``turn is not None``): an assistant message the model
      actually generated this turn, fed in via ``record_turn``. ``turn`` holds
      its :class:`TurnRecord` -- the prompt/output ids, logprobs and finish
      reason that ``get_trajectory`` linearizes into training tokens.
    * **routing-only** (``turn is None``): the message came from the prompt, not
      from generation, so it only exists to route. This is every
      system/user/tool node, AND any assistant we did NOT generate: a foreign
      assistant the client replayed in a later prompt.
    """

    def __init__(
        self,
        *,
        role: str | None = None,
        message: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        parent: MessageNode | None = None,
    ) -> None:
        self.role = role
        self.message = message
        self.metadata = dict(metadata or {})
        self.parent: MessageNode | None = parent
        self.children: list[MessageNode] = []
        self.turn: TurnRecord | None = None  # the generated TurnRecord, else None (routing-only)
        self.turn_index: int | None = None
        # Shared by sibling leaf paths; the first to reach it trains on it, the rest
        # re-emit it as loss_mask=0 context -- so each response is trained exactly once.
        self.response_trained: bool = False

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def add_child(self, child: MessageNode) -> MessageNode:
        child.parent = self
        self.children.append(child)
        return child

    def path_from_root(self) -> list[MessageNode]:
        """Ordered list of nodes from the first non-root ancestor down to self."""
        chain: list[MessageNode] = []
        node: MessageNode | None = self
        while node is not None and not node.is_root:
            chain.append(node)
            node = node.parent
        chain.reverse()
        return chain

    def leaves(self) -> Iterator[MessageNode]:
        # Iterative left-to-right DFS with an explicit stack: children are pushed
        # in reverse so they pop in order, and depth is bounded by the tree's
        # breadth rather than the interpreter's C stack.
        stack: list[MessageNode] = [self]
        while stack:
            node = stack.pop()
            if not node.children:
                yield node
            else:
                stack.extend(reversed(node.children))


# ===========================================================================
# drift classification — how an incoming turn's prompt relates to held tokens
# ===========================================================================


def _common_prefix_len(a: list[int], b: list[int], chunk: int = 4096) -> int:
    limit = min(len(a), len(b))
    matched = 0
    while matched < limit:
        chunk_end = min(matched + chunk, limit)
        if a[matched:chunk_end] == b[matched:chunk_end]:
            matched = chunk_end
        else:
            while matched < chunk_end and a[matched] == b[matched]:
                matched += 1
            return matched
    return matched


class DriftKind(enum.Enum):
    CLEAN = "clean"  # drift == 0: prompt_ids exactly extends held tokens; append the tail beyond them
    FORK = "fork"  # everything else: close this builder, open a fresh one as a fork


# ===========================================================================
# SampleBuilder — accumulates turns into one trainable TraceRecord (fork closes it)
# ===========================================================================


class _SampleBuilder:
    """Accumulates a chain's turns into the token sequence of one ``TraceRecord``.

    A chain of turns is appended one at a time via :meth:`append_turn`. Ideally
    each turn's prompt exactly extends the tokens we already hold, but a replayed
    turn rarely re-tokenizes byte-for-byte: decoding a turn back to text and
    re-rendering it (chat-template round-trips) can perturb the ids of content we've
    already seen. The builder handles this drift in a source-agnostic way, classified
    by whether the prompt diverges from the held tokens (see
    :meth:`classify_token_drift`):

    * **CLEAN** -- no drift; append the prompt tail beyond what we hold.
    * **FORK** -- the prompt diverges from the held tokens; this builder is
      rejected and the caller closes it and opens a fresh one. That boundary is
      the "fork".

    Each surviving builder yields one TraceRecord.
    """

    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.loss_mask: list[int] = []
        self.logprobs: list[float] = []
        self.last_response_start_idx: int | None = None
        self.leading_prompt_len: int = 0

    def classify_token_drift(self, turn: TurnRecord) -> DriftKind:
        """Decide how this builder should absorb ``turn``'s prompt.

        The incoming turn's prompt is expected to match the tokens this builder
        already holds as an exact prefix. When token drift has occurred -- the
        prompt diverges from the held tokens -- we FORK to preserve the
        previously generated response. With no drift the turn is handled the CLEAN
        way -- a plain prefix extension.
        """
        prefix_len = _common_prefix_len(self.tokens, turn.prompt_ids)
        drift = len(self.tokens) - prefix_len

        if drift == 0:
            return DriftKind.CLEAN

        return DriftKind.FORK

    def append_turn(self, turn: TurnRecord, kind: DriftKind, *, trained: bool = True) -> None:
        """Append a CLEAN turn's prompt tail and generated response."""
        assert kind is not DriftKind.FORK, "append_turn called on a builder that would fork"

        is_first_turn = self.last_response_start_idx is None

        # --- append this turn's prompt tail (loss_mask=0) ---
        self._append_tokens(turn.prompt_ids[len(self.tokens) :], loss_mask=0)

        # --- append this turn's generated response (loss_mask=1 unless re-emitted as context) ---
        self.last_response_start_idx = len(self.tokens)
        self._append_tokens(
            turn.output_ids, loss_mask=int(trained), logprobs=turn.output_log_probs if trained else None
        )

        if is_first_turn:
            self.leading_prompt_len = len(turn.prompt_ids)

    def _append_tokens(self, ids: list[int], *, loss_mask: int, logprobs: list[float] | None = None) -> None:
        self.tokens.extend(ids)
        self.loss_mask.extend([loss_mask] * len(ids))
        self.logprobs.extend(logprobs if logprobs else [0.0] * len(ids))

    def has_trained_response(self) -> bool:
        return any(self.loss_mask[self.leading_prompt_len :])

    def to_sample(
        self, base_sample: BaseTrace, extra_metadata: dict[str, Any] | None, max_sample_tokens: int = 0
    ) -> TraceRecord:
        """Emit the accumulated tokens as one ``TraceRecord``, stripping the first-turn
        prompt so loss_mask / logprobs cover only the response region."""
        start = self.leading_prompt_len  # first-turn prompt stripped; response region starts here
        tokens = list(self.tokens)
        loss_mask = self.loss_mask
        logprobs = self.logprobs
        if max_sample_tokens and len(tokens) > max_sample_tokens:
            tokens = tokens[:max_sample_tokens]
            loss_mask = loss_mask[:max_sample_tokens]
            logprobs = logprobs[:max_sample_tokens]
        md = dict(extra_metadata or {})
        return TraceRecord(
            token_ids=tokens,
            loss_mask=loss_mask[start:],
            logprobs=logprobs[start:],
            rollout_id=base_sample.rollout_id if base_sample.rollout_id is not None else base_sample.index,
            reward=0.0,
            response_length=len(loss_mask) - start,
            response="",
            metadata=md,
            status=Status.COMPLETED,
        )


# ===========================================================================
# TrajectoryManager
# ===========================================================================


class TrajectoryManager:
    def __init__(self) -> None:
        self._trees: dict[str, MessageNode] = {}
        self._turn_count: dict[str, int] = {}

    # -------------------- public ------------------------------------------

    def has_session(self, sid: str) -> bool:
        return sid in self._trees

    def turn_count(self, sid: str) -> int:
        return self._turn_count.get(sid, 0)

    def record_turn(
        self,
        sid: str,
        *,
        turn: TurnRecord,
        prompt_messages: list[dict[str, Any]],
        response_message: dict[str, Any] | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not prompt_messages:
            logger.warning("record_turn(sid=%s): empty prompt_messages; skipping", sid)
            return
        assert not turn.output_log_probs or len(turn.output_log_probs) == len(turn.output_ids), (
            f"turn.output_log_probs length {len(turn.output_log_probs)} != "
            f"turn.output_ids length {len(turn.output_ids)}"
        )

        root = self._trees.setdefault(sid, MessageNode())

        node, depth = self._find_mount_point(root, prompt_messages)
        node = self._mount_prompt_messages(node, prompt_messages[depth:])
        self._attach_assistant_leaf(sid, node, turn=turn, response_message=response_message, metadata=metadata)

    def get_trajectory(
        self,
        sid: str,
        *,
        base_sample: BaseTrace,
        reward: float = 0.0,
        extra_metadata: dict[str, Any] | None = None,
        max_sample_tokens: int = 0,
    ) -> list[TraceRecord]:
        """Linearize this sid's routing tree into ``TraceRecord`` objects and
        consume the session.

        Each routing leaf yields one or more TraceRecords; ``reward`` is assigned in
        full to every emitted record (not split across them), so each trained
        turn carries the trajectory's outcome reward. The sid is dropped
        afterwards, so a second call for the same sid returns ``[]``.
        """
        root = self._trees.get(sid)
        if root is None:
            return []

        samples: list[TraceRecord] = []
        for routing_leaf in root.leaves():
            if routing_leaf.is_root:
                continue
            chain = routing_leaf.path_from_root()
            samples.extend(
                self._chain_to_samples(
                    chain, base_sample=base_sample, extra_metadata=extra_metadata, max_sample_tokens=max_sample_tokens
                )
            )

        for s in samples:
            s.reward = reward

        self._trees.pop(sid, None)
        self._turn_count.pop(sid, None)
        return samples

    def drop_session(self, sid: str) -> None:
        self._trees.pop(sid, None)
        self._turn_count.pop(sid, None)

    # -------------------- internals ----------------------------------------

    def _find_mount_point(self, root: MessageNode, messages: list[dict[str, Any]]) -> tuple[MessageNode, int]:
        """Walk down the tree matching each message by role and dict equality (==),
        returning the deepest node that still matches and where to mount the rest."""
        node = root
        depth = 0
        while depth < len(messages):
            msg = messages[depth]
            next_child = None
            for child in node.children:
                if child.role == msg.get("role") and child.message == msg:
                    next_child = child
                    break
            if next_child is None:
                break
            node = next_child
            depth += 1
        return node, depth

    def _mount_prompt_messages(
        self,
        node: MessageNode,
        remaining_messages: list[dict[str, Any]],
    ) -> MessageNode:
        for m in remaining_messages:
            node = node.add_child(MessageNode(role=m.get("role"), message=m))
        return node

    def _attach_assistant_leaf(
        self,
        sid: str,
        node: MessageNode,
        *,
        turn: TurnRecord,
        response_message: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        asst = MessageNode(
            role="assistant",
            message=response_message,
            metadata=dict(metadata or {}),
        )
        asst.turn = turn
        asst.turn_index = self._turn_count.get(sid, 0) + 1
        node.add_child(asst)
        self._turn_count[sid] = asst.turn_index

    def _split_chain_into_builders(self, chain: list[MessageNode]) -> list[_SampleBuilder]:
        """Pack the chain's generated turns into per-sample token builders.

        Turns flow into the current builder until one can't extend it as an
        exact prefix (re-tokenization drift); that turn
        opens a new builder -- a fork. A generated turn shared by sibling leaves
        is trained only on the first leaf to claim it; later leaves re-emit it
        as loss_mask=0 context so the shared prefix isn't double-counted.
        """
        asst_nodes = [n for n in chain if n.role == "assistant" and n.turn is not None]

        builders: list[_SampleBuilder] = []
        for asst_node in asst_nodes:
            trained = not asst_node.response_trained
            asst_node.response_trained = True

            if not builders or (kind := builders[-1].classify_token_drift(asst_node.turn)) is DriftKind.FORK:
                builders.append(_SampleBuilder())
                builders[-1].append_turn(asst_node.turn, DriftKind.CLEAN, trained=trained)
            else:
                builders[-1].append_turn(asst_node.turn, kind, trained=trained)
        return builders

    def _chain_to_samples(
        self,
        chain: list[MessageNode],
        *,
        base_sample: BaseTrace,
        extra_metadata: dict[str, Any] | None,
        max_sample_tokens: int = 0,
    ) -> list[TraceRecord]:
        asst_nodes = [n for n in chain if n.role == "assistant" and n.turn is not None]
        truncated = bool(asst_nodes) and asst_nodes[-1].turn.finish_reason == "length"
        use_tool = any(bool((n.message or {}).get("tool_calls")) for n in asst_nodes)
        ill_formed = any(n.turn.ill_formed for n in asst_nodes)
        md = {
            **(extra_metadata or {}),
            "truncated": truncated,
            "use_tool": use_tool,
            "ill_formed": ill_formed,
        }
        return [
            builder.to_sample(base_sample, md, max_sample_tokens)
            for builder in self._split_chain_into_builders(chain)
            if builder.has_trained_response()
        ]


__all__ = [
    "MessageNode",
    "TrajectoryManager",
    "TurnRecord",
]
