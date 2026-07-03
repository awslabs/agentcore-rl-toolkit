"""``TraceRecord`` — the torch-free output boundary of the rollout gateway.

The trajectory core (``trajectory.py``) linearizes a per-session message tree into
a list of ``TraceRecord`` objects. A ``TraceRecord`` is a plain dataclass holding
loss-masked, token-level trajectory data, with no dependency on torch or any training
framework. Each training backend converts it into its own native sample type (e.g. a
verl ``DataProto`` or a tinker ``Datum``) in the backend's own process.

Fields are the loss-masked token data the trajectory tree produces, plus a
``rollout_id`` — the per-episode grouping key, set by the caller that orchestrates
rollouts (all records sharing a ``rollout_id`` belong to one episode).
"""

import enum
from dataclasses import dataclass, field
from typing import Any


class Status(enum.Enum):
    """Terminal status of a captured trajectory. String-valued so it serializes
    cleanly to JSON via ``.value``."""

    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass
class BaseTrace:
    """Per-episode identity/context carried into ``TrajectoryManager.get_trajectory``.

    The trajectory core reads these scalar fields off the ``base_sample`` it is handed
    when linearizing a tree into :class:`TraceRecord` rows (e.g. to populate
    ``rollout_id``); it holds no tensors or framework types.
    """

    index: int = 0
    group_index: int = 0
    rollout_id: str | int | None = None
    prompt: Any = None
    label: Any = None


@dataclass
class TraceRecord:
    """One loss-masked training row emitted from a session's trajectory tree.

    ``token_ids`` is the full sequence for this row; ``loss_mask`` / ``logprobs`` cover
    the response region only (the first-turn prompt is stripped). ``rollout_id`` groups
    all rows of one episode: the loss reducer treats rows sharing a ``rollout_id`` as a
    single rollout.
    """

    token_ids: list[int]
    loss_mask: list[int]
    logprobs: list[float]
    rollout_id: str | int | None = None
    reward: float = 0.0
    response_length: int = 0
    response: str = ""
    metadata: dict = field(default_factory=dict)
    status: Status = Status.COMPLETED

    # Off-policy / MoE fields (weight_version, routed_experts) are intentionally
    # omitted for now; add them as optional fields if a consumer needs them.


__all__ = ["BaseTrace", "Status", "TraceRecord"]
