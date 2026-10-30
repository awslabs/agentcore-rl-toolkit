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


@dataclass(frozen=True)
class AuxiliaryTokenSamples:
    """``K`` extra sampler draws per generated token, not appended to the sequence.

    ``ids`` is ``(n, K)``: row ``i`` holds tokens drawn from the distribution token ``i``
    was sampled from. ``versions`` is the optional ``(n,)`` sampler weight version.
    """

    ids: Any
    versions: Any = None

    def __len__(self) -> int:
        return len(self.ids)

    def prefix(self, n: int) -> "AuxiliaryTokenSamples":
        return AuxiliaryTokenSamples(self.ids[:n], None if self.versions is None else self.versions[:n])


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
    # (offset into the response region, draws) for trained tokens.
    auxiliary_token_samples: list[tuple[int, AuxiliaryTokenSamples]] = field(default_factory=list)

    # Off-policy / MoE fields (weight_version, routed_experts) are intentionally
    # omitted for now; add them as optional fields if a consumer needs them.

    def __post_init__(self) -> None:
        if len(self.loss_mask) > len(self.token_ids):
            raise ValueError(
                f"TraceRecord loss_mask has {len(self.loss_mask)} entries but token_ids "
                f"has only {len(self.token_ids)}"
            )
        if len(self.logprobs) != len(self.loss_mask):
            raise ValueError(
                f"TraceRecord logprobs has {len(self.logprobs)} entries but loss_mask " f"has {len(self.loss_mask)}"
            )


__all__ = ["BaseTrace", "Status", "AuxiliaryTokenSamples", "TraceRecord"]
