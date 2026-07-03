"""The one per-backend seam: :class:`SamplingBackend`.

A sampling backend takes rendered prompt ``token_ids`` + canonical sampling params and
returns a :class:`~agentcore_rl_toolkit.rollout_gateway.trajectory.TurnRecord` — the
same type the adapter already feeds to ``TrajectoryManager.record_turn``. Token-in /
token-out; the entire per-backend difference (HTTP vs SDK, vLLM vs SGLang vs Tinker)
is confined to this one duck-typed method.

The contract intentionally reuses ``TurnRecord`` rather than a dedicated result type:
a backend returns exactly ``token_ids`` + logprobs + finish reason, nothing more.
Off-policy / MoE fields (weight version, routed experts) can be added as optional
``TurnRecord`` fields if and when a backend and consumer both need them.
"""

from typing import Any, Protocol, runtime_checkable

from ..trajectory import TurnRecord


@runtime_checkable
class SamplingBackend(Protocol):
    """token_ids -> token_ids + logprobs, as a ``TurnRecord``."""

    async def generate(
        self,
        *,
        prompt_ids: list[int],
        sampling_params: dict,
        session_id: str | None = None,
        image_data: Any = None,
        video_data: Any = None,
    ) -> TurnRecord:
        ...


__all__ = ["SamplingBackend"]
