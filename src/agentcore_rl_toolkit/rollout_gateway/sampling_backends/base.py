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

import inspect
from typing import Any, Protocol, runtime_checkable

from ..trajectory import TurnRecord


@runtime_checkable
class SamplingBackend(Protocol):
    """token_ids -> token_ids + logprobs, as a ``TurnRecord``.

    A backend that accepts ``num_auxiliary_token_samples`` can also draw ``K`` extra tokens per
    output token from the sampling distribution, returned as ``TurnRecord.output_auxiliary_token_samples``.
    The gateway passes it only when a session asks for draws, and rejects such sessions
    for backends whose ``generate`` lacks the parameter.
    """

    async def generate(
        self,
        *,
        prompt_ids: list[int],
        sampling_params: dict,
        session_id: str | None = None,
        image_data: Any = None,
        video_data: Any = None,
        num_auxiliary_token_samples: int = 0,
    ) -> TurnRecord:
        ...


def supports_auxiliary_token_samples(backend: Any) -> bool:
    return "num_auxiliary_token_samples" in inspect.signature(backend.generate).parameters


__all__ = ["SamplingBackend", "supports_auxiliary_token_samples"]
