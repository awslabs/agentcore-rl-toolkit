"""``TinkerSdkBackend`` — in-process sampling via the Tinker SDK (no HTTP).

Wraps a Tinker ``SamplingClient``: rendering is done by the gateway (Tinker cannot
render itself — see :class:`TinkerRenderer`), and this backend only samples token_ids
-> token_ids + logprobs.

``tinker`` is imported lazily so the core stays torch-free. Install ``tinker`` manually
(it requires Python >=3.11) to use this backend.
"""

import logging
from typing import Any

from ..trajectory import TurnRecord

logger = logging.getLogger(__name__)


class TinkerSdkBackend:
    """``SamplingBackend`` over an in-process Tinker ``SamplingClient``.

    Args:
        sampling_client: a ``tinker`` ``SamplingClient`` (from
            ``service_client.create_sampling_client(...)``). Passed in so the caller
            controls model / weight-version lifecycle (create a fresh client after a
            weight update — stale clients silently sample old weights).
    """

    def __init__(self, sampling_client: Any) -> None:
        self.sampling_client = sampling_client

    async def generate(
        self,
        *,
        prompt_ids: list[int],
        sampling_params: dict,
        session_id: str | None = None,
        image_data: Any = None,
        video_data: Any = None,
    ) -> TurnRecord:
        import tinker  # lazy: pulls torch via tinker; only when this backend is used

        model_input = tinker.ModelInput.from_ints(list(prompt_ids))
        sp = tinker.SamplingParams(
            max_tokens=int(sampling_params.get("max_new_tokens", 4096)),
            temperature=sampling_params.get("temperature", 1.0),
            top_p=sampling_params.get("top_p", 1.0),
            stop=sampling_params.get("stop") or [],
        )
        result = await self.sampling_client.sample_async(prompt=model_input, num_samples=1, sampling_params=sp)
        seq = result.sequences[0]
        output_ids = list(seq.tokens)
        output_log_probs = list(seq.logprobs) if getattr(seq, "logprobs", None) is not None else []
        # tinker StopReason -> our coarse finish_reason; "length" must be preserved so
        # the trajectory metadata marks truncation.
        stop_reason = getattr(seq, "stop_reason", "stop")
        finish = "length" if str(stop_reason).lower() in ("length", "max_tokens") else "stop"

        return TurnRecord(
            prompt_ids=list(prompt_ids),
            output_ids=output_ids,
            finish_reason=finish,
            output_log_probs=output_log_probs,
        )


__all__ = ["TinkerSdkBackend"]
