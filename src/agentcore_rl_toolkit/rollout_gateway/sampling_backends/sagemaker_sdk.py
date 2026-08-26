"""``SageMakerSdkBackend`` — in-process sampling via the SageMaker Training Sessions SDK.

Wraps a SageMaker ``SamplingClient``; rendering is done by the gateway (same model
as :class:`TinkerSdkBackend` — the SageMaker backend cannot render itself). This backend
only samples ``token_ids -> token_ids + logprobs``.

Unlike Tinker (which takes a fixed client), the SageMaker client needs to be rebound after
every weight update (``TrainingClient.save_weights_and_get_sampling_client()`` returns
a new client each time, and reusing the old one silently samples stale weights).
``set_sampling_client()`` handles this atomically while the gateway's background thread
may already be in a ``generate()`` call — in-flight requests finish against the old
weights, all subsequent calls use the new client.

``sagemaker`` is imported lazily so the core gateway stays lean. Install the SDK wheel
(``pip install sagemaker_train-*.whl``) before using this backend.
"""

import asyncio
import logging
import threading
from typing import Any

from sagemaker.train.training_session import SamplingParams as SagemakerSamplingParams
from sagemaker.train.training_session import StopReason

from ..trajectory import TurnRecord

logger = logging.getLogger(__name__)


class SageMakerSdkBackend:
    """``SamplingBackend`` over a SageMaker Training Sessions ``SamplingClient``.

    Args:
        sampling_client: A SageMaker ``SamplingClient`` obtained from
            ``TrainingClient.create_sampling_client()`` or
            ``TrainingClient.save_weights_and_get_sampling_client()``.
            Weights are fixed at client creation — call ``set_sampling_client()``
            after every ``save_weights_and_get_sampling_client()`` to stay on-policy.
    """

    def __init__(self, sampling_client: Any) -> None:
        self._sampling_client = sampling_client
        self._lock = threading.Lock()

    def set_sampling_client(self, sampling_client: Any) -> None:
        """Rebind to a new ``SamplingClient`` (e.g. after a weight update).

        Thread-safe: the training thread can call this while the gateway's background
        thread may be mid-generate. In-flight calls finish against the old client; all
        subsequent ``generate()`` calls acquire the new one.
        """
        with self._lock:
            self._sampling_client = sampling_client

    async def generate(
        self,
        *,
        prompt_ids: list[int],
        sampling_params: dict,
        session_id: str | None = None,
        image_data: Any = None,
        video_data: Any = None,
    ) -> TurnRecord:
        with self._lock:
            sc = self._sampling_client

        sp = SagemakerSamplingParams(
            max_tokens=int(sampling_params.get("max_new_tokens", 4096)),
            temperature=sampling_params.get("temperature", 1.0),
            top_p=sampling_params.get("top_p", 1.0),
            top_k=sampling_params.get("top_k"),
            stop=sampling_params.get("stop") or [],
        )
        # Run the blocking SDK poll in a thread-pool executor so the gateway's
        # event loop stays responsive during the wait. The SDK's APIFuture.result()
        # polls until the operation completes; asyncio.to_thread yields the loop
        # for the duration, matching the pattern used in the gsm8k demo.
        timeout = float(sampling_params.get("timeout", 900.0))
        op = sc.sample(prompt=list(prompt_ids), num_samples=1, sampling_params=sp)
        result = await asyncio.to_thread(op.result, timeout=timeout)

        seq = result.sequences[0]
        output_ids = list(seq.tokens)
        output_log_probs = list(seq.logprobs) if getattr(seq, "logprobs", None) is not None else []
        finish = "length" if seq.stop_reason == StopReason.LENGTH else "stop"

        return TurnRecord(
            prompt_ids=list(prompt_ids),
            output_ids=output_ids,
            finish_reason=finish,
            output_log_probs=output_log_probs,
        )


__all__ = ["SageMakerSdkBackend"]
