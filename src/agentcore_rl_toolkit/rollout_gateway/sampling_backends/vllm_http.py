"""``VllmHttpBackend`` — token-in/token-out HTTP sampling backend for vLLM.

POSTs rendered ``token_ids`` to ``{url}/inference/v1/generate`` and parses the
response into a :class:`TurnRecord`. Expects this response shape:
```
{"choices": [{"token_ids": [int, ...],
              "logprobs": {"content": [{"logprob": float}, ...]},
              "finish_reason": "length"|"stop"|...,
              "routed_experts": null}]}
```
``routed_experts`` may be populated for MoE models but is currently ignored.
"""

import asyncio
import logging
from typing import Any

import aiohttp

from ..trace import AuxiliaryTokenSamples
from ..trajectory import TurnRecord
from .vllm_token_samples import SAMPLES_ENV

logger = logging.getLogger(__name__)


def _vllm_sampling_body(sp: dict) -> dict:
    """Map the canonical sampling dict to a vLLM ``/inference/v1/generate``
    ``sampling_params`` body. vLLM uses ``max_tokens`` (not ``max_new_tokens``) and
    returns per-token logprobs when ``logprobs`` is set."""
    body: dict[str, Any] = {
        "max_tokens": int(sp.get("max_new_tokens", 4096)),
        "logprobs": 1,
    }
    if "temperature" in sp:
        body["temperature"] = sp["temperature"]
    if "top_p" in sp:
        body["top_p"] = sp["top_p"]
    tk = sp.get("top_k")
    if tk is not None and (tk > 0 or tk == -1):
        body["top_k"] = tk
    if sp.get("stop"):
        body["stop"] = sp["stop"]
    if sp.get("stop_token_ids"):
        body["stop_token_ids"] = sp["stop_token_ids"]
    if sp.get("skip_special_tokens") is not None:
        body["skip_special_tokens"] = bool(sp["skip_special_tokens"])
    return body


def _tokens_and_logprobs_from_choice(choice: dict) -> tuple[list[int], list[float]]:
    """Parse ``token_ids`` + ``logprobs.content[i].logprob`` from a vLLM
    ``/inference/v1/generate`` choice."""
    tids_raw = choice.get("token_ids")
    if not (isinstance(tids_raw, list) and tids_raw and all(isinstance(x, int) for x in tids_raw)):
        return [], []
    tids = [int(x) for x in tids_raw]
    lp = choice.get("logprobs")
    if not isinstance(lp, dict):
        return tids, [0.0] * len(tids)
    content = lp.get("content")
    if isinstance(content, list) and content:
        lps: list[float] = []
        for i in range(len(tids)):
            if i < len(content) and isinstance(content[i], dict):
                lps.append(float(content[i].get("logprob", 0.0)))
            else:
                lps.append(0.0)
        return tids, lps
    return tids, [0.0] * len(tids)


def _auxiliary_token_samples_from_choice(choice: dict, num_tokens: int, k: int) -> AuxiliaryTokenSamples:
    """Draws from each position's ``top_logprobs`` (entry 0 is the sampled token)."""
    content = (choice.get("logprobs") or {}).get("content") or []
    rows = [[int(e["token"].removeprefix("token_id:")) for e in step.get("top_logprobs") or []] for step in content]
    if len(rows) != num_tokens or any(len(row) != k + 1 for row in rows):
        raise RuntimeError(f"vLLM did not return {k} token samples per position; is it running {SAMPLES_ENV}=1?")
    return AuxiliaryTokenSamples([row[1:] for row in rows])


class VllmHttpBackend:
    """``SamplingBackend`` over a vLLM ``/inference/v1/generate`` endpoint.

    ``num_auxiliary_token_samples`` needs the server started with the ``vllm_token_samples``
    plugin enabled (``AGENTCORE_VLLM_TOKEN_SAMPLES=1``).
    """

    def __init__(self, url: str, *, sock_read_timeout: float = 900.0) -> None:
        self.url = url.rstrip("/")
        self._sock_read_timeout = sock_read_timeout

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
        body = _vllm_sampling_body(sampling_params)
        if num_auxiliary_token_samples:
            body.update(logprobs=num_auxiliary_token_samples, flat_logprobs=True)
        payload: dict[str, Any] = {"token_ids": list(prompt_ids), "sampling_params": body}
        if image_data:
            payload["images"] = image_data
        # session_id routes via vllm-router's consistent_hash policy (x-session-id
        # header); harmlessly ignored by a bare single server.
        headers = {"x-session-id": session_id} if session_id and session_id != "default" else None
        timeout = aiohttp.ClientTimeout(total=None, sock_read=self._sock_read_timeout)
        task = asyncio.current_task()
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as sess,
                sess.post(
                    f"{self.url}/inference/v1/generate",
                    json=payload,
                    headers=headers,
                ) as r,
            ):
                if r.status >= 400:
                    text = await r.text()
                    logger.warning("[vllm_http] sid=%s vllm upstream %d: %.200s", session_id, r.status, text)
                    raise RuntimeError(f"vllm upstream {r.status}: {text[:400]}")
                data = await r.json(content_type=None)
            choice = (data.get("choices") or [{}])[0]
            output_ids, output_log_probs = _tokens_and_logprobs_from_choice(choice)
            fr = choice.get("finish_reason")
            finish = fr if isinstance(fr, str) and fr else "stop"
        except (asyncio.CancelledError, aiohttp.ClientError, asyncio.TimeoutError) as e:
            # vLLM ``/inference/v1/generate`` has no per-request HTTP abort endpoint.
            # Cancelling the in-flight task tears down the aiohttp request, dropping
            # the connection so vLLM stops generating.
            logger.debug("[vllm_http] sid=%s turn aborted: %s", session_id, type(e).__name__)
            if task is not None:
                task.cancel()
            raise

        return TurnRecord(
            prompt_ids=list(prompt_ids),
            output_ids=output_ids,
            finish_reason=finish,
            output_log_probs=output_log_probs,
            output_auxiliary_token_samples=(
                _auxiliary_token_samples_from_choice(choice, len(output_ids), num_auxiliary_token_samples)
                if num_auxiliary_token_samples
                else None
            ),
        )


__all__ = ["VllmHttpBackend"]
