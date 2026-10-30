"""``SglangHttpBackend`` — token-in/token-out HTTP sampling backend for SGLang.

POSTs rendered ``input_ids`` to ``{url}/generate`` and parses
``meta_info.output_token_logprobs`` into a :class:`TurnRecord`. On cancel/timeout it
eagerly hits ``/abort_request`` so an orphaned generation doesn't keep occupying KV
cache.

``num_auxiliary_token_samples`` needs the server started with the ``sglang_token_samples``
plugin enabled (``AGENTCORE_SGLANG_TOKEN_SAMPLES=1``).
"""

import asyncio
import logging
import uuid
from typing import Any

import aiohttp

from ..trace import AuxiliaryTokenSamples
from ..trajectory import TurnRecord
from .sglang_token_samples import SAMPLES_ENV

logger = logging.getLogger(__name__)


def _auxiliary_token_samples_from_meta(meta: dict, output_ids: list[int], k: int) -> AuxiliaryTokenSamples:
    """Draws from each position's ``output_top_logprobs`` (entry 0 is the sampled token)."""
    rows = [[int(entry[1]) for entry in step or []] for step in meta.get("output_top_logprobs") or []]
    if len(rows) != len(output_ids) or any(
        len(row) != k + 1 or row[0] != token for row, token in zip(rows, output_ids, strict=True)
    ):
        raise RuntimeError(
            f"SGLang did not return {k} token samples per position; is it running {SAMPLES_ENV}=1 "
            "and sampling without top-k, top-p, or min-p?"
        )
    version = str(meta.get("weight_version", ""))
    versions = [int(version)] * len(rows) if version.isdigit() else None
    return AuxiliaryTokenSamples([row[1:] for row in rows], versions)


class SglangHttpBackend:
    """``SamplingBackend`` over an SGLang ``/generate`` endpoint."""

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
        rid = uuid.uuid4().hex
        body: dict[str, Any] = {
            "rid": rid,
            "input_ids": list(prompt_ids),
            "sampling_params": dict(sampling_params),
            "return_logprob": True,
        }
        if num_auxiliary_token_samples:
            body["top_logprobs_num"] = num_auxiliary_token_samples
        headers = {"X-SMG-Routing-Key": session_id} if session_id and session_id != "default" else None
        timeout = aiohttp.ClientTimeout(total=None, sock_read=self._sock_read_timeout)
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as sess,
                sess.post(f"{self.url}/generate", json=body, headers=headers) as r,
            ):
                if r.status >= 400:
                    text = await r.text()
                    logger.warning(
                        "[sglang_http] sid=%s rid=%s sglang upstream %d: %.200s",
                        session_id,
                        rid,
                        r.status,
                        text,
                    )
                    raise RuntimeError(f"sglang upstream {r.status}: {text[:400]}")
                data = await r.json(content_type=None)
            meta = data.get("meta_info") or {}
            output_token_logprobs = meta.get("output_token_logprobs") or []
            output_ids = [x[1] for x in output_token_logprobs]
            output_log_probs = [float(x[0]) for x in output_token_logprobs]
            finish = (meta.get("finish_reason") or {}).get("type", "stop") or "stop"
        except (asyncio.CancelledError, aiohttp.ClientError, asyncio.TimeoutError) as e:
            # free the sglang slot eagerly on client cancel/timeout, else the
            # orphaned generation keeps occupying KV until its own length cap.
            logger.debug("[sglang_http] sid=%s rid=%s turn aborted: %s", session_id, rid, type(e).__name__)
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s2:
                    await s2.post(f"{self.url}/abort_request", json={"rid": rid})
            except Exception:
                pass
            raise

        return TurnRecord(
            prompt_ids=list(prompt_ids),
            output_ids=output_ids,
            finish_reason=finish,
            output_log_probs=output_log_probs,
            output_auxiliary_token_samples=(
                _auxiliary_token_samples_from_meta(meta, output_ids, num_auxiliary_token_samples)
                if num_auxiliary_token_samples
                else None
            ),
        )


__all__ = ["SglangHttpBackend"]
