"""``SglangHttpBackend`` — token-in/token-out HTTP sampling backend for SGLang.

POSTs rendered ``input_ids`` to ``{url}/generate`` and parses
``meta_info.output_token_logprobs`` into a :class:`TurnRecord`. On cancel/timeout it
eagerly hits ``/abort_request`` so an orphaned generation doesn't keep occupying KV
cache.

``num_top_logprobs`` requests SGLang's ``top_logprobs_num`` head. The binary
``output_topk_*`` fields added by slime's SGLang patch are preferred over the JSON
``output_top_logprobs`` that stock SGLang returns.
"""

import asyncio
import logging
import uuid
from typing import Any

import aiohttp

from ..trace import TopLogprobs
from ..trajectory import TurnRecord

logger = logging.getLogger(__name__)


def _top_logprobs_from_meta(meta: dict, n: int, k: int) -> TopLogprobs:
    """The ``(n, k)`` sampler head of a ``top_logprobs_num=k`` response."""
    import base64

    import numpy as np

    if "output_topk_shape" in meta:
        if list(meta["output_topk_shape"]) != [n, k]:
            raise RuntimeError(f"SGLang top-k head has shape {meta['output_topk_shape']}, expected {[n, k]}")
        ids = np.frombuffer(base64.b64decode(meta["output_topk_token_ids"]), dtype="<i4").reshape(n, k)
        logprobs = np.frombuffer(base64.b64decode(meta["output_topk_log_probs"]), dtype="<f4").reshape(n, k)
        return TopLogprobs(ids.astype(np.int32), logprobs.astype(np.float32))
    rows = meta.get("output_top_logprobs") or []
    if len(rows) != n or any(len(row or []) != k for row in rows):
        raise RuntimeError(f"SGLang did not return a top-{k} head for each of the {n} output tokens")
    ids = np.asarray([[entry[1] for entry in row] for row in rows], dtype=np.int32).reshape(n, k)
    logprobs = np.asarray([[entry[0] for entry in row] for row in rows], dtype=np.float32).reshape(n, k)
    return TopLogprobs(ids, logprobs)


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
        num_top_logprobs: int = 0,
    ) -> TurnRecord:
        rid = uuid.uuid4().hex
        body: dict[str, Any] = {
            "rid": rid,
            "input_ids": list(prompt_ids),
            "sampling_params": dict(sampling_params),
            "return_logprob": True,
        }
        if num_top_logprobs:
            body["top_logprobs_num"] = num_top_logprobs
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
            output_top_logprobs=(
                _top_logprobs_from_meta(meta, len(output_ids), num_top_logprobs) if num_top_logprobs else None
            ),
        )


__all__ = ["SglangHttpBackend"]
