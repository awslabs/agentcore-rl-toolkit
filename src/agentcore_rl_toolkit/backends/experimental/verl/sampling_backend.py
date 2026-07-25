"""``VerlSamplingBackend`` — token-in/token-out sampling backend over verl's
``LLMServerClient``.

The gateway renders canonical messages to ``prompt_ids`` and calls
:meth:`VerlSamplingBackend.generate`; this backend forwards them to verl's
in-cluster rollout replicas via ``LLMServerClient.generate`` (a Ray call, not
HTTP) and maps the resulting ``TokenOutput`` back to a ``TurnRecord``.

Passing the gateway session id as verl's ``request_id`` gives the whole episode
sticky routing to one replica (prefix-cache affinity); verl regenerates a fresh
per-turn engine request id internally.
"""

import logging
from typing import Any

from agentcore_rl_toolkit.rollout_gateway.trajectory import TurnRecord

logger = logging.getLogger(__name__)


def _verl_sampling_params(sp: dict) -> dict:
    """Map the gateway's canonical sampling dict to what verl's rollout servers
    accept.

    Whitelist rather than passthrough: the leftover keys go into the engine's
    ``SamplingParams(**...)`` constructor (vLLM) or sglang sampling dict, so an
    unknown key raises deep inside the engine. ``max_new_tokens`` is accepted by
    both verl servers (vLLM maps it to ``max_tokens``); ``logprobs`` is forced on
    because the trajectory always needs per-token logprobs.
    """
    body: dict[str, Any] = {
        "max_new_tokens": int(sp.get("max_new_tokens", 4096)),
        "logprobs": True,
    }
    if "temperature" in sp:
        body["temperature"] = sp["temperature"]
    if "top_p" in sp:
        body["top_p"] = sp["top_p"]
    tk = sp.get("top_k")
    if isinstance(tk, int) and (tk > 0 or tk == -1):
        body["top_k"] = tk
    if sp.get("stop"):
        body["stop"] = sp["stop"]
    if sp.get("stop_token_ids"):
        body["stop_token_ids"] = sp["stop_token_ids"]
    return body


class VerlSamplingBackend:
    """``SamplingBackend`` over verl's ``LLMServerClient`` (token-in/token-out)."""

    def __init__(self, server_manager: Any) -> None:
        """``server_manager`` is the ``LLMServerClient`` verl hands to every agent
        loop in this worker process (``AgentLoopBase.__init__``'s ``server_manager``)."""
        self.server_manager = server_manager
        # Per-sid fold of TokenOutput.extra_fields across turns: verl's v1 trainer
        # tags each trajectory with min/max weight versions for staleness control.
        self._extra_fields: dict[str, dict[str, Any]] = {}

    async def generate(
        self,
        *,
        prompt_ids: list[int],
        sampling_params: dict,
        session_id: str | None = None,
        image_data: Any = None,
        video_data: Any = None,
    ) -> TurnRecord:
        params = _verl_sampling_params(sampling_params)
        max_new_tokens = params["max_new_tokens"]
        output = await self.server_manager.generate(
            request_id=session_id or "default",
            prompt_ids=list(prompt_ids),
            sampling_params=params,
            image_data=image_data,
            video_data=video_data,
        )

        stop_reason = output.stop_reason
        if stop_reason in ("aborted", "abort"):
            # v1's FullyAsyncLLMServerClient resumes aborted generations internally,
            # so an abort surfacing here is a real failure. Raising 500s this turn on
            # the gateway; the agent sees an HTTP error and reports it via S3.
            raise RuntimeError(f"verl rollout aborted (stop_reason={stop_reason!r}) for session {session_id}")

        output_ids = list(output.token_ids)
        # verl's vLLM server collapses both 'stop' and 'length' into 'completed',
        # while SGLang passes the raw finish type through. The trajectory manager
        # keys truncation off finish_reason == "length", so infer it from the
        # token budget when the server didn't say.
        if stop_reason == "length" or len(output_ids) >= max_new_tokens:
            finish_reason = "length"
        else:
            finish_reason = "stop"

        if session_id:
            self._fold_extra_fields(session_id, output.extra_fields or {})

        log_probs = list(output.log_probs) if output.log_probs else [0.0] * len(output_ids)
        return TurnRecord(
            prompt_ids=list(prompt_ids),
            output_ids=output_ids,
            finish_reason=finish_reason,
            output_log_probs=log_probs,
        )

    def _fold_extra_fields(self, sid: str, extra: dict[str, Any]) -> None:
        folded = self._extra_fields.setdefault(sid, {})
        for key, pick in (("min_global_steps", min), ("max_global_steps", max)):
            value = extra.get(key)
            if value is None:
                continue
            folded[key] = value if key not in folded else pick(folded[key], value)
        if extra.get("global_steps") is not None:
            folded["global_steps"] = extra["global_steps"]

    def pop_extra_fields(self, sid: str) -> dict[str, Any]:
        """Return and clear the folded extra fields for ``sid`` (min/max global
        steps across all turns) — called once by the agent loop at finish."""
        return self._extra_fields.pop(sid, {})


__all__ = ["VerlSamplingBackend"]
