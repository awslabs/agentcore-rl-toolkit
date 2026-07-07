"""Shared adapter primitives for token-capturing agent rollouts.

A protocol adapter (Anthropic / OpenAI) subclasses BaseAdapter and fills in the
wire-specific hooks (_register_routes, _session_id, _translate, _build_reply,
_respond, and optionally _preprocess_body) plus a few class attributes (logger,
log_prefix, max_token_keys, stop_keys). The session lifecycle, per-sid turn cap,
inflight-task bookkeeping and the one-turn _run_turn pipeline are inherited.

The adapter is wired with two injected seams so it stays engine- and
tokenizer-agnostic:

1. ``backend: SamplingBackend`` — ``_run_turn`` calls ``backend.generate(...)`` for the
   token-in / token-out step. The whole per-engine difference (vLLM / SGLang / Tinker)
   lives behind that one method, which returns a ``TurnRecord``.
2. ``renderer: Renderer`` — rendering prompts to token ids and parsing responses both
   go through the renderer, so tokenization is pluggable.

``finish_session`` returns ``list[TraceRecord]``.
"""

import asyncio
import dataclasses
import logging
import time
from collections.abc import Callable
from typing import Any

from aiohttp import web

from ..render import Renderer
from ..sampling_backends.base import SamplingBackend
from ..trace import BaseTrace, TraceRecord
from ..trajectory import TrajectoryManager, TurnRecord

__all__ = ["BaseAdapter", "Reply", "Session", "sid_from_bearer", "sid_from_body"]


@dataclasses.dataclass
class Session:
    """Per-sid adapter state: sampling defaults and context budget.

    Trajectory state lives in the shared TrajectoryManager (BaseAdapter.manager),
    not here.
    """

    sampling_defaults: dict = dataclasses.field(default_factory=dict)
    max_context_tokens: int = 0


@dataclasses.dataclass
class Reply:
    """Output of an adapter's _build_reply, consumed by _run_turn.

    manager_message and finish_reason feed record_turn and the debug callback;
    wire is opaque to the pipeline and only the adapter's own _respond reads it.
    """

    manager_message: dict
    finish_reason: str
    wire: Any


def flatten_content(c: Any) -> str:
    """Flatten a wire content value into a chat-template string.

    Handles both Anthropic and OpenAI block shapes. A non-list value (str /
    dict / other) is returned via str() unchanged.
    """
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if not isinstance(c, list):
        return str(c)
    parts: list[str] = []
    for b in c:
        if isinstance(b, str):
            parts.append(b)
            continue
        if not isinstance(b, dict):
            parts.append(str(b))
            continue
        t = b.get("type")
        if t in {"text", "input_text", "output_text"}:
            parts.append(b.get("text", ""))
        elif t == "tool_result":
            parts.append(flatten_content(b.get("content")))
        elif t in {"image", "image_url", "input_image"}:
            parts.append("[image omitted]")
        elif "content" in b:
            parts.append(flatten_content(b.get("content")))
        elif "text" in b:
            parts.append(str(b.get("text") or ""))
    return "\n".join(p for p in parts if p)


def tool_call_dict(name: str, arguments: dict | None) -> dict:
    """Canonical OpenAI-shape tool call stored on manager_message.

    arguments stays a dict (not a JSON string): the chat template needs a
    mapping, and the trajectory manager matches history by dict equality, so a
    sampled leaf and its replayed echo compare equal regardless of key order.
    The wire-only tool-call id is dropped for the same reason.
    """
    return {"type": "function", "function": {"name": name, "arguments": arguments or {}}}


def manager_finish_reason(tool_uses: list[dict], raw_finish: str) -> str:
    """Finish reason stored on the manager turn: tool_calls if the turn called a
    tool, else the raw backend finish."""
    return "tool_calls" if tool_uses else (raw_finish or "stop")


class BaseAdapter:
    """Base HTTP adapter: session lifecycle plus the shared one-turn pipeline.

    See the module docstring for the class attributes and hooks a subclass must
    supply; everything else is inherited.
    """

    logger: logging.Logger = logging.getLogger(__name__)
    log_prefix: str = "adapter"
    # body keys that cap max_new_tokens and carry stop sequences, in priority order
    max_token_keys: tuple[str, ...] = ()
    stop_keys: tuple[str, ...] = ()
    manager: Any

    def __init__(
        self,
        *,
        backend: SamplingBackend,
        renderer: Renderer,
        tokenizer=None,
        max_turns_per_sid: int | None = None,
        fork_threshold_tokens: int | None = None,
        debug_callback: Callable[..., None] | None = None,
        manager: TrajectoryManager | None = None,
        app: web.Application | None = None,
    ) -> None:
        self.backend = backend
        self.renderer = renderer
        # tokenizer kept for finish_session's response-tail decode (manager is
        # tokenizer-free); optional so a renderer that decodes itself can omit it.
        self.tokenizer = tokenizer
        self.store: dict[str, Any] = {}
        self.inflight: dict[str, set[asyncio.Task]] = {}
        self.closed: set[str] = set()

        # one manager shared across all sids (and across co-mounted adapters, when
        # passed in); per-sid trees live inside it. fork_threshold_tokens left None
        # means the manager uses its own default.
        if manager is not None:
            self.manager = manager
        else:
            mgr_kwargs: dict[str, int] = {}
            if fork_threshold_tokens is not None:
                mgr_kwargs["fork_threshold_tokens"] = fork_threshold_tokens
            self.manager = TrajectoryManager(**mgr_kwargs)

        self.debug_callback: Callable[..., None] | None = debug_callback
        # per-sid turn cap: return 429 to kill the run once exceeded
        self.max_turns_per_sid: int | None = max_turns_per_sid
        self._sid_turn_count: dict[str, int] = {}

        # app may be shared across co-mounted adapters (one aiohttp app, many routes)
        self.app = app if app is not None else web.Application(client_max_size=64 * 1024 * 1024)
        if app is None:
            self.app.router.add_get("/healthz", _health)
            self.app.router.add_get("/v1/models", _health)
        self._register_routes(self.app)

    # -- wire hooks (subclass overrides) -------------------------------------

    def _register_routes(self, app: web.Application) -> None:
        """Register the protocol's POST route(s) and bind self._run_turn."""
        raise NotImplementedError

    def _session_id(self, request: web.Request, body: dict) -> str:
        raise NotImplementedError

    def _preprocess_body(self, body: dict) -> None:
        """Mutate the parsed body in place before sid resolution (default no-op)."""

    def _translate(self, body: dict) -> tuple[list[dict], list[dict] | None]:
        """Return (chat_messages, tools_schema) from the wire body."""
        raise NotImplementedError

    def _build_reply(self, parsed, raw_finish: str, translated: list[dict], tools_schema: list[dict] | None) -> Reply:
        """Pack parsed model output into a Reply."""
        raise NotImplementedError

    async def _respond(
        self,
        request: web.Request,
        body: dict,
        reply: Reply,
        in_tok: int,
        out_tok: int,
        stream: bool,
    ) -> web.StreamResponse:
        raise NotImplementedError

    # -- session lifecycle ---------------------------------------------------

    def open_session(
        self,
        sid: str,
        *,
        sampling_defaults: dict | None = None,
        max_context_tokens: int = 0,
    ) -> None:
        """Register a fresh per-sid Session; sids must be unique."""
        if sid in self.store:
            raise ValueError(f"session_id {sid!r} already exists; sids must be unique per agent run")
        self.store[sid] = Session(
            sampling_defaults=dict(sampling_defaults or {}),
            max_context_tokens=int(max_context_tokens or 0),
        )

    async def shutdown_session(self, sid: str, *, wait_timeout: float = 5.0) -> None:
        """Mark a sid closed and drain its in-flight turn tasks."""
        self.closed.add(sid)
        tasks = [t for t in self.inflight.pop(sid, ()) if not t.done()]
        if not tasks:
            return

        async def _drain() -> None:
            _, pending = await asyncio.wait(tasks, timeout=wait_timeout)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        loop = tasks[0].get_loop()
        try:
            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(_drain(), loop))
        except Exception:
            self.logger.exception("[%s] sid=%s shutdown drain failed", self.log_prefix, sid)

    async def finish_session(
        self,
        sid: str,
        *,
        base_sample: BaseTrace | None = None,
        reward: float = 0.0,
        extra_metadata: dict | None = None,
        wait_timeout: float = 5.0,
    ) -> list[TraceRecord]:
        """Drain a session's trajectory into fully-formed TraceRecord objects.

        Waits out in-flight requests for the sid, linearises the per-sid tree,
        then decodes each record's trained tail into .response (the manager is
        tokenizer-free, so the adapter that owns the tokenizer fills this in).
        Idempotent: a second call for an already-popped sid returns [].
        """
        await self.shutdown_session(sid, wait_timeout=wait_timeout)
        session = self.store.pop(sid, None)
        base_sample = base_sample if base_sample is not None else BaseTrace()
        max_sample_tokens = int(getattr(session, "max_context_tokens", 0) or 0) if session is not None else 0
        samples = self.manager.get_trajectory(
            sid,
            base_sample=base_sample,
            reward=reward,
            extra_metadata=extra_metadata,
            max_sample_tokens=max_sample_tokens,
        )
        if self.tokenizer is not None:
            for s in samples:
                rlen = int(s.response_length or 0)
                s.response = (
                    self.tokenizer.decode(s.token_ids[-rlen:], skip_special_tokens=False)
                    if rlen and s.token_ids
                    else ""
                )
        return samples

    async def drop_session(self, sid: str, *, wait_timeout: float = 5.0) -> None:
        await self.shutdown_session(sid, wait_timeout=wait_timeout)
        self.store.pop(sid, None)
        self.manager.drop_session(sid)

    # -- shared request pipeline ---------------------------------------------

    def _check_turn_cap(self, sid: str) -> web.Response | None:
        """Enforce max_turns_per_sid, returning a 429 response once exceeded.

        Increments the per-sid counter as a side effect when under the cap.
        """
        cap = self.max_turns_per_sid
        if cap is None:
            return None
        prior = self._sid_turn_count.get(sid, 0)
        if prior >= cap:
            self.logger.warning("[%s] sid=%s exceeded max_turns_per_sid=%d; killing run", self.log_prefix, sid, cap)
            return web.json_response(
                {
                    "error": {
                        "type": "rate_limit_error",
                        "message": (f"adapter: sid {sid!r} exceeded max_turns_per_sid={cap}; killing run"),
                    }
                },
                status=429,
            )
        self._sid_turn_count[sid] = prior + 1
        return None

    def _run_debug_callback(self, sid, translated, tools_schema, manager_message, turn) -> None:
        """Run the optional debug-only data dump callback; unset in production."""
        callback = self.debug_callback
        if callback is None:
            return
        try:
            callback(sid, translated, tools_schema, manager_message, turn)
        except Exception:
            self.logger.exception("debug_callback failed (sid=%s)", sid)

    async def _run_turn(self, request: web.Request) -> web.StreamResponse:
        """One full agent turn: translate -> render -> sample -> parse -> append -> respond.

        The wire-specific steps are delegated to the subclass hooks; rendering and
        sampling go through the injected renderer/backend; the rest (sid resolution,
        closed/cap guards, inflight tracking, record_turn) is shared across protocols.
        """
        body = await request.json()
        self._preprocess_body(body)
        sid = self._session_id(request, body)
        if sid in self.closed:  # session drained; refuse stragglers
            self.logger.debug("[%s] sid=%s request after session closed", self.log_prefix, sid)
            return web.Response(status=503, text="session closed")
        capped = self._check_turn_cap(sid)
        if capped is not None:
            return capped

        s = self.store.setdefault(sid, Session())
        task = asyncio.current_task()
        self.inflight.setdefault(sid, set()).add(task)
        t0 = time.monotonic()
        try:
            translated, tools_schema = self._translate(body)
            prompt_ids = self.renderer.render(translated, tools=tools_schema, add_generation_prompt=True)

            sampling_params = _sampling_params(s, body, max_token_keys=self.max_token_keys, stop_keys=self.stop_keys)
            # context-budget clamp (backend-neutral): if the prompt already exceeds the
            # per-sid budget, short-circuit with an empty length-capped turn.
            if s.max_context_tokens > 0:
                remaining = s.max_context_tokens - len(prompt_ids)
                if remaining <= 0:
                    self.logger.warning(
                        "[%s] sid=%s prompt exceeds max_context_tokens (%d >= %d)",
                        self.log_prefix,
                        sid,
                        len(prompt_ids),
                        s.max_context_tokens,
                    )
                    turn = TurnRecord(prompt_ids=list(prompt_ids), output_ids=[], finish_reason="length")
                else:
                    sampling_params["max_new_tokens"] = min(
                        int(sampling_params.get("max_new_tokens", remaining)), remaining
                    )
                    turn = await self.backend.generate(
                        prompt_ids=prompt_ids, sampling_params=sampling_params, session_id=sid
                    )
            else:
                turn = await self.backend.generate(
                    prompt_ids=prompt_ids, sampling_params=sampling_params, session_id=sid
                )

            parsed = self.renderer.parse(turn.output_ids, tools_schema=tools_schema)
            reply = self._build_reply(parsed, turn.finish_reason, translated, tools_schema)
            turn = dataclasses.replace(turn, ill_formed=parsed.ill_formed)

            in_tok, out_tok = len(prompt_ids), len(turn.output_ids)
            stream = body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", "")

            # Flush the response before recording the trajectory: a client that
            # disconnected during generation makes _respond raise here, and we
            # must not record a turn the client never received.
            try:
                response = await self._respond(request, body, reply, in_tok, out_tok, stream)
            except (ConnectionResetError, asyncio.CancelledError) as e:
                self.logger.warning(
                    "[%s] sid=%s client disconnected before response flush: %s after %.1fs",
                    self.log_prefix,
                    sid,
                    type(e).__name__,
                    time.monotonic() - t0,
                )
                if isinstance(e, asyncio.CancelledError):
                    raise
                return web.Response(status=499, text="client disconnected")

            self._run_debug_callback(
                sid,
                translated,
                tools_schema,
                reply.manager_message,
                turn,
            )

            self.manager.record_turn(
                sid,
                turn=turn,
                prompt_messages=translated,
                response_message=reply.manager_message,
                metadata={"sid": sid},
            )
            return response
        finally:
            self.inflight.get(sid, set()).discard(task)


def sid_from_bearer(request: web.Request) -> str | None:
    """sid from the Authorization: Bearer <sid> header, or None if absent."""
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return None


def sid_from_body(body: dict | None) -> str | None:
    """sid from the OpenAI-shape body (metadata.session_id / user), or None."""
    if not body:
        return None
    metadata = body.get("metadata")
    if isinstance(metadata, dict) and metadata.get("session_id"):
        return str(metadata["session_id"])
    if body.get("user"):
        return str(body["user"])
    return None


def _sampling_params(session: Any, body: dict, *, max_token_keys: tuple[str, ...], stop_keys: tuple[str, ...]) -> dict:
    """Map a wire request body to the canonical sampling-params dict handed to the
    backend. Backend-neutral: each backend translates keys like ``max_new_tokens`` to
    its own wire names."""
    sp: dict[str, Any] = {
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "no_stop_trim": True,
        "max_new_tokens": 4096,
        **(session.sampling_defaults or {}),
    }

    for key in max_token_keys:
        if body.get(key) is not None:
            sp["max_new_tokens"] = min(int(sp.get("max_new_tokens", body[key])), int(body[key]))
            break

    for src_k, dst_k in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
        if src_k in body:
            sp[dst_k] = body[src_k]

    for key in stop_keys:
        if body.get(key):
            sp["stop"] = body[key]
            break

    return sp


async def _health(request: web.Request) -> web.Response:
    """Handler for /healthz and /v1/models readiness probes."""
    return web.json_response({"ok": True})
