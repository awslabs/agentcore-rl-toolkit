"""``RolloutGateway`` — assembles the capture layer into one serving unit.

Wires a tokenizer + renderer + sampling backend + one or more protocol adapters onto
a single aiohttp app, all sharing one :class:`TrajectoryManager` (so per-sid trees are
visible regardless of which wire protocol a turn arrived on). Typically run as a thread
inside the trainer process, or as a standalone process.

Session state is held in-memory, keyed by ``session_id``. Session identity rides in the
api-key / Bearer slot (the adapters resolve it via ``sid_from_bearer``); the agent's
``base_url`` is a plain fixed gateway address — there are no per-session URLs.
"""

import logging
from typing import Any

from aiohttp import web

from .adapters.anthropic import AnthropicAdapter
from .adapters.common import BaseAdapter, _health
from .adapters.openai import OpenAIAdapter
from .render import Renderer
from .sampling_backends.base import SamplingBackend
from .trace import BaseTrace, TraceRecord
from .trajectory import TrajectoryManager

logger = logging.getLogger(__name__)

# name -> adapter class for the `adapters=[...]` convenience arg
_ADAPTER_REGISTRY: dict[str, type[BaseAdapter]] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
}


class RolloutGateway:
    """One HTTP surface capturing token-level trajectories across protocols.

    Args:
        backend: the sampling backend (token-in / token-out).
        renderer: the tokenization authority (render prompts / parse responses).
        tokenizer: HF tokenizer used only to decode trained tails into ``.response``
            in ``finish_session`` (the manager itself is tokenizer-free). Optional.
        adapters: which wire protocols to mount — names ("openai", "anthropic") or
            ``BaseAdapter`` subclasses. Defaults to both.
        fork_threshold_tokens / max_turns_per_sid / debug_callback: forwarded to the
            shared ``TrajectoryManager`` / adapters.
    """

    def __init__(
        self,
        *,
        backend: SamplingBackend,
        renderer: Renderer,
        tokenizer=None,
        adapters: list[str | type[BaseAdapter]] | None = None,
        fork_threshold_tokens: int | None = None,
        max_turns_per_sid: int | None = None,
        debug_callback: Any = None,
    ) -> None:
        self.backend = backend
        self.renderer = renderer
        self.tokenizer = tokenizer

        mgr_kwargs: dict[str, int] = {}
        if fork_threshold_tokens is not None:
            mgr_kwargs["fork_threshold_tokens"] = fork_threshold_tokens
        self.manager = TrajectoryManager(**mgr_kwargs)

        # one aiohttp app shared by all adapters; register health once here.
        self.app = web.Application(client_max_size=64 * 1024 * 1024)
        self.app.router.add_get("/healthz", _health)
        self.app.router.add_get("/v1/models", _health)

        adapter_specs = adapters if adapters is not None else ["openai", "anthropic"]
        self.adapters: list[BaseAdapter] = []
        for spec in adapter_specs:
            cls = _ADAPTER_REGISTRY[spec] if isinstance(spec, str) else spec
            adapter = cls(
                backend=backend,
                renderer=renderer,
                tokenizer=tokenizer,
                max_turns_per_sid=max_turns_per_sid,
                debug_callback=debug_callback,
                manager=self.manager,  # SHARED across adapters -> one tree per sid
                app=self.app,  # SHARED app -> all routes on one server
            )
            self.adapters.append(adapter)

    # -- session lifecycle (delegates to the shared manager + all adapters) ---

    def create_session(
        self,
        sid: str,
        *,
        sampling_defaults: dict | None = None,
        max_context_tokens: int = 0,
    ) -> None:
        """Register a fresh per-sid session. Registers on every adapter so a turn on
        any protocol under this sid finds its Session state."""
        for adapter in self.adapters:
            adapter.open_session(sid, sampling_defaults=sampling_defaults, max_context_tokens=max_context_tokens)

    async def finish_session(
        self,
        sid: str,
        *,
        base_sample: BaseTrace | None = None,
        reward: float = 0.0,
        extra_metadata: dict | None = None,
        wait_timeout: float = 5.0,
    ) -> list[TraceRecord]:
        """Drain the sid's trajectory into TraceRecords and consume it.

        The tree lives in the shared manager, so we drain in-flight tasks on every
        adapter, then linearize once via the first adapter (they share ``self.manager``).
        """
        for adapter in self.adapters:
            await adapter.shutdown_session(sid, wait_timeout=wait_timeout)
        primary = self.adapters[0]
        return await primary.finish_session(
            sid,
            base_sample=base_sample,
            reward=reward,
            extra_metadata=extra_metadata,
            wait_timeout=wait_timeout,
        )

    async def drop_session(self, sid: str, *, wait_timeout: float = 5.0) -> None:
        for adapter in self.adapters:
            await adapter.shutdown_session(sid, wait_timeout=wait_timeout)
            adapter.store.pop(sid, None)
        self.manager.drop_session(sid)

    # -- serving --------------------------------------------------------------

    def run(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Serve the app (blocking). For the in-thread deployment, run this in a
        dedicated thread with its own event loop."""
        web.run_app(self.app, host=host, port=port)


__all__ = ["RolloutGateway"]
