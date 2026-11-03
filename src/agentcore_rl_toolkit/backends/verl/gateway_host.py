"""One gateway handle per verl AgentLoopWorker, with loop-owned capture state."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import ray.util

from agentcore_rl_toolkit.rollout_gateway import HfTemplateRenderer, TraceRecord
from agentcore_rl_toolkit.rollout_gateway.gateway import RolloutGateway
from agentcore_rl_toolkit.rollout_gateway.server import ThreadedGatewayServer

from .sampling_backend import VerlSamplingBackend


@dataclass(frozen=True)
class GatewayResult:
    records: list[TraceRecord]
    num_turns: int
    extra_fields: dict[str, Any]


@dataclass(frozen=True)
class GatewayHandle:
    """Own serving and dispatch complete session operations to its loop."""

    gateway: RolloutGateway
    backend: VerlSamplingBackend
    _server: ThreadedGatewayServer

    @property
    def base_url(self) -> str:
        # verl adds /v1 when building the agent payload.
        return self._server.base_url.removesuffix("/v1")

    async def create_session(self, sid: str, **kwargs) -> None:
        async def create():
            self.gateway.create_session(sid, **kwargs)

        await self._server.call(create)

    async def finish_session(self, sid: str, **kwargs) -> GatewayResult:
        async def finish():
            num_turns = self.gateway.manager.turn_count(sid)
            records = await self.gateway.finish_session(sid, **kwargs)
            extra = self.backend.pop_extra_fields(sid)
            return GatewayResult(records, num_turns, extra)

        return await self._server.call(finish)

    async def drop_session(self, sid: str) -> None:
        async def drop():
            await self.gateway.drop_session(sid)

        await self._server.call(drop)

    async def session_turn_counts(self, sids) -> dict[str, int]:
        async def snapshot():
            return {sid: self.gateway.manager.turn_count(sid) for sid in sids}

        return await self._server.call(snapshot)

    def stop(self) -> None:
        self._server.shutdown(join_timeout=5.0)


_LOCK = threading.Lock()
_HANDLE: GatewayHandle | None = None


def _node_ip() -> str:
    return ray.util.get_node_ip_address().strip("[]")


def get_or_start_gateway(
    *,
    server_manager: Any,
    tokenizer: Any,
    host: str = "0.0.0.0",
    port: int = 0,
    public_host: str | None = None,
    adapters: list[str] | None = None,
    max_turns_per_sid: int | None = None,
    history_mode: str = "tree",
    chat_template_kwargs: dict | None = None,
    linear_on_nonlinear: str = "reset",
) -> GatewayHandle:
    """Lazily create or return this process's gateway singleton."""
    global _HANDLE
    with _LOCK:
        if _HANDLE is not None:
            return _HANDLE

        adapter_names: list = list(adapters) if adapters else ["openai", "anthropic"]
        backend = VerlSamplingBackend(server_manager)
        gateway = RolloutGateway(
            backend=backend,
            renderer=HfTemplateRenderer(tokenizer, chat_template_kwargs=chat_template_kwargs),
            tokenizer=tokenizer,
            adapters=adapter_names,
            max_turns_per_sid=max_turns_per_sid,
            history_mode=history_mode,
            linear_on_nonlinear=linear_on_nonlinear,
        )
        server = ThreadedGatewayServer(
            gateway,
            host=host,
            port=port,
            public_host=public_host or _node_ip(),
            startup_timeout=15.0,
        )
        server.start()
        _HANDLE = GatewayHandle(gateway, backend, server)
        return _HANDLE


def _reset_for_tests() -> None:
    global _HANDLE
    with _LOCK:
        if _HANDLE is not None:
            _HANDLE.stop()
            _HANDLE = None


__all__ = ["GatewayHandle", "get_or_start_gateway"]
