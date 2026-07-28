"""Process-level RolloutGateway host for verl AgentLoopWorker processes.

verl instantiates agent loop classes per trajectory (``hydra.utils.instantiate``
in ``AgentLoopWorker._run_agent_loop``), so the gateway — an aiohttp server plus
its ``VerlSamplingBackend`` — must live at process scope: created lazily by the
first ``AgentCoreAgentLoop.__init__`` in the process, then shared by every
subsequent instantiation.

The serving shape is slime's ``aiohttp_threaded.run_app_in_thread`` (the same
code family this gateway's adapters were vendored from): a daemon thread with
its own event loop running ``web.AppRunner`` + ``web.TCPSite`` — NOT
``web.run_app``, which blocks and installs main-thread-only signal handlers.
``handler_cancellation=True`` makes a client disconnect cancel the in-flight
handler coroutine so the backend's abort path fires instead of leaving an
orphaned generate racing the engine.

The advertised base_url uses the Ray node IP (uni-agent's GatewayActor pattern);
ACR containers must be able to reach trainer CPU nodes on the gateway port.
"""

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Any

from aiohttp import web
from aiohttp.web_log import AccessLogger
from verl.utils.net_utils import is_valid_ipv6_address

from agentcore_rl_toolkit.rollout_gateway import HfTemplateRenderer
from agentcore_rl_toolkit.rollout_gateway.gateway import RolloutGateway

from .sampling_backend import VerlSamplingBackend

logger = logging.getLogger(__name__)


class FilteredAccessLogger(AccessLogger):
    """Log only errors and slow requests; suppress HEAD and fast-200 noise."""

    SLOW_THRESHOLD_SEC = 120.0

    def log(self, request, response, time):
        if request.method == "HEAD":
            return
        if response.status == 200 and time <= self.SLOW_THRESHOLD_SEC:
            return
        super().log(request, response, time)


@dataclass(frozen=True)
class GatewayHandle:
    """The per-process gateway singleton: the gateway, its verl-backed sampling
    backend, and the base_url ACR agents should dial."""

    gateway: RolloutGateway
    # Same object as gateway.backend, but narrowed: that one is typed as the
    # SamplingBackend protocol, which has no pop_extra_fields (VerlSamplingBackend
    # only, for verl's staleness tags).
    backend: VerlSamplingBackend
    base_url: str
    _loop: asyncio.AbstractEventLoop
    _runner: web.AppRunner
    _thread: threading.Thread

    def stop(self) -> None:
        """Tear down the serving thread (tests only; in production the daemon
        thread dies with the worker process)."""

        async def _shutdown() -> None:
            await self._runner.cleanup()

        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), self._loop).result(timeout=10)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


_LOCK = threading.Lock()
_HANDLE: GatewayHandle | None = None


def _node_ip() -> str:
    """The Ray node IP when inside a Ray worker; localhost otherwise (tests).

    Returned bare (unbracketed), the form ``bind()`` wants; ``_url_host`` puts it
    in URL form. Same split as uni-agent's GatewayActor."""
    try:
        import ray.util  # type: ignore[import-not-found]

        ip = ray.util.get_node_ip_address()
        # bracket-strip mirrors uni-agent's IPv6 handling
        return ip.strip("[]")
    except Exception:
        return "127.0.0.1"


def _url_host(host: str) -> str:
    """``host`` in the authority form a URL needs: an IPv6 literal must be
    bracketed, or its own colons make the ``:port`` suffix ambiguous and parsers
    reject the address (``http://2001:db8::1:8080``). Mirrors verl's own
    convention for advertised server addresses (see sglang/vllm rollout servers)."""
    return f"[{host}]" if is_valid_ipv6_address(host) else host


def _serve_in_thread(app: web.Application, host: str, port: int, start_timeout: float = 15.0):
    """Start ``app`` on a daemon thread with its own event loop; block until it
    is listening. Returns (loop, runner, bound_port, thread)."""
    started = threading.Event()
    box: dict[str, Any] = {}
    err_box: list[BaseException] = []

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            runner = web.AppRunner(app, handler_cancellation=True, access_log_class=FilteredAccessLogger)
            loop.run_until_complete(runner.setup())
            site = web.TCPSite(runner, host, port)
            loop.run_until_complete(site.start())
            bound_port = port
            for sock in site._server.sockets:  # type: ignore[union-attr]
                bound_port = sock.getsockname()[1]
                break
            box["loop"], box["runner"], box["port"] = loop, runner, bound_port
            started.set()
            loop.run_forever()
        except BaseException as e:  # pragma: no cover
            err_box.append(e)
            started.set()
            raise

    thread = threading.Thread(target=_run, name="agentcore-rollout-gateway", daemon=True)
    thread.start()
    if not started.wait(timeout=start_timeout):
        raise RuntimeError("rollout gateway server did not start within timeout")
    if err_box:
        raise err_box[0]
    return box["loop"], box["runner"], box["port"], thread


def get_or_start_gateway(
    *,
    server_manager: Any,
    tokenizer: Any,
    host: str = "0.0.0.0",
    port: int = 0,
    public_host: str | None = None,
    adapters: list[str] | None = None,
    max_turns_per_sid: int | None = None,
    fork_threshold_tokens: int | None = None,
) -> GatewayHandle:
    """Lazily create (or return) this process's RolloutGateway singleton.

    ``port=0`` auto-assigns — required when verl round-robins multiple
    AgentLoopWorkers onto one node. ``public_host`` overrides the advertised
    host for NAT setups where the Ray node IP is not ACR-reachable.
    """
    global _HANDLE
    with _LOCK:
        if _HANDLE is not None:
            return _HANDLE

        adapter_names: list = list(adapters) if adapters else ["openai", "anthropic"]
        backend = VerlSamplingBackend(server_manager)
        gateway = RolloutGateway(
            backend=backend,
            renderer=HfTemplateRenderer(tokenizer),
            tokenizer=tokenizer,
            adapters=adapter_names,
            max_turns_per_sid=max_turns_per_sid,
            fork_threshold_tokens=fork_threshold_tokens,
        )
        loop, runner, bound_port, thread = _serve_in_thread(gateway.app, host, port)
        base_url = f"http://{_url_host(public_host or _node_ip())}:{bound_port}"
        _HANDLE = GatewayHandle(
            gateway=gateway,
            backend=backend,
            base_url=base_url,
            _loop=loop,
            _runner=runner,
            _thread=thread,
        )
        logger.info("rollout gateway serving at %s (adapters=%s)", base_url, adapters or ["openai", "anthropic"])
        return _HANDLE


def _reset_for_tests() -> None:
    global _HANDLE
    with _LOCK:
        if _HANDLE is not None:
            _HANDLE.stop()
            _HANDLE = None


__all__ = ["GatewayHandle", "get_or_start_gateway"]
