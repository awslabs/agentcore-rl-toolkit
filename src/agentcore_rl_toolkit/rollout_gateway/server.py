"""HTTP serving and loop dispatch for an assembled rollout gateway.

``GatewayServer`` owns aiohttp resources on its caller's loop.
``ThreadedGatewayServer`` supplies a separate loop for deployments that keep
HTTP serving independent of rollout orchestration. Its synchronous start and
shutdown interface remains available to existing integrations.

The threaded serving mechanics originate from slime's aiohttp_threaded.py
(baseline commit ``fa3c990a``; see NOTICE).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import TypeVar

from aiohttp import web
from aiohttp.web_log import AccessLogger

logger = logging.getLogger(__name__)
T = TypeVar("T")


def url_host(host: str) -> str:
    """Bracket IPv6 literals in an advertised HTTP authority."""
    return f"[{host.strip('[]')}]" if ":" in host else host


class FilteredAccessLogger(AccessLogger):
    """Log only failures and slow requests; healthy fast traffic is noise."""

    SLOW_THRESHOLD_SEC = 120.0

    def log(self, request, response, time):
        if request.method == "HEAD":
            return
        if response.status == 200 and time <= self.SLOW_THRESHOLD_SEC:
            return
        super().log(request, response, time)


class GatewayServer:
    """Own an aiohttp listener on one loop; construction only stores configuration."""

    def __init__(self, gateway, *, host: str, port: int = 0, public_host: str | None = None):
        self.gateway = gateway
        self.host = host
        self.port = port
        self.public_host = public_host or host
        self._runner: web.AppRunner | None = None

    @property
    def base_url(self) -> str:
        return f"http://{url_host(self.public_host)}:{self.port}/v1"

    async def start(self) -> None:
        self._runner = web.AppRunner(self.gateway.app, handler_cancellation=True, access_log_class=FilteredAccessLogger)
        await self._runner.setup()
        await web.TCPSite(self._runner, self.host, self.port).start()
        self.port = self._runner.addresses[0][1]

    async def close(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()


class ThreadedGatewayServer:
    """Supply a serving thread, with synchronous startup and shutdown.

    ``base_url`` includes ``/v1`` for compatibility with existing integrations.
    ``call`` submits lifecycle operations without blocking the caller's loop.
    """

    def __init__(
        self,
        gateway,
        *,
        host: str,
        port: int = 0,
        startup_timeout: float = 120.0,
        public_host: str | None = None,
    ):
        self.gateway = gateway
        self.host = host
        self.port = port
        self.startup_timeout = startup_timeout
        self._server = GatewayServer(gateway, host=host, port=port, public_host=public_host)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return self._server.base_url

    def start(self) -> None:
        started = threading.Event()
        startup_error: list[BaseException] = []

        def _serve() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._server.start())
                self.port = self._server.port
                self._loop = loop
                loop.call_soon(started.set)
                loop.run_forever()
            except BaseException as exc:
                startup_error.append(exc)
                started.set()

        self._thread = threading.Thread(target=_serve, name="rollout-gateway", daemon=True)
        self._thread.start()
        if not started.wait(timeout=self.startup_timeout):
            raise TimeoutError(f"Rollout gateway did not start within {self.startup_timeout}s")
        if startup_error:
            raise RuntimeError(f"Rollout gateway failed to start on {self.host}:{self.port}") from startup_error[0]
        logger.info("Rollout gateway serving at %s", self.base_url)

    async def call(self, operation: Callable[[], Awaitable[T]]) -> T:
        loop = self._loop
        if loop is None or not loop.is_running():
            raise RuntimeError("gateway server is not ready")
        if asyncio.get_running_loop() is loop:
            return await operation()
        coroutine = operation()
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        except BaseException:
            coroutine.close()
            raise
        return await asyncio.wrap_future(future)

    def shutdown(self, *, join_timeout: float = 10.0) -> None:
        """Close HTTP serving with the existing best-effort shutdown behavior."""
        if self._loop is None:
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self._server.close(), self._loop)
            future.result(timeout=10)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=join_timeout)
        self._loop = None
