"""Address formatting and cross-thread gateway dispatch tests."""

import asyncio
import threading

import pytest

from agentcore_rl_toolkit.rollout_gateway import ThreadedGatewayServer
from agentcore_rl_toolkit.rollout_gateway.server import url_host

from .test_gateway_integration import make_gateway


@pytest.mark.parametrize(
    "host,expected",
    [
        ("127.0.0.1", "127.0.0.1"),
        ("10.4.132.156", "10.4.132.156"),
        ("gateway.internal", "gateway.internal"),
        ("::1", "[::1]"),
        ("2001:db8::1", "[2001:db8::1]"),
    ],
)
def test_url_host_brackets_ipv6(host, expected):
    assert url_host(host) == expected


def test_ipv6_public_host_yields_parseable_base_url():
    """IPv6 brackets distinguish address colons from the port separator."""
    from urllib.parse import urlparse

    url = f"http://{url_host('2001:db8::1')}:1234"
    parsed = urlparse(url)
    assert parsed.hostname == "2001:db8::1"
    assert parsed.port == 1234


@pytest.mark.asyncio
async def test_cancellation_propagates_to_dispatched_operation():
    gateway, _ = make_gateway([])
    server = ThreadedGatewayServer(gateway, host="127.0.0.1")
    await asyncio.to_thread(server.start)
    entered = threading.Event()
    cancelled = threading.Event()

    async def operation():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    call = asyncio.create_task(server.call(operation))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert await asyncio.to_thread(cancelled.wait, 5)
    finally:
        await asyncio.to_thread(server.shutdown)
