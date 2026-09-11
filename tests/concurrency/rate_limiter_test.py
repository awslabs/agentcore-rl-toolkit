#!/usr/bin/env python
"""Unit tests for the session-creation rate limiter: the :class:`RateLimiter` protocol,
:class:`RayRateLimiter` forwarding across the actor boundary, and that a burst of
callers is smeared rather than let through together. No Ray, no AWS.
"""

import asyncio
import time
import unittest
from unittest import IsolatedAsyncioTestCase

from agentcore_rl_toolkit.concurrency.rate_limiter import ACRRateLimiter, RateLimiter
from agentcore_rl_toolkit.concurrency.ray_adapters import RayRateLimiter


class ProtocolTest(unittest.TestCase):
    def test_the_toolkit_limiter_satisfies_the_protocol(self):
        self.assertIsInstance(ACRRateLimiter(25), RateLimiter)

    def test_the_ray_adapter_satisfies_it_without_a_cluster(self):
        # Must hold before any actor exists.
        self.assertIsInstance(RayRateLimiter(None), RateLimiter)  # type: ignore[arg-type]


class RayRateLimiterTest(IsolatedAsyncioTestCase):
    async def test_it_forwards_wait_async_to_the_actor(self):
        class FakeObjectRef:
            def __await__(self):
                async def resolve():
                    calls.append("awaited")

                return resolve().__await__()

        class FakeMethod:
            def remote(self):
                calls.append("remote")
                return FakeObjectRef()

        class FakeActor:
            wait_async = FakeMethod()

        calls: list[str] = []
        await RayRateLimiter(FakeActor()).wait_async()  # type: ignore[arg-type]
        # Awaited, not fired and forgotten.
        self.assertEqual(calls, ["remote", "awaited"])


class ACRRateLimiterTest(IsolatedAsyncioTestCase):
    """Timed against the real clock because the vendored limiter reads ``time.time``
    and sleeps internally. The rate is high and the bounds loose, so the assertions are
    about smearing having happened at all, not precise pacing."""

    RATE = 50  # 20ms between grants
    INTERVAL = 1 / RATE

    async def test_sequential_callers_are_paced(self):
        limiter = ACRRateLimiter(self.RATE)
        started = time.monotonic()
        for _ in range(6):
            await limiter.wait_async()
        elapsed = time.monotonic() - started
        # The first caller is free, so five intervals is the floor.
        self.assertGreaterEqual(elapsed, 5 * self.INTERVAL * 0.9)

    async def test_a_simultaneous_burst_is_smeared_not_let_through(self):
        limiter = ACRRateLimiter(self.RATE)
        started = time.monotonic()
        await asyncio.gather(*(limiter.wait_async() for _ in range(6)))
        elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, 5 * self.INTERVAL * 0.9)

    async def test_an_idle_limiter_does_not_delay_the_next_caller(self):
        limiter = ACRRateLimiter(self.RATE)
        await limiter.wait_async()
        await asyncio.sleep(self.INTERVAL * 3)
        started = time.monotonic()
        await limiter.wait_async()
        self.assertLess(time.monotonic() - started, self.INTERVAL)


class ACRRateLimiterLoopTest(unittest.TestCase):
    """Driven with its own loops, so it cannot live in the async test case above."""

    def test_it_survives_a_new_event_loop(self):
        # One limiter outlives any single asyncio.run(), and its internal lock is
        # bound to a loop, so it has to notice when that loop is gone.
        limiter = ACRRateLimiter(50)
        asyncio.run(limiter.wait_async())
        asyncio.run(limiter.wait_async())  # must not raise "attached to a different loop"


if __name__ == "__main__":
    unittest.main()
