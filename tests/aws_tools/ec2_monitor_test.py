#!/usr/bin/env python
"""Unit tests for the EC2 -> agent-session instance-id monitor: polling bookkeeping,
the background loop, and stats logging. EC2 lookups and dynamodb writes are faked.
"""

import asyncio
import unittest
from unittest import IsolatedAsyncioTestCase, mock

from agentcore_rl_toolkit.aws_tools.ec2_monitor import EC2Monitor, start_ec2_monitor

CAPACITY_PROVIDER_ARN = "arn:aws:bedrock-agentcore:us-west-2:1234:capacity-provider/cp-test"
LOGGER = "agentcore_rl_toolkit.aws_tools.ec2_monitor"


def stats_lines(logs) -> list[str]:
    """The periodic stats lines out of an ``assertLogs`` capture, in order."""
    return [r.getMessage() for r in logs.records if "stats" in r.getMessage()]


def instance(instance_id: str, session_id: str | None) -> dict:
    """A DescribeInstances-shaped instance, tagged like an AgentCore host."""
    tags = [{"Key": "aws:ec2:managed-launch", "Value": "agentcore-runtime-instance"}]
    if session_id is not None:
        tags.append({"Key": "bedrock-agentcore:runtime-session-id", "Value": session_id})
    return {"InstanceId": instance_id, "Tags": tags}


def patch_instances(instances: list[dict]):
    """Make every poll see ``instances`` instead of calling EC2."""
    return mock.patch(
        "agentcore_rl_toolkit.aws_tools.ec2_monitor.find_running_instances",
        mock.AsyncMock(return_value=instances),
    )


class RecordingMonitor(EC2Monitor):
    """An EC2Monitor whose dynamodb write is recorded, not performed. ``fail`` names
    the sessions whose write is reported as errored."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writes: list[dict[str, str]] = []
        self.fail: set[str] = set()

    async def _write(self, updates):
        if not updates:  # mirrors the real method's early return
            return {}
        self.writes.append(dict(updates))
        return {s: i for s, i in updates.items() if s not in self.fail}


def make_monitor(**kwargs) -> RecordingMonitor:
    return RecordingMonitor(CAPACITY_PROVIDER_ARN, "sessions_table", **kwargs)


class EC2MonitorPollTest(IsolatedAsyncioTestCase):
    async def test_stamps_each_session_once(self):
        # A session already stamped with the same instance is never written again.
        monitor = make_monitor()
        instances = [instance("i-1", "verl_a"), instance("i-2", "verl_b")]

        with patch_instances(instances):
            first = await monitor.poll_once()
            second = await monitor.poll_once()

        self.assertEqual(first, {"verl_a": "i-1", "verl_b": "i-2"})
        self.assertEqual(second, {})
        self.assertEqual(monitor.writes, [{"verl_a": "i-1", "verl_b": "i-2"}])
        self.assertEqual(monitor.stats()["sessions_stamped"], 2)

    async def test_untagged_instances_are_ignored(self):
        monitor = make_monitor()
        with patch_instances([instance("i-1", None), instance("i-2", "verl_a")]):
            written = await monitor.poll_once()

        self.assertEqual(written, {"verl_a": "i-2"})

    async def test_terminated_instances_are_forgotten(self):
        # Bookkeeping is pruned to live instances, so memory tracks live capacity
        # rather than the run's total session count.
        monitor = make_monitor()
        with patch_instances([instance("i-1", "verl_a")]):
            await monitor.poll_once()
        self.assertEqual(monitor.stats()["sessions_tracked"], 1)

        with patch_instances([]):
            await monitor.poll_once()
        self.assertEqual(monitor.stats()["sessions_tracked"], 0)

    async def test_new_instance_for_same_session_is_restamped(self):
        monitor = make_monitor()
        with patch_instances([instance("i-1", "verl_a")]):
            await monitor.poll_once()
        with patch_instances([instance("i-2", "verl_a")]):
            written = await monitor.poll_once()

        self.assertEqual(written, {"verl_a": "i-2"})

    async def test_errored_write_is_reattempted_while_the_instance_runs(self):
        # A write only counts as done when it succeeds.
        monitor = make_monitor()
        monitor.fail = {"verl_a"}

        with patch_instances([instance("i-1", "verl_a")]):
            self.assertEqual(await monitor.poll_once(), {})
            self.assertEqual(await monitor.poll_once(), {})
            monitor.fail = set()
            self.assertEqual(await monitor.poll_once(), {"verl_a": "i-1"})
            self.assertEqual(await monitor.poll_once(), {})  # and not written twice

        self.assertEqual(len(monitor.writes), 3)


class EC2MonitorLoopTest(IsolatedAsyncioTestCase):
    async def test_run_forever_survives_a_failing_poll(self):
        # Instance ids are only diagnostics, so a throttled EC2 call must not take
        # the monitor down.
        monitor = make_monitor(poll_interval=0)
        calls = []
        polled_again = asyncio.Event()

        async def poll_once():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("throttled")
            polled_again.set()

        with mock.patch.object(monitor, "poll_once", poll_once):
            with self.assertLogs(LOGGER, "INFO") as logs:
                await monitor.start()
                await asyncio.wait_for(polled_again.wait(), timeout=5)
                await monitor.stop()

        self.assertGreaterEqual(len(calls), 2)
        # Read off the log because the counters are zeroed after every line.
        self.assertIn("'poll_errors': 1", stats_lines(logs)[0])
        self.assertIn("RuntimeError('throttled')", stats_lines(logs)[0])

    async def test_start_is_idempotent_and_stop_cancels(self):
        monitor = make_monitor(poll_interval=3600)
        with patch_instances([]):
            await monitor.start()
            task = monitor._task
            await monitor.start()
            self.assertIs(monitor._task, task)
            await monitor.stop()
            await monitor.stop()

        assert task is not None
        self.assertTrue(task.cancelled())

    async def test_context_manager_runs_and_stops_the_loop(self):
        monitor = make_monitor(poll_interval=3600)
        with patch_instances([instance("i-1", "verl_a")]):
            async with monitor as entered:
                self.assertIs(entered, monitor)
                task = monitor._task
                assert task is not None
                while not monitor.writes:  # let the loop reach its first poll
                    await asyncio.sleep(0)

        self.assertTrue(task.done())
        self.assertEqual(monitor.writes, [{"verl_a": "i-1"}])

    async def test_stats_are_logged_once_per_stats_interval(self):
        monitor = make_monitor(poll_interval=0, stats_interval=3600)
        with patch_instances([instance("i-1", "verl_a")]):
            with self.assertLogs(LOGGER, "INFO") as logs:
                await monitor.start()
                while monitor.stats()["polls"] < 3:
                    await asyncio.sleep(0)
                await monitor.stop()

        # Many polls, but one stats line: the first, then not again for an hour.
        self.assertEqual(len(stats_lines(logs)), 1)
        self.assertIn("'sessions_stamped': 1", stats_lines(logs)[0])

    async def test_stats_reset_on_every_logging_step(self):
        # Each line covers only the interval since the previous one.
        monitor = make_monitor(poll_interval=0, stats_interval=0)
        with patch_instances([instance("i-1", "verl_a")]):
            with self.assertLogs(LOGGER, "INFO") as logs:
                await monitor.start()
                while len(stats_lines(logs)) < 2:
                    await asyncio.sleep(0)
                await monitor.stop()

        first, second = stats_lines(logs)[:2]
        # The stamp count belongs to the first line only; the gauges keep reporting it.
        self.assertIn("'polls': 1", first)
        self.assertIn("'sessions_stamped': 1", first)
        self.assertIn("'polls': 1", second)
        self.assertIn("'sessions_stamped': 0", second)
        self.assertIn("'instances': 1", second)
        self.assertIn("'sessions_tracked': 1", second)


class StartEC2MonitorTest(IsolatedAsyncioTestCase):
    async def test_helper_starts_polling_on_the_callers_loop(self):
        writes = []

        async def fake_write(self, updates):
            if not updates:  # mirrors the real method's early return
                return {}
            writes.append(dict(updates))
            return dict(updates)

        with patch_instances([instance("i-1", "verl_a")]), mock.patch.object(EC2Monitor, "_write", fake_write):
            monitor = await start_ec2_monitor(CAPACITY_PROVIDER_ARN, "sessions_table", poll_interval=0)
            try:
                while not writes:
                    await asyncio.sleep(0)
            finally:
                await monitor.stop()

        self.assertEqual(writes[0], {"verl_a": "i-1"})


if __name__ == "__main__":
    unittest.main()
