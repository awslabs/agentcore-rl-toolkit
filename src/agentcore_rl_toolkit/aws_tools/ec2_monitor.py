"""A long-lived poller that stamps ``ec2_instance_id`` onto agent sessions.

Every AgentCore session runs on an EC2 instance tagged with its
``bedrock-agentcore:runtime-session-id``, and recording which instance served a
session is what makes host-level forensics possible after a run (CloudWatch
metrics per instance, capacity/throttling analysis, ssh into a stuck host).

The obvious place to record it is each session's own setup path, one
``find_running_instances`` call per session -- but that scales with the number of
rollouts: at hundreds of concurrent sessions those ``DescribeInstances`` calls
exhaust the EC2 API rate limit, and the rollout then fails for a reason that has
nothing to do with the rollout.

This module inverts the lookup. One monitor polls ``DescribeInstances`` for the
whole capacity provider on a fixed interval -- a constant API cost, independent of
session count -- reads the session id off each instance's tags, and writes
``ec2_instance_id`` to those sessions' items in the agent-session table. The trade
is timing: a session is stamped within one poll interval of starting rather than
synchronously during setup.

Every running instance is worth monitoring, so the write is an unconditional
upsert: a session whose item does not exist yet gets one holding just its instance
id, and the rest of its metadata lands on top of that row whenever its owner writes
it.

Nothing ever calls into the monitor -- it writes to dynamodb and logs its own stats
once a minute -- so it needs no Ray actor to be addressable from elsewhere in the
cluster. It is a plain asyncio background task, hosted either way:

* inside a training run, on the host's event loop -- :func:`start_ec2_monitor`,
  which the trainer's entrypoint starts alongside the cluster-wide actors;
* as its own process -- a standalone script around :class:`EC2Monitor`, for
  stamping a run already in flight, or when the hosting process's credentials are
  not authorized for ``ec2:DescribeInstances`` (the HyperPod execution role is not;
  see :func:`~.ec2_tools.get_current_instance_type`). An unauthorized poll is
  logged and retried, never fatal, so the in-run task is safe to leave enabled
  either way.
"""

import asyncio
import logging
import time
from contextlib import suppress
from typing import Any

from agentcore_rl_toolkit.aws_tools.boto3_tools import get_aioboto3_session
from agentcore_rl_toolkit.aws_tools.dynamodb_tools import update_dict
from agentcore_rl_toolkit.aws_tools.ec2_tools import find_running_instances, instance_to_session

logger = logging.getLogger(__name__)

__all__ = ["EC2Monitor", "start_ec2_monitor", "DEFAULT_POLL_INTERVAL"]

DEFAULT_POLL_INTERVAL = 30.0
DEFAULT_STATS_INTERVAL = 60.0


def _new_stats() -> dict[str, Any]:
    """Zeroed counters for one reporting interval (see :meth:`EC2Monitor.stats`)."""
    return {
        "polls": 0,
        "poll_errors": 0,
        "write_errors": 0,
        "instances": 0,
        "sessions_stamped": 0,
        "last_error": None,
    }


class EC2Monitor:
    """Polls a capacity provider's instances and records them on their sessions.

    One instance of this class serves a whole run: call :meth:`start` once to
    spawn the polling loop as a background task on the caller's event loop (or
    drive :meth:`poll_once` yourself), and :meth:`stop` to cancel it. Both are
    idempotent. The monitor holds the only reference to its task, so the caller
    has to keep the monitor itself alive.

    A session is written at most once per instance it is seen on: the
    session -> instance map of already-written pairs is kept in memory and
    pruned to the instances still running on every poll, so it stays bounded by
    live capacity rather than growing with the run's total session count.
    """

    def __init__(
        self,
        capacity_provider_arn: str,
        dynamodb_table: str,
        *,
        session_prefix: str | None = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        stats_interval: float = DEFAULT_STATS_INTERVAL,
        region_name: str = "us-west-2",
        write_concurrency: int = 16,
    ):
        """
        ``session_prefix`` narrows the instances considered to sessions whose id
        starts with it (the agent loop's ids all start with ``verl_``); ``None``
        considers every instance the capacity provider launched.

        ``stats_interval`` is how often :meth:`run_forever` logs :meth:`stats`
        and then zeroes its counters -- the monitor's only routine output, since
        nothing queries it.

        ``region_name`` is the dynamodb region -- the EC2 region is taken from
        ``capacity_provider_arn``.
        """
        self.capacity_provider_arn = capacity_provider_arn
        self.dynamodb_table = dynamodb_table
        self.session_prefix = session_prefix
        self.poll_interval = poll_interval
        self.stats_interval = stats_interval
        self.region_name = region_name
        self.write_concurrency = write_concurrency

        # session_id -> instance_id, for pairs already written to dynamodb
        self._stamped: dict[str, str] = {}
        self._task: asyncio.Task | None = None
        self._stats: dict[str, Any] = _new_stats()

    # --- polling -----------------------------------------------------------

    async def poll_once(self) -> dict[str, str]:
        """Run one EC2 lookup and stamp every newly-seen session.

        Returns the ``session_id -> instance_id`` pairs written by this poll.
        """
        instances = await find_running_instances(self.capacity_provider_arn, self.session_prefix)

        live: dict[str, str] = {}
        for instance in instances:
            session_id = instance_to_session(instance)
            if session_id:
                live[session_id] = instance["InstanceId"]

        # Only sessions not yet stamped with this instance are worth a write.
        pending = {
            session_id: instance_id
            for session_id, instance_id in live.items()
            if self._stamped.get(session_id) != instance_id
        }
        written = await self._write(pending)

        # Forget sessions whose instances are gone; their state can no longer
        # change and keeping them would grow with the run.
        self._stamped = {s: i for s, i in self._stamped.items() if s in live}
        self._stamped.update(written)

        self._stats["polls"] += 1
        self._stats["instances"] = len(instances)
        self._stats["sessions_stamped"] += len(written)
        logger.debug(
            "ec2_monitor poll: %d instance(s), %d session(s) live, %d stamped, %d failed",
            len(instances),
            len(live),
            len(written),
            len(pending) - len(written),
        )
        return written

    async def _write(self, updates: dict[str, str]) -> dict[str, str]:
        """Write ``ec2_instance_id`` for each session, sharing one dynamodb client.

        Returns the pairs successfully written; a session whose write errored is
        simply left unstamped, so the next poll picks it up again for as long as
        its instance runs.
        """
        if not updates:
            return {}

        semaphore = asyncio.Semaphore(self.write_concurrency)
        session = await get_aioboto3_session()
        async with session.resource("dynamodb", region_name=self.region_name) as dynamodb:  # type: ignore
            table = await dynamodb.Table(self.dynamodb_table)

            async def write_one(session_id: str, instance_id: str):
                async with semaphore:
                    # An upsert: a session whose item does not exist yet is
                    # created here and filled in by its agent loop later.
                    return await update_dict(
                        table,
                        {"session_id": session_id},
                        {"ec2_instance_id": instance_id},
                    )

            results = await asyncio.gather(
                *(write_one(s, i) for s, i in updates.items()),
                return_exceptions=True,
            )

        written: dict[str, str] = {}
        for (session_id, instance_id), result in zip(updates.items(), results, strict=True):
            if isinstance(result, BaseException):
                self._stats["write_errors"] += 1
                logger.warning("ec2_monitor: failed to stamp %s: %r", session_id, result)
            else:
                written[session_id] = instance_id
        return written

    async def run_forever(self) -> None:
        """Poll every ``poll_interval`` seconds until cancelled.

        A failing poll (throttling, expired credentials, a missing permission)
        is logged and retried on the next tick rather than killing the monitor:
        the instance ids it records are diagnostics, so an outage here must
        never take a training run with it.

        :meth:`stats` is logged every ``stats_interval`` seconds (starting with
        the first poll), which is how a run's log shows the monitor is alive and
        keeping up -- per-poll detail stays at debug level. The counters are
        reset after each such line, so consecutive lines read as a rate and a
        long-past outage does not keep showing up in them.
        """
        last_stats_at = -self.stats_interval
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._stats["poll_errors"] += 1
                self._stats["last_error"] = repr(e)
                logger.warning("ec2_monitor poll failed: %r", e, exc_info=e)

            if time.monotonic() - last_stats_at >= self.stats_interval:
                last_stats_at = time.monotonic()
                logger.info("ec2_monitor stats: %s", self.stats())
                self._stats = _new_stats()

            await asyncio.sleep(self.poll_interval)

    # --- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Spawn the polling loop as a background task. Idempotent.

        Async because the task has to be created on a *running* loop, which the
        caller's ``await`` guarantees.
        """
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run_forever())

    async def stop(self) -> None:
        """Cancel the polling loop and wait for it to unwind. Idempotent."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    def stats(self) -> dict[str, Any]:
        """What happened since the last :meth:`run_forever` stats line.

        ``polls``, ``poll_errors``, ``write_errors`` and ``sessions_stamped``
        count only the current reporting interval -- :meth:`run_forever` zeroes
        them right after logging them -- while ``instances`` and
        ``sessions_tracked`` are gauges read off the latest poll.

        Sorted by key, so successive lines in a run's log line up column-wise.
        """
        return dict(sorted(dict(self._stats, sessions_tracked=len(self._stamped)).items()))

    async def __aenter__(self) -> "EC2Monitor":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()


async def start_ec2_monitor(
    capacity_provider_arn: str,
    dynamodb_table: str,
    *,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    stats_interval: float = DEFAULT_STATS_INTERVAL,
    session_prefix: str | None = None,
    region_name: str = "us-west-2",
) -> EC2Monitor:
    """Build an :class:`EC2Monitor` and start polling on the caller's loop.

    The one-liner for an async host (the trainer's entrypoint): keep the returned
    monitor referenced for as long as the run -- it owns the only reference to
    its polling task -- and optionally ``await monitor.stop()`` at the end.
    """
    monitor = EC2Monitor(
        capacity_provider_arn,
        dynamodb_table,
        session_prefix=session_prefix,
        poll_interval=poll_interval,
        stats_interval=stats_interval,
        region_name=region_name,
    )
    await monitor.start()
    return monitor
