"""A long-lived poller that stamps ``ec2_instance_id`` onto agent sessions.

One ``DescribeInstances`` poll per interval covers a whole capacity provider -- a
constant API cost -- and upserts each running instance's id onto its session's
dynamodb item, instead of one lookup per session at rollout setup.
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

    :meth:`start`/:meth:`stop` are idempotent. The monitor holds the only reference
    to its polling task, so the caller must keep the monitor itself alive.
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
        starts with it; ``None`` considers all of them. ``region_name`` is the
        dynamodb region -- the EC2 region comes from ``capacity_provider_arn``.
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
        """Run one EC2 lookup; return the ``session_id -> instance_id`` pairs written."""
        instances = await find_running_instances(self.capacity_provider_arn, self.session_prefix)

        live: dict[str, str] = {}
        for instance in instances:
            session_id = instance_to_session(instance)
            if session_id:
                live[session_id] = instance["InstanceId"]

        pending = {
            session_id: instance_id
            for session_id, instance_id in live.items()
            if self._stamped.get(session_id) != instance_id
        }
        written = await self._write(pending)

        # Prune to live instances, so the map stays bounded by capacity rather than
        # growing with the run's total session count.
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

        Returns only the pairs written; a failed write leaves the session unstamped
        so the next poll retries it.
        """
        if not updates:
            return {}

        semaphore = asyncio.Semaphore(self.write_concurrency)
        session = await get_aioboto3_session()
        async with session.resource("dynamodb", region_name=self.region_name) as dynamodb:  # type: ignore
            table = await dynamodb.Table(self.dynamodb_table)

            async def write_one(session_id: str, instance_id: str):
                async with semaphore:
                    # An upsert: a session with no item yet is created here and
                    # filled in by its agent loop later.
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

        A failing poll is logged and retried on the next tick, never fatal: the ids
        recorded are diagnostics and must not take a training run down. :meth:`stats`
        is logged every ``stats_interval`` seconds and its counters then zeroed.
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

        Async because the task has to be created on a *running* loop.
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

        Counters cover only the current reporting interval; ``instances`` and
        ``sessions_tracked`` are gauges off the latest poll. Sorted by key so
        successive log lines line up column-wise.
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

    Keep the returned monitor referenced for the life of the run -- it owns the only
    reference to its polling task.
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
