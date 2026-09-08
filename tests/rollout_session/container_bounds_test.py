#!/usr/bin/env python
"""Unit tests for the shared bounded container rollout.

:func:`run_rollout_with_bounds` is the one definition of how a container
rollout is sequenced -- container slot, creation-rate throttle, timed ``setup``,
rollout slot, timed ``run``, teardown -- used by the training loop
(``verl_extensions/container_agent_loop.py``), the eval harness
(``swe_agent/rollout_batch.py``) and
``tests/integration/rollout_session_integration.py``. The bounds it
applies are :class:`ContainerBounds`, whose fields are all protocol types, so the
same bundle is filled with Ray actors in the cluster and with the process-local
implementations here. These tests drive it with the local implementations and a
fake session: no Ray, no AWS, no container.

They also cover :meth:`RolloutDumpResponse.failure_reason`, the predicate both
harnesses use for their abort decision, and the boundary it draws with the bounded
run: a rollout that ran and failed inside the container comes back as a dump to be
recorded, while only failures on this side (the timeouts here, transport) raise.
"""

import asyncio
import unittest
from unittest import IsolatedAsyncioTestCase

from agentcore_rl_toolkit.aws_tools.persistent_dict import NullPersister, PersistentDict
from agentcore_rl_toolkit.concurrency.priority_assigner import LocalPriorityAssigner, PriorityAssigner
from agentcore_rl_toolkit.concurrency.priority_semaphore import LocalPrioritySemaphore, PrioritySemaphore
from agentcore_rl_toolkit.concurrency.rate_limiter import RateLimiter
from agentcore_rl_toolkit.rollout_session.lifecycle import (
    ContainerBounds,
    RolloutSession,
    run_rollout_with_bounds,
)
from agentcore_rl_toolkit.rollout_session.wire import RolloutDumpResponse

_UNSET = object()


def dump(reward=1.0, exception=None, task_output=_UNSET, **metrics) -> RolloutDumpResponse:
    """A dump response; the defaults describe a rollout that worked."""
    # ``None`` is a meaningful value here -- a rollout that produced no output --
    # so "argument omitted" needs a sentinel of its own.
    if task_output is _UNSET:
        task_output = {"patch": "diff"}
    return RolloutDumpResponse(
        task_output=task_output,
        reward=reward,
        exception=exception,
        metrics=metrics,
    )


def session_state(session_id: str = "s") -> PersistentDict:
    """A fresh state dict per rollout -- measure_span_persistent refuses to reuse one."""
    return PersistentDict(data={"session_id": session_id}, persister=NullPersister())


class FakeSession:
    """A session that does nothing but hand back the dump it was given.

    Honours the :class:`RolloutSession` contract that ``__aexit__`` tears down,
    which is what lets the bounded run own the container's lifetime without a
    ``finally`` of its own.
    """

    def __init__(
        self,
        rollout: RolloutDumpResponse | None = None,
        slow_phase: str | None = None,
        dwell: float = 0.0,
    ):
        self.rollout = rollout if rollout is not None else dump()
        self.slow_phase = slow_phase
        # Every phase yields to the event loop, as a real session's HTTP calls do.
        # Without that a rollout would run start to finish in one step and never
        # overlap another, so no concurrency ceiling could be observed at all.
        self.dwell = dwell
        self.calls: list[str] = []
        self.tasks: list[dict] = []
        self.torn_down = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        await self.shutdown()

    async def setup(self, task):
        self.calls.append("setup")
        self.tasks.append(task)
        await asyncio.sleep(10 if self.slow_phase == "setup" else self.dwell)

    async def run(self, task):
        self.calls.append("run")
        self.tasks.append(task)
        await asyncio.sleep(10 if self.slow_phase == "run" else self.dwell)
        return self.rollout

    async def shutdown(self):
        self.calls.append("shutdown")
        self.torn_down = True


class _RecordingSlot:
    """The permit-held block, with the spy's counters updated inside it.

    Hand-rolled rather than ``@asynccontextmanager`` so ``held``/``peak`` move on
    the same boundaries the permit does, which is what makes ``peak`` a truthful
    reading of how many rollouts were ever inside at once.
    """

    def __init__(self, spy: "RecordingSemaphore", priority: int) -> None:
        self._spy, self._priority = spy, priority

    async def __aenter__(self):
        await self._spy.acquire(self._priority)
        self._spy.priorities.append(self._priority)
        self._spy.held += 1
        self._spy.peak = max(self._spy.peak, self._spy.held)
        return self

    async def __aexit__(self, *exc_info):
        self._spy.held -= 1
        await self._spy.release()


class RecordingSemaphore:
    """A real :class:`LocalPrioritySemaphore` that also records how it was used.

    Delegating rather than faking means the ceiling under test is the one
    production enforces; the recording is only there to prove the bounded run
    consulted this field at all, and with which priority.
    """

    def __init__(self, value: int) -> None:
        self._sem = LocalPrioritySemaphore(value)
        self.priorities: list[int] = []
        self.held = 0
        self.peak = 0

    async def acquire(self, priority: int = 0) -> None:
        await self._sem.acquire(priority)

    async def release(self) -> None:
        await self._sem.release()

    def slot(self, priority: int = 0) -> _RecordingSlot:
        return _RecordingSlot(self, priority)


class RecordingAssigner:
    """A real :class:`LocalPriorityAssigner` that records the keys it was asked about."""

    def __init__(self) -> None:
        self._assigner = LocalPriorityAssigner()
        self.keys: list[str] = []

    async def get_priority(self, key: str) -> int:
        self.keys.append(key)
        return await self._assigner.get_priority(key)


class RecordingLimiter:
    """Counts waits instead of sleeping: the throttle's timing is not under test here."""

    def __init__(self) -> None:
        self.waits = 0

    async def wait_async(self) -> None:
        self.waits += 1


def bounds(
    container_slots: int = 1,
    rollout_slots: int = 1,
    setup_timeout: float = 30.0,
    run_timeout: float = 30.0,
    limiter: RecordingLimiter | None = None,
) -> ContainerBounds:
    return ContainerBounds(
        container_semaphore=RecordingSemaphore(container_slots),
        rollout_semaphore=RecordingSemaphore(rollout_slots),
        container_priority_assigner=RecordingAssigner(),
        rollout_priority_assigner=RecordingAssigner(),
        session_rate_limiter=limiter,
        container_setup_timeout=setup_timeout,
        agent_run_timeout=run_timeout,
    )


class SuccessPredicateTest(unittest.TestCase):
    """The rollout-success predicate that used to be inlined in each harness."""

    def test_a_working_rollout_is_successful(self):
        self.assertTrue(dump().is_successful())
        self.assertIsNone(dump().failure_reason())

    def test_the_reported_exception_is_the_reason_verbatim(self):
        # Unwrapped on purpose: the container's own traceback is the only useful
        # one to be logged.
        failed = dump(reward=None, exception="Traceback: agent crashed")
        self.assertFalse(failed.is_successful())
        self.assertEqual(failed.failure_reason(), "Traceback: agent crashed")

    def test_a_missing_reward_is_a_failure(self):
        # Nothing to train on and nothing to score, so it cannot be reported as a
        # rollout that worked -- even though the container said nothing went wrong.
        failed = dump(reward=None)
        self.assertFalse(failed.is_successful())
        self.assertIn("no reward", failed.failure_reason())

    def test_a_missing_task_output_is_a_failure(self):
        failed = dump(task_output=None, reward=0.0)
        self.assertFalse(failed.is_successful())
        self.assertIn("no task output", failed.failure_reason())

    def test_a_silent_failure_still_gets_a_reason(self):
        # The "or nowhere" case: the container failed without recording a trace.
        # The reason stands in for it so the recorded exception is never null while
        # `aborted` is true, which would make the failure uncategorisable.
        for failed in (dump(reward=None), dump(task_output=None, reward=0.0)):
            self.assertIsInstance(failed.failure_reason(), str)

    def test_zero_reward_is_a_success(self):
        # An unresolved task is a rollout that worked and scored 0, not a failure;
        # reading `reward is None` rather than falsiness is what keeps it in the
        # pass@k denominator without being counted as an abort.
        self.assertTrue(dump(reward=0.0).is_successful())


class BoundedRunTest(IsolatedAsyncioTestCase):
    async def test_returns_the_dump_on_success(self):
        session = FakeSession(dump(reward=0.5))
        state = session_state()
        got = await run_rollout_with_bounds(state, bounds(), "g", session, {"index": 1})
        self.assertIs(got, session.rollout)
        self.assertEqual(got.reward, 0.5)

    async def test_phases_run_in_order_and_the_container_is_torn_down(self):
        session = FakeSession()
        await run_rollout_with_bounds(session_state(), bounds(), "g", session, {})
        self.assertEqual(session.calls, ["setup", "run", "shutdown"])

    async def test_the_task_reaches_both_phases_unchanged(self):
        session = FakeSession()
        task = {"index": 7, "agent": "swe"}
        await run_rollout_with_bounds(session_state(), bounds(), "g", session, task)
        self.assertEqual(session.tasks, [task, task])

    # -- a container-side failure is data, not an exception -----------------------

    async def test_a_failed_dump_is_returned_rather_than_raised(self):
        # Raising here would replace the container's own account of what went wrong
        # with a traceback of these lines. The dump comes back intact and the caller
        # asks it; only failures on this side raise.
        session = FakeSession(dump(reward=None, exception="agent crashed"))
        got = await run_rollout_with_bounds(session_state(), bounds(), "g", session, {})
        self.assertIs(got, session.rollout)
        self.assertFalse(got.is_successful())

    async def test_a_failed_dump_is_recorded_as_run_not_timed_out(self):
        # A rollout that crashed must stay distinguishable in the session store from
        # one that was killed by the deadline.
        session = FakeSession(dump(reward=None, exception="agent crashed"))
        state = session_state()
        await run_rollout_with_bounds(state, bounds(), "g", session, {})
        self.assertEqual(state["agent_run_timeout_exceeded"], 0.0)
        self.assertIn("agent_run_start_at", state)
        self.assertIn("agent_run_end_at", state)

    async def test_a_failed_dump_still_tears_the_container_down(self):
        session = FakeSession(dump(reward=None, exception="agent crashed"))
        await run_rollout_with_bounds(session_state(), bounds(), "g", session, {})
        self.assertTrue(session.torn_down)

    async def test_a_failed_dump_frees_its_slots(self):
        b = bounds(container_slots=1, rollout_slots=1)
        for i in range(3):
            await run_rollout_with_bounds(
                session_state(f"s{i}"),
                b,
                str(i),
                FakeSession(dump(reward=None, exception="agent crashed")),
                {},
            )
        self.assertEqual(b.container_semaphore.held, 0)
        self.assertEqual(b.rollout_semaphore.held, 0)

    # -- every ContainerBounds field is actually consulted -----------------------

    async def test_both_semaphores_cap_concurrency(self):
        # Six rollouts that all dwell in every phase, so they would overlap freely
        # if nothing capped them; each semaphore's peak must be its own ceiling.
        b = bounds(container_slots=2, rollout_slots=1)
        await asyncio.gather(
            *(
                run_rollout_with_bounds(
                    session_state(f"s{i}"),
                    b,
                    str(i),
                    FakeSession(dwell=0.01),
                    {"index": i},
                )
                for i in range(6)
            )
        )
        self.assertEqual(b.container_semaphore.peak, 2)
        self.assertEqual(b.rollout_semaphore.peak, 1)

    async def test_each_assigner_is_asked_for_the_priority_key(self):
        b = bounds()
        await run_rollout_with_bounds(session_state(), b, "step:task", FakeSession(), {})
        self.assertEqual(b.container_priority_assigner.keys, ["step:task"])
        self.assertEqual(b.rollout_priority_assigner.keys, ["step:task"])

    async def test_the_assigned_priority_is_what_the_semaphore_receives(self):
        # The assigners are separate so the two dimensions are numbered
        # independently; each semaphore must see its own assigner's number.
        b = bounds(container_slots=2, rollout_slots=2)
        for i in range(3):
            await run_rollout_with_bounds(
                session_state(f"s{i}"),
                b,
                f"g{i}",
                FakeSession(),
                {},
            )
        self.assertEqual(b.container_semaphore.priorities, [0, 1, 2])
        self.assertEqual(b.rollout_semaphore.priorities, [0, 1, 2])

    async def test_one_priority_for_every_member_of_a_group(self):
        # A group is one prompt's samples: they queue together, at one priority.
        b = bounds(container_slots=4, rollout_slots=4)
        await asyncio.gather(
            *(run_rollout_with_bounds(session_state(f"s{i}"), b, "same-group", FakeSession(), {}) for i in range(4))
        )
        self.assertEqual(b.container_semaphore.priorities, [0, 0, 0, 0])

    async def test_the_rate_limiter_is_consulted_once_per_rollout(self):
        limiter = RecordingLimiter()
        b = bounds(limiter=limiter)
        for i in range(3):
            await run_rollout_with_bounds(session_state(f"s{i}"), b, str(i), FakeSession(), {})
        self.assertEqual(limiter.waits, 3)

    async def test_the_rate_limit_wait_is_timed_and_sits_before_setup(self):
        # Held inside the container slot and ahead of provisioning: that ordering is
        # what makes the throttle bound container *creation* rather than dispatch.
        class OrderedSession(FakeSession):
            async def setup(self, task):
                order.append("setup")
                await super().setup(task)

        class OrderedLimiter(RecordingLimiter):
            async def wait_async(self):
                order.append("rate_limit")
                await super().wait_async()

        order: list[str] = []
        state = session_state()
        b = bounds(limiter=OrderedLimiter())
        await run_rollout_with_bounds(state, b, "g", OrderedSession(), {})
        self.assertEqual(order, ["rate_limit", "setup"])
        # the container slot was taken before the wait
        self.assertIn("container_slot_start_at", state)
        self.assertIn("rate_limit_wait", state)

    async def test_no_rate_limiter_is_tolerated(self):
        # None means the semaphores are the only brake, which is the default.
        b = bounds(limiter=None)
        self.assertIsNone(b.session_rate_limiter)
        state = session_state()
        await run_rollout_with_bounds(state, b, "g", FakeSession(), {})
        self.assertNotIn("rate_limit_wait", state)

    async def test_the_rollout_slot_is_taken_only_after_setup(self):
        # Containers may sit warm while fewer of them talk to inference, so the
        # rollout permit must not be held across provisioning.
        order: list[str] = []

        class OrderedSession(FakeSession):
            async def setup(self, task):
                order.append("setup")
                await super().setup(task)

        class WatchingSemaphore(RecordingSemaphore):
            def slot(self, priority=0):
                order.append("rollout_slot")
                return _RecordingSlot(self, priority)

        b = bounds()
        b.rollout_semaphore = WatchingSemaphore(1)
        await run_rollout_with_bounds(session_state(), b, "g", OrderedSession(), {})
        self.assertEqual(order, ["setup", "rollout_slot"])

    # -- the two timeouts --------------------------------------------------------

    async def test_setup_timeout_fires_and_is_recorded(self):
        state = session_state()
        with self.assertRaises(TimeoutError):
            await run_rollout_with_bounds(
                state,
                bounds(setup_timeout=0.05, run_timeout=30.0),
                "g",
                FakeSession(slow_phase="setup"),
                {},
            )
        self.assertEqual(state["container_setup_timeout_exceeded"], 1.0)
        # the run never started, so its flag must be absent rather than 0.0
        self.assertNotIn("agent_run_timeout_exceeded", state)
        # the span still closed: measure_span_persistent writes in a finally
        self.assertIn("container_setup_end_at", state)

    async def test_run_timeout_fires_independently_and_is_recorded(self):
        state = session_state()
        with self.assertRaises(TimeoutError):
            await run_rollout_with_bounds(
                state,
                bounds(setup_timeout=30.0, run_timeout=0.05),
                "g",
                FakeSession(slow_phase="run"),
                {},
            )
        self.assertEqual(state["container_setup_timeout_exceeded"], 0.0)
        self.assertEqual(state["agent_run_timeout_exceeded"], 1.0)
        self.assertIn("agent_run_end_at", state)

    async def test_a_timed_out_rollout_still_tears_the_container_down(self):
        session = FakeSession(slow_phase="run")
        with self.assertRaises(TimeoutError):
            await run_rollout_with_bounds(
                session_state(),
                bounds(run_timeout=0.05),
                "g",
                session,
                {},
            )
        self.assertTrue(session.torn_down)

    async def test_a_timed_out_rollout_frees_its_slots(self):
        # Nothing is released by hand, so a stuck rollout must not strand a permit.
        b = bounds(container_slots=1, rollout_slots=1, run_timeout=0.05)
        for i in range(3):
            with self.assertRaises(TimeoutError):
                await run_rollout_with_bounds(
                    session_state(f"s{i}"),
                    b,
                    str(i),
                    FakeSession(slow_phase="run"),
                    {},
                )
        self.assertEqual(b.container_semaphore.held, 0)
        self.assertEqual(b.rollout_semaphore.held, 0)

    # -- the timing spans the session store is read for --------------------------

    async def test_the_lifecycle_is_recorded_on_the_session_state(self):
        state = session_state()
        await run_rollout_with_bounds(state, bounds(limiter=RecordingLimiter()), "g", FakeSession(), {})
        for key in (
            "container_priority",
            "container_slot_start_at",
            "rate_limit_wait",
            "container_setup",
            "container_created_at",
            "rollout_priority",
            "rollout_slot_start_at",
            "agent_run",
        ):
            self.assertIn(key, state, f"{key} not recorded")


class ProtocolConformanceTest(unittest.TestCase):
    """The bounds hold interfaces, which is what lets one bundle describe both
    deployments -- Ray actors in the cluster, these locals in a single process."""

    def test_the_local_implementations_satisfy_the_bounds_interfaces(self):
        self.assertIsInstance(LocalPrioritySemaphore(1), PrioritySemaphore)
        self.assertIsInstance(LocalPriorityAssigner(), PriorityAssigner)
        self.assertIsInstance(RecordingLimiter(), RateLimiter)

    def test_the_ray_wrappers_satisfy_them_too_without_a_cluster(self):
        # Structural check only: instantiating the wrapper around a placeholder
        # handle needs no Ray runtime, which is the point of the wrappers.
        from agentcore_rl_toolkit.concurrency.ray_adapters import (
            RayPriorityAssigner,
            RayPrioritySemaphore,
            RayRateLimiter,
        )

        self.assertIsInstance(RayPrioritySemaphore(None), PrioritySemaphore)  # type: ignore[arg-type]
        self.assertIsInstance(RayPriorityAssigner(None), PriorityAssigner)  # type: ignore[arg-type]
        self.assertIsInstance(RayRateLimiter(None), RateLimiter)  # type: ignore[arg-type]

    def test_the_fake_session_satisfies_the_session_interface(self):
        # Keeps these tests honest: the double stands in for a real session only
        # as long as it still matches what the bounded run is typed against.
        self.assertIsInstance(FakeSession(), RolloutSession)


if __name__ == "__main__":
    unittest.main()
