"""Unit tests for :class:`LocalPrioritySemaphore`: the in-process semantics.

Priority ordering, FIFO among equal priorities, permit accounting under
cancellation, and the ``slot`` context manager -- the behaviour every caller
gets, whether it holds the object directly (``swe_agent/rollout_batch.py``, the
integration test) or reaches a shared one through
:class:`~...concurrency.ray_adapters.RayPrioritySemaphore`.

What this file used to also do was host the same class as a Ray actor and check
which of those semantics survive the actor RPC boundary. Those tests are gone:
booting a local Ray cluster cost ~10s of a ~19s suite, and their real subject was
never this class but the wiring in ``verl_extensions/container_resources.py``
(named ``get_if_exists`` actors, the adapters over the handles). That belongs in a
Ray + ``verl_extensions`` test, still to be written. Two findings from them worth
not rediscovering, both already recorded in ``ray_adapters.py``: an actor handle
exposes methods but not ``@property`` accessors (so ``locked()``, never
``value``), and ``slot`` comes back as an ``ObjectRefGenerator`` that is not an
async context manager, which is why the adapter rebuilds it from
``acquire``/``release``.

No Ray, no AWS.
"""

import asyncio

import pytest

from agentcore_rl_toolkit.concurrency.priority_semaphore import LocalPrioritySemaphore


def test_rejects_negative_initial_value():
    with pytest.raises(ValueError):
        LocalPrioritySemaphore(value=-1)


@pytest.mark.asyncio
async def test_uncontended_acquire_decrements_value():
    sem = LocalPrioritySemaphore(value=2)
    assert sem.value == 2
    assert not sem.locked()

    await sem.acquire()
    assert sem.value == 1

    await sem.acquire()
    assert sem.value == 0
    assert sem.locked()


@pytest.mark.asyncio
async def test_release_restores_value_when_uncontended():
    sem = LocalPrioritySemaphore(value=1)
    await sem.acquire()
    assert sem.locked()
    await sem.release()
    assert sem.value == 1
    assert not sem.locked()


@pytest.mark.asyncio
async def test_waiters_ordered_by_priority_then_fifo():
    sem = LocalPrioritySemaphore(value=1)
    await sem.acquire()  # exhaust the single permit

    order: list[str] = []

    async def worker(priority: int, tag: str):
        await sem.acquire(priority)
        order.append(tag)

    # Queue in an order that is neither priority order nor reverse-FIFO, so a
    # passing result cannot be an accident of insertion order.
    tasks = [
        asyncio.create_task(worker(5, "low-a")),
        asyncio.create_task(worker(1, "high-a")),
        asyncio.create_task(worker(5, "low-b")),
        asyncio.create_task(worker(1, "high-b")),
    ]
    # Let every worker enqueue itself before any permit is handed out.
    await asyncio.sleep(0.05)

    # Hand out one permit per release; each frees the next-best waiter.
    for _ in tasks:
        await sem.release()
        await asyncio.sleep(0.01)

    await asyncio.gather(*tasks)

    # Priority first (lower int == higher priority), FIFO within a priority.
    assert order == ["high-a", "high-b", "low-a", "low-b"]


@pytest.mark.asyncio
async def test_new_caller_does_not_bypass_existing_waiter():
    sem = LocalPrioritySemaphore(value=1)
    await sem.acquire()  # no permits left

    acquired: list[str] = []

    async def waiter():
        await sem.acquire(priority=0)
        acquired.append("waiter")

    w = asyncio.create_task(waiter())
    await asyncio.sleep(0.05)  # ensure `waiter` is queued

    # A brand-new caller must not jump ahead of the already-queued waiter even
    # though it arrives while a permit is (about to be) available.
    async def latecomer():
        await sem.acquire(priority=0)
        acquired.append("latecomer")

    lc = asyncio.create_task(latecomer())
    await asyncio.sleep(0.05)

    await sem.release()  # only enough for one of them
    await asyncio.sleep(0.05)

    assert acquired == ["waiter"]

    await sem.release()
    await asyncio.gather(w, lc)
    assert acquired == ["waiter", "latecomer"]


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_consume_permit():
    sem = LocalPrioritySemaphore(value=1)
    await sem.acquire()

    async def waiter():
        await sem.acquire()

    w = asyncio.create_task(waiter())
    await asyncio.sleep(0.05)
    w.cancel()
    with pytest.raises(asyncio.CancelledError):
        await w

    # Releasing should skip the cancelled waiter and restore the permit rather
    # than being silently swallowed by the dead future.
    await sem.release()
    assert sem.value == 1


@pytest.mark.asyncio
async def test_slot_context_manager_releases_on_exit():
    sem = LocalPrioritySemaphore(value=1)
    async with sem.slot():
        assert sem.locked()
    assert not sem.locked()

    # Even if the body raises, the permit must come back.
    with pytest.raises(RuntimeError):
        async with sem.slot():
            raise RuntimeError("boom")
    assert not sem.locked()
