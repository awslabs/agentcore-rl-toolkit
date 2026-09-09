"""Unit tests for :class:`LocalPrioritySemaphore`: priority ordering, FIFO among equal
priorities, permit accounting under cancellation, and the ``slot`` context manager.
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

    # Neither priority order nor reverse-FIFO, so passing cannot be an accident of
    # insertion order.
    tasks = [
        asyncio.create_task(worker(5, "low-a")),
        asyncio.create_task(worker(1, "high-a")),
        asyncio.create_task(worker(5, "low-b")),
        asyncio.create_task(worker(1, "high-b")),
    ]
    await asyncio.sleep(0.05)  # let every worker enqueue before any permit is handed out

    for _ in tasks:
        await sem.release()
        await asyncio.sleep(0.01)

    await asyncio.gather(*tasks)

    # Lower int == higher priority.
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

    # The release must skip the dead future rather than be swallowed by it.
    await sem.release()
    assert sem.value == 1


@pytest.mark.asyncio
async def test_slot_context_manager_releases_on_exit():
    sem = LocalPrioritySemaphore(value=1)
    async with sem.slot():
        assert sem.locked()
    assert not sem.locked()

    with pytest.raises(RuntimeError):
        async with sem.slot():
            raise RuntimeError("boom")
    assert not sem.locked()
