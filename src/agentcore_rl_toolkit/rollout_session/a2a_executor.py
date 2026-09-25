"""Server-side A2A executor that runs one rollout as one A2A ``Task``.

First ``message/send`` -> :meth:`run_setup`, then park at ``input-required``; the resuming
message -> :meth:`run_rollout`, then ``completed`` with the dump as an artifact (or
``failed`` if the rollout couldn't be attempted). The blocking hooks run in a worker thread
because ``execute`` is a coroutine on the server loop; blocking there would stall ``/ping``.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, Task, TaskState, TaskStatus
from bedrock_agentcore.runtime.models import PingStatus

from agentcore_rl_toolkit.rollout_session.a2a_protocol import (
    dump_to_parts,
    extract_task_input,
)
from agentcore_rl_toolkit.rollout_session.exception_utils import exception_to_string
from agentcore_rl_toolkit.rollout_session.wire import RolloutDumpResponse

logger = logging.getLogger(__name__)


class RolloutAgentExecutor(AgentExecutor, ABC):
    """Runs setup then a rollout for one A2A task, tracking whether any task is working.

    Setup vs rollout is distinguished by whether the task already exists
    (``context.current_task``). ``_working`` holds ids of tasks mid setup/rollout; a ping
    handler reads it to report ``HealthyBusy``.
    """

    def __init__(self) -> None:
        self._working: set[str] = set()

    @abstractmethod
    def run_setup(self, task_input: dict) -> None:
        """Prepare the environment for ``task_input``. Runs in a worker thread."""

    @abstractmethod
    def run_rollout(self, task_input: dict) -> RolloutDumpResponse:
        """Run one rollout and return its dump. Runs in a worker thread.

        Failures *inside* the container go in the dump's ``exception`` field (data the
        trainer keeps); raise only when the rollout couldn't be attempted at all.
        """

    def cleanup(self) -> None:
        """Best-effort teardown on cancellation; a no-op unless an agent overrides it."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if context.current_task is None:
            await self._do_setup(context, updater)
        else:
            await self._do_rollout(context, updater)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        try:
            self.cleanup()
        except Exception:
            logger.exception("cleanup during cancel failed for task %s", context.task_id)
        finally:
            self._working.discard(context.task_id)
            await updater.cancel()

    async def _do_setup(self, context: RequestContext, updater: TaskUpdater) -> None:
        task_id = context.task_id
        self._working.add(task_id)
        try:
            # v1 requires the initial Task be enqueued before any status update (no TaskUpdater.submit).
            await updater.event_queue.enqueue_event(
                Task(id=task_id, context_id=context.context_id, status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED))
            )
            await updater.start_work()
            task_input = extract_task_input(context.message)
            await asyncio.to_thread(self.run_setup, task_input)
        except Exception as e:
            await updater.failed(message=self._text(updater, exception_to_string(e)))
            logger.error("setup failed for task %s", task_id, exc_info=e)
            return
        finally:
            self._working.discard(task_id)
        # park until the rollout message arrives on the same task id
        await updater.requires_input(message=self._text(updater, "ready"))

    async def _do_rollout(self, context: RequestContext, updater: TaskUpdater) -> None:
        task_id = context.task_id
        self._working.add(task_id)
        try:
            await updater.start_work()
            task_input = extract_task_input(context.message)
            dump = await asyncio.to_thread(self.run_rollout, task_input)
        except Exception as e:
            await updater.failed(message=self._text(updater, exception_to_string(e)))
            logger.error("rollout failed for task %s", task_id, exc_info=e)
            return
        finally:
            self._working.discard(task_id)
        await updater.add_artifact(dump_to_parts(dump), name="rollout")
        await updater.complete()

    def _text(self, updater: TaskUpdater, text: str):
        return updater.new_agent_message([Part(text=text)])

    def ping_status(self) -> PingStatus:
        """``HealthyBusy`` while any task is mid setup/rollout, else ``Healthy``."""
        return PingStatus.HEALTHY_BUSY if self._working else PingStatus.HEALTHY


def make_ping_handler(executor: RolloutAgentExecutor):
    """A ``ping_handler`` for ``build_a2a_app`` reporting ``executor``'s busy state."""

    def handler() -> PingStatus:
        return executor.ping_status()

    return handler
