"""Agent server running inside the task container.

One rollout is one A2A task: the first message runs setup, the second runs the rollout.
Both are blocking, so they run in worker threads while the caller polls between calls.
"""

import logging

from bedrock_agentcore.runtime.a2a import build_a2a_app
from bedrock_agentcore.runtime.app import RequestContextFormatter
from swe_agent_server.observability import (
    configure_tracing,
    set_session_id,
    stop_instrumenting_child_processes,
    traced_background_work,
)
from swe_agent_server.rollout import run_rollout, run_setup

from agentcore_rl_toolkit.rollout_session.a2a_executor import RolloutAgentExecutor, make_ping_handler
from agentcore_rl_toolkit.rollout_session.wire import RolloutDumpResponse


def configure_logging() -> None:
    """Put every log line into the app's format: JSON stamped with request/session ids.

    uvicorn's loggers are folded into the root handler to avoid duplicate lines.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(RequestContextFormatter())
    logging.basicConfig(force=True, level=logging.INFO, handlers=[handler])

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True


configure_logging()
configure_tracing()
# At import, so every process a rollout spawns inherits the cleaned environment.
stop_instrumenting_child_processes()


class SweAgentExecutor(RolloutAgentExecutor):
    """The SWE agent's setup/rollout hooks, run as one A2A task per rollout."""

    async def execute(self, context, event_queue) -> None:
        # contextId is the ACR session id; set here so it propagates into worker threads via contextvars.
        set_session_id(context.context_id)
        await super().execute(context, event_queue)

    def run_setup(self, task_input: dict) -> None:
        work = traced_background_work("rollout_setup", lambda: run_setup(task_input))
        work()

    def run_rollout(self, task_input: dict) -> RolloutDumpResponse:
        work = traced_background_work("rollout_start", lambda: run_rollout(task_input))
        return work()


_executor = SweAgentExecutor()
app = build_a2a_app(_executor, ping_handler=make_ping_handler(_executor))
