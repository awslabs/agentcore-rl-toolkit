"""The agent server that runs inside the task container.

The ``rollout_session.wire`` calls (setup, start, status, dump) all arrive as a POST
to ``/invocations``. Setup and rollouts are long and blocking, so they run in a
thread pool and the caller polls with further invocations.
"""

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.runtime.app import RequestContextFormatter
from bedrock_agentcore.runtime.context import BedrockAgentCoreContext
from swe_agent_server.observability import (
    configure_tracing,
    set_session_id,
    stop_instrumenting_child_processes,
    traced_background_work,
)
from swe_agent_server.rollout import run_rollout, run_setup
from swe_agent_server.utils import exc_to_full_string

from agentcore_rl_toolkit.rollout_session.wire import (
    InvocationOutput,
    InvocationRequest,
    InvocationResponse,
    RolloutDumpRequest,
    RolloutSetupRequest,
    RolloutSetupResponse,
    RolloutStartRequest,
    RolloutStartResponse,
    RolloutStatusRequest,
    RolloutStatusResponse,
)

app = BedrockAgentCoreApp()
executor = ThreadPoolExecutor()
future: Future | None = None


def configure_logging() -> None:
    """Put every log line into the app's format: JSON stamped with request/session ids.

    The app's own logger and uvicorn's three loggers each have a handler already, so
    they are folded into the root handler to avoid duplicate or unstamped lines.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(RequestContextFormatter())
    logging.basicConfig(force=True, level=logging.INFO, handlers=[handler])

    app.logger.propagate = False

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True


configure_logging()
configure_tracing()
# At import, so every process a rollout spawns inherits the cleaned environment.
stop_instrumenting_child_processes()


def submit(name: str, work: Callable[[], object]) -> None:
    """Start one unit of background work, tracked in the app's async task registry.

    The protocol is sequential (setup, then one rollout), so ``future`` is a single
    slot. The registration is what ``/ping`` answers from and is released only in the
    done callback, so the session stays busy while the work runs. Wrapping happens
    here rather than on the worker so the spans land in this invocation's trace.
    """
    global future

    task_id = app.add_async_task(name)
    future = executor.submit(traced_background_work(name, work))
    future.add_done_callback(lambda finished: on_task_done(finished, task_id))


def on_task_done(finished: Future, task_id: int) -> None:
    """Release the busy status once work ends, and log a failure on its way out.

    The status call reports the same exception, but the log is the only record left if
    the caller has already given up on the session.
    """
    try:
        exception = finished.exception(timeout=0)
        if exception is not None:
            logging.error(exc_to_full_string(exception))
    finally:
        app.complete_async_task(task_id)


@app.entrypoint
def invocations(payload: dict) -> dict:
    """Serve one call of the wire protocol."""
    # Request-scoped state (a ContextVar), so copy it out here: the rollout runs on a
    # pool thread that never sees it.
    set_session_id(BedrockAgentCoreContext.get_session_id())

    request = InvocationRequest.model_validate(payload)
    response: InvocationOutput

    match request.payload:
        case RolloutSetupRequest() as setup:
            logging.info(f"Request setup: {setup.model_dump_json(indent=True)}")
            submit("rollout_setup", lambda: run_setup(setup))
            response = RolloutSetupResponse()

        case RolloutStartRequest() as start:
            logging.info(f"Request start: {start.model_dump_json(indent=True)}")
            submit("rollout_start", lambda: run_rollout(start))
            response = RolloutStartResponse()

        case RolloutStatusRequest():
            assert future is not None, "status requested before setup or start"

            # possible states: not done, done with no exception, done with exception
            done, exception_s = future.done(), None
            if done:
                exception = future.exception(timeout=0)
                if exception is not None:
                    exception_s = exc_to_full_string(exception)

            response = RolloutStatusResponse(done=done, exception=exception_s)

        case RolloutDumpRequest():
            assert future is not None, "dump requested before start"
            response = future.result(timeout=0)

    return InvocationResponse(payload=response).model_dump(mode="json")
