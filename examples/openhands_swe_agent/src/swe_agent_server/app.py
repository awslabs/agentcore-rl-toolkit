"""The agent server that runs inside the task container.

The four calls of the ``rollout_session.wire`` protocol -- setup, start, status,
dump -- all arrive as a POST of an :class:`InvocationRequest` to ``/invocations``,
which is one of the two endpoints AgentCore requires of a container.
:class:`BedrockAgentCoreApp` *is* that container contract, so this module registers
a handler for the payload and never spells out a route or a health response of its
own: it speaks the wire protocol and lets the app speak HTTP.

Setup and a rollout are long, blocking and synchronous -- ``swe_unpack.sh``, then an
agent loop that can run for many minutes -- so neither can happen inside the
invocation that asks for it; the caller reads the immediate response and then polls
with further invocations. They run in a thread pool, and the invocation returns as
soon as the work is submitted.

What keeps that safe from AgentCore's session reaper is the app's async task
registry: ``/ping`` reports ``HealthyBusy`` while any task registered with
``add_async_task`` is outstanding and ``Healthy`` once they all complete, and its
``time_of_last_update`` is when the status last *changed* rather than when the ping
arrived -- which is what lets an idle session be collected without a session in the
middle of a rollout ever being cut short. Registering the task before the work is
submitted is part of that: the caller can ping the instant this invocation returns,
and a registration made on the worker thread could land after that ping.

The registry is driven by hand rather than through ``@app.async_task`` because the
work is blocking: that decorator accepts async functions only, and the coroutine it
wraps would run on the app's single worker event loop, where a blocking rollout
would stall every other handler. For the same reason the entrypoint below is sync,
which is what makes the app run it in a thread pool and leave the main event loop
free to answer ``/ping``.
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
    """Put everything the container logs into one format: the app's JSON lines.

    The app formats its own log with :class:`RequestContextFormatter` -- one JSON
    object per line, stamped with the request and session ids, which is what a log
    query for a session keys off. That is the format worth having for every line, so
    the root handler is given the same formatter rather than a second one that would
    have to be kept in step with it, and the two sources that format themselves are
    folded in:

    - the app's own logger has a handler already and propagates on top of it, which
      with an identically formatted root handler is each line twice rather than a
      second view of it, so the propagation goes;
    - uvicorn configures three loggers of its own, with a handler each and
      propagation off, before it imports this module -- the server's startup lines
      and one access line per request, ``/ping`` included. Dropping those handlers
      hands the lines to the root one.
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
# At import, i.e. before the first shell command a rollout runs: what it removes is
# inherited by every process this one spawns.
stop_instrumenting_child_processes()


def submit(name: str, work: Callable[[], object]) -> None:
    """Start one unit of background work, tracked in the app's async task registry.

    Exactly one is ever outstanding -- the protocol is sequential (setup, then one
    rollout) -- so ``future`` is a single slot, and it is what the status and dump
    calls read. The registration is what ``/ping`` answers from, and it is released
    in the done callback rather than here, so the session stays busy for as long as
    the work actually runs.

    The work is wrapped before it is submitted, not inside the worker: that is what
    puts its spans in this invocation's trace, and it can only be done from here --
    see :func:`swe_agent_server.observability.traced_background_work`.
    """
    global future

    task_id = app.add_async_task(name)
    future = executor.submit(traced_background_work(name, work))
    future.add_done_callback(lambda finished: on_task_done(finished, task_id))


def on_task_done(finished: Future, task_id: int) -> None:
    """Release the busy status once work ends, and log a failure on its way out.

    The status call reports the same exception to the caller; this is what puts it
    in the container's log, where it is the only record if the caller has already
    given up on the session.
    """
    try:
        exception = finished.exception(timeout=0)
        if exception is not None:
            logging.error(exc_to_full_string(exception))
    finally:
        app.complete_async_task(task_id)


@app.entrypoint
def invocations(payload: dict) -> dict:
    """Serve one call of the wire protocol.

    The app hands over the request body as parsed JSON and nothing more, so the
    wire types are applied here: this function is the one place the protocol is
    turned into work, and both directions of it are validated by the models rather
    than by hand.
    """
    # The session id is request-scoped state (the app keeps it in a ContextVar) and a
    # rollout runs on a pool thread that never sees it, so it is copied out here, on
    # the request thread, for the span processor that stamps it -- see
    # swe_agent_server.observability.
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
