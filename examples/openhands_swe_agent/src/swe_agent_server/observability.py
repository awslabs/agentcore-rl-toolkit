"""Make the agent loop's spans reach CloudWatch, and be findable by session.

The container runs under ``opentelemetry-instrument`` (see ``image/entrypoint.sh``),
so AWS's distro owns the global tracer provider and exports through
``OTLPAwsSpanExporter``, which SigV4-signs its POST to the X-Ray endpoint. Two things
are wrong with what comes out of that, and this module is the fix for both (measured in
``docs/otel_swe_agent_missing.md``):

- OpenHands traces through Laminar, which under ADOT keeps a provider of its own and
  exports it unsigned to ADOT's endpoint -- 403, so the agent loop's spans are dropped
  and ADOT's httpx leaves are left parented to spans nothing ever exported;
- a span started off the request thread carries no ``session.id``, and that attribute is
  what CloudWatch's session view and the AgentCore span collector filter on -- so the
  whole rollout is unfindable by the only query that matters.

Five entry points, in the order the container reaches them:

- :func:`configure_tracing` -- called once at app import. Stamps ``session.id`` on
  every span the process starts, wherever it starts it.
- :func:`stop_instrumenting_child_processes` -- called once at app import too. Keeps
  the distro's auto-instrumentation out of the shells the agent runs, which it is
  otherwise pulled into by an inherited ``PYTHONPATH``.
- :func:`set_session_id` -- called on the request thread, which is the only place the
  session id can be read; one ACR session is one container, so a process-wide value
  is exactly the right scope.
- :func:`traced_background_work` -- wraps setup and a rollout, the two things this
  server does off the request thread, so their spans land in the trace of the
  invocation that asked for them instead of in a rootless trace of their own.
- :func:`configure_openhands_tracing` -- called before the OpenHands backend is
  imported, i.e. before its import-time ``maybe_init_laminar()``. Routes Laminar's
  provider through ADOT's signing pipeline, or (``SWE_AGENT_OPENHANDS_TRACING=off``)
  leaves the layer dormant.

Nothing here may break a rollout: an observability failure is logged and swallowed.
The parts that reach into ADOT's and Laminar's private attributes are the reason --
both are pinned dependencies, and an upgrade that moves them should cost a line in
the log, not every rollout in the run.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable, Iterator, Sequence

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Span, SpanProcessor, TracerProvider
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)

# Resolved lazily by the SDK on each use, so importing this module before
# ``opentelemetry-instrument`` has installed its provider is harmless.
_tracer = trace.get_tracer(__name__)

# The attribute CloudWatch's session view and bedrock_agentcore's
# CloudWatchAgentSpanCollector filter on -- not to be confused with Laminar's
# ``session_id`` association property, which OpenHands fills with the *conversation*
# id and which no AWS-side query reads.
SESSION_ID_ATTRIBUTE = "session.id"

# The tail of the directory ``opentelemetry-instrument`` puts on PYTHONPATH to
# instrument its child. Matched as a path rather than imported: importing the package
# to ask it for its own location would be a heavier thing than reading a string.
_AUTO_INSTRUMENTATION_DIR = os.path.join("opentelemetry", "instrumentation", "auto_instrumentation")

MODE_ENV = "SWE_AGENT_OPENHANDS_TRACING"
MODE_ADOT = "adot"
MODE_OFF = "off"

# Route OpenHands' spans into ADOT's exporter by default: they are the agent loop, and
# an evaluator that reads the session's trace can only see what is exported. The image
# is deployed with no environment of its own, so this default is what runs; set
# MODE_ENV to "off" on the runtime to leave the Laminar layer dormant instead.
DEFAULT_MODE = MODE_ADOT

# What ``openhands.sdk.observability.laminar.should_enable_observability()`` gates on:
# any one of these being set arms the whole layer.
_OBSERVABILITY_ENV_KEYS = (
    "LMNR_PROJECT_API_KEY",
    "OTEL_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
)

# Set once on the request thread and read from every thread the process traces on.
_session_id: str | None = None

_tracing_configured = False
_openhands_tracing_configured = False


def set_session_id(session_id: str | None) -> None:
    """Record the ACR session id this container is serving.

    Called from the invocation handler, which runs on the request thread -- the only
    context where the id is available (the app keeps it in a ``ContextVar``, and the
    rollout runs on a pool thread that never sees it). One session id routes to one
    container for the life of the session, so the value is a property of the process
    rather than of the request, and a second, different id means an assumption this
    recipe is built on no longer holds: worth a warning, but the newer id is still
    the one to stamp.
    """
    global _session_id

    if session_id is None or session_id == _session_id:
        return
    if _session_id is not None:
        logger.warning("session id changed within one container: %s -> %s", _session_id, session_id)
    _session_id = session_id


class SessionIdSpanProcessor(SpanProcessor):
    """Stamps :data:`SESSION_ID_ATTRIBUTE` on every span, on any thread.

    The distro's own ``BaggageSpanProcessor`` does this from the request's baggage,
    which only reaches spans started on the request thread. This one reads the
    process-wide id instead, so the agent loop's spans -- started on the rollout
    worker, in threads OpenHands spawns under it, and in Laminar's own provider --
    carry the attribute too. Setting it a second time on a span that already has it
    from baggage is a write of the same value.
    """

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        if _session_id is not None:
            span.set_attribute(SESSION_ID_ATTRIBUTE, _session_id)

    def on_end(self, span: Span) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def configure_tracing() -> None:
    """Stamp ``session.id`` on everything this process exports. Idempotent."""
    global _tracing_configured

    if _tracing_configured:
        return
    try:
        provider = _sdk_tracer_provider()
        if provider is None:
            logger.info("no SDK tracer provider (not running under opentelemetry-instrument); tracing untouched")
            return
        provider.add_span_processor(SessionIdSpanProcessor())
        _tracing_configured = True
    except Exception:
        logger.error("failed to configure tracing", exc_info=True)


def stop_instrumenting_child_processes() -> None:
    """Take the auto-instrumentation off ``PYTHONPATH``, for everything spawned below.

    ``opentelemetry-instrument`` instruments its child by prepending the distro's
    ``auto_instrumentation`` directory to ``PYTHONPATH``: that directory holds a
    ``sitecustomize.py``, which every interpreter started with that path imports, and
    which imports ``opentelemetry``. The variable is inherited, so the agent's own shell
    commands get it too -- and the interpreter they mean is a task one
    (``/opt/miniconda3/envs/testbed/bin/python`` in the SWE-bench images), which has no
    ``opentelemetry``. So every ``python`` the agent runs prints

        Error in sitecustomize; set PYTHONVERBOSE for traceback:
        ModuleNotFoundError: No module named 'opentelemetry'

    into the observation the model is then trained on. Harmless to the command's exit
    code, but it is noise in a trajectory, it is two lines of tokens per tool call, and a
    test that compares stderr exactly would fail on it -- which would be a wrong reward,
    not just noise. In a task environment that *does* have opentelemetry importable it is
    worse than noise: the child would be fully auto-instrumented, and the ``OTEL_*``
    variables are inherited as well, so a ``pytest`` run would try to export spans of its
    own to the session's endpoint.

    Only that one entry goes; the rest of ``PYTHONPATH`` is the image's own (``/agent``)
    and children still need it. Removing it here is safe because this process was
    instrumented at interpreter startup, long before this module is imported -- the
    variable has no further use for us, only for children we do not want instrumented.
    """
    path = os.environ.get("PYTHONPATH")
    if not path:
        return

    entries = path.split(os.pathsep)
    kept = [entry for entry in entries if not os.path.normpath(entry).endswith(_AUTO_INSTRUMENTATION_DIR)]
    if kept == entries:
        return

    if kept:
        os.environ["PYTHONPATH"] = os.pathsep.join(kept)
    else:
        # An empty PYTHONPATH is not the same as an unset one: it puts the working
        # directory on sys.path for every command the agent runs.
        del os.environ["PYTHONPATH"]
    logger.info("removed the auto-instrumentation from PYTHONPATH for child processes: %r -> %r", path, kept)


def traced_background_work(name: str, work: Callable[[], object]) -> Callable[[], object]:
    """Wrap ``work`` so what it traces belongs to the invocation that submitted it.

    Called on the request thread and run on a pool thread, which is the whole point:
    the span is *started here*, where the invocation's own server span is current (the
    app copies the request context and runs this sync handler through ``ctx.run`` in a
    thread pool, so the ASGI instrumentation's ``POST /invocations`` span is the parent
    OTel offers), and the context carrying it is attached over there. So the setup or
    rollout span is a child of the invocation that asked for it, and what the work goes
    on to trace descends from it in that same trace instead of starting a fresh rootless
    one as it does today. OpenHands' conversation root is the one exception, and it
    takes :func:`_laminar_roots_under_current_span` to bring in.

    Two consequences worth naming. The attached context carries the request's baggage
    as well as its span, so the distro's ``BaggageSpanProcessor`` stamps ``session.id``
    wherever the context survives -- :class:`SessionIdSpanProcessor` remains the
    backstop for the threads that lose it, which is most of the agent loop. And the
    span outlives the invocation that opened it by minutes: a parent that ends before
    its children is well-formed, and the invocation span is exported immediately, so
    the trace has a root from the start even if a container killed mid-rollout never
    exports this span at all.
    """
    span = _tracer.start_span(name)
    context = trace.set_span_in_context(span)

    def run() -> object:
        token = otel_context.attach(context)
        try:
            return work()
        except Exception as error:
            # The status and the exception event are what make a failed rollout
            # visible in the trace; app.on_task_done separately puts it in the log.
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            raise
        finally:
            otel_context.detach(token)
            span.end()

    return run


def configure_openhands_tracing(mode: str | None = None) -> None:
    """Settle what happens to OpenHands' Laminar layer, before it initialises.

    Must be called before anything imports ``openhands.sdk``, because that import is
    the initialisation (its package ``__init__`` reaches the module-level
    ``maybe_init_laminar()`` calls): the ``off`` mode has to disarm the gate first, and
    the routed mode has to fix the resource before Laminar builds its provider (see
    :func:`_laminar_resource_pinned_to`). Idempotent, and it performs the import itself
    in the routed mode so the wiring is done in one place rather than left to whichever
    import happens first.
    """
    global _openhands_tracing_configured

    mode = mode or os.environ.get(MODE_ENV, DEFAULT_MODE)
    try:
        if not _openhands_tracing_configured:
            # The routed mode copies the global provider's processors, so the session-id
            # stamper has to be among them by now. It is, when the app is serving; this
            # makes it so for any other caller too.
            configure_tracing()

            if mode == MODE_OFF:
                _disarm_observability_gate()
            elif mode == MODE_ADOT:
                _route_laminar_through_adot()
            else:
                raise ValueError(f"{MODE_ENV} must be {MODE_ADOT!r} or {MODE_OFF!r}, got {mode!r}")
            _openhands_tracing_configured = True

        if mode == MODE_ADOT:
            # Per call rather than once, because this one is thread-local and belongs
            # to the rollout being started, not to the process.
            _laminar_roots_under_current_span()
    except Exception:
        logger.error("failed to configure OpenHands tracing (mode=%s)", mode, exc_info=True)


def _disarm_observability_gate() -> None:
    """Leave OpenHands' observability layer dormant, and say so in the log.

    The gate cannot be left unarmed by configuring less: the variable that arms it is
    one the AWS distro sets in this process at interpreter startup. So it is unset
    here instead, before the import that reads it -- and left unset, rather than
    restored once ``maybe_init_laminar()`` has passed on it, because
    ``should_enable_observability()`` is re-read by every ``@observe`` call and a
    restore would arm the layer again mid-rollout. ADOT's own exporter is unaffected:
    it captured its endpoint when it was built, long before this module is imported.
    """
    dropped = [key for key in _OBSERVABILITY_ENV_KEYS if os.environ.pop(key, None) is not None]
    logger.info("OpenHands tracing off: unset %s", ", ".join(dropped) or "nothing (gate was already unarmed)")


def _route_laminar_through_adot() -> None:
    """Give Laminar's spans ADOT's resource and ADOT's signing exporter.

    Three edits, all to the provider Laminar keeps for itself:

    - it is built with ADOT's resource, so its spans are attributed to the same
      service as everything else the container exports (Laminar names the service
      after ``sys.argv[0]``, which here is uvicorn);
    - ADOT's span processors are attached to it. They are the SigV4-signing exporter
      plus the distro's own enrichment, so a Laminar span is now exported exactly as
      an ADOT one is -- and carries ``scope.name``, the conjunct
      ``CloudWatchAgentSpanCollector``'s query fails on for AWS's scope-less
      front-door envelopes;
    - Laminar's own span processor is detached, which is what stops the 403 loop.
      Only the export path goes: the processor object stays alive for the
      instrumentations that hold it directly, and what is lost with it is Laminar's
      ``lmnr.span.path`` bookkeeping, which nothing on the AWS side reads.
    """
    adot = _sdk_tracer_provider()
    if adot is None:
        # Then Laminar's own provider becomes the global one and exports through
        # whatever it was configured with -- no ADOT pipeline to route into.
        logger.info("no SDK tracer provider; leaving OpenHands' tracing as configured")
        return

    with _laminar_resource_pinned_to(adot.resource):
        # The import is the initialisation: ``openhands.sdk``'s package __init__
        # reaches modules that call ``maybe_init_laminar()`` at import (agent.py,
        # acp_agent.py), so importing even this one function has already built
        # Laminar's provider by the time it returns. Hence the import belongs inside
        # the pin, and the call after it is only the case where some other module
        # imported openhands with observability disabled.
        from openhands.sdk.observability.laminar import maybe_init_laminar

        maybe_init_laminar()

    from lmnr.opentelemetry_lib.tracing import TracerWrapper

    if not TracerWrapper.verify_initialized():
        logger.info("Laminar did not initialise; no OpenHands spans to route")
        return

    wrapper = TracerWrapper.instance
    lmnr_provider = wrapper._tracer_provider
    if lmnr_provider is None or lmnr_provider is adot:
        logger.info("Laminar is already using the global tracer provider; nothing to route")
        return

    _detach_span_processor(lmnr_provider, wrapper._span_processor)
    for processor in _span_processors(adot):
        lmnr_provider.add_span_processor(processor)

    logger.info(
        "routed OpenHands' Laminar spans through ADOT: %s",
        ", ".join(type(processor).__name__ for processor in _span_processors(lmnr_provider)),
    )


def _laminar_roots_under_current_span() -> None:
    """Have Laminar parent its next root at the span current on this thread.

    Laminar keeps a context of its own -- a ContextVar in
    ``lmnr.opentelemetry_lib.tracing.context``, deliberately isolated from OTel's --
    and parents ``Laminar.start_span`` from that rather than from the ambient context.
    So the conversation root comes out a *root* whatever the caller had current, and
    since the agent loop hangs off it through ``Laminar.use_span``, the whole
    conversation lands in a trace of its own. Seeding that isolated context with the
    current span -- the rollout's, see :func:`traced_background_work` -- is what puts
    the conversation under the invocation that asked for it.

    Per rollout rather than once, and here rather than at startup, because the isolated
    context is a ContextVar: the value has to be set on the thread that creates the
    conversation root, which is the rollout worker precisely because this is reached
    from ``run_rollout``. A root created on a thread OpenHands spawns later does not
    inherit it and would start a trace of its own. Nothing is detached -- the pool
    thread is there to run this one rollout.

    (``Laminar.use_span_context`` is the public API for the same effect, but it is a
    context manager, so using it would mean wrapping the rollout call itself. This
    keeps the tracing wiring in one place, at the cost of one internal import.)
    """
    span = trace.get_current_span()
    if not span.is_recording():
        logger.info("no span current on the rollout thread; OpenHands will trace a root of its own")
        return

    from lmnr.opentelemetry_lib.tracing.context import attach_context

    attach_context(trace.set_span_in_context(span))


@contextlib.contextmanager
def _laminar_resource_pinned_to(resource: Resource) -> Iterator[None]:
    """Make Laminar build its provider with ``resource``.

    ``Laminar.initialize`` sets the resource attributes it is about to build the
    provider from (``TracerWrapper.set_static_params``), overwriting anything set
    beforehand, so the seam is that call rather than the class attribute it writes.
    It has to be taken at construction time and cannot be repaired afterwards: an OTel
    ``Tracer`` copies the resource off its provider when it is created, and Laminar
    creates the auto-instrumentations' tracers while it initialises.
    """
    from lmnr.opentelemetry_lib.tracing import TracerWrapper

    original = TracerWrapper.set_static_params

    def pinned(resource_attributes: dict, enable_content_tracing: bool) -> None:
        original({**resource_attributes, **dict(resource.attributes)}, enable_content_tracing)

    TracerWrapper.set_static_params = staticmethod(pinned)
    try:
        yield
    finally:
        TracerWrapper.set_static_params = staticmethod(original)


def _sdk_tracer_provider() -> TracerProvider | None:
    """The global provider, if the SDK is installed on it.

    Under ``opentelemetry-instrument`` this is ADOT's provider. Without it the global
    provider is a no-op proxy, which is the case in a plain ``pytest`` run and on a
    developer node.
    """
    provider = trace.get_tracer_provider()
    return provider if isinstance(provider, TracerProvider) else None


def _span_processors(provider: TracerProvider) -> Sequence[SpanProcessor]:
    """The processors registered on ``provider``, in registration order."""
    return tuple(provider._active_span_processor._span_processors)


def _detach_span_processor(provider: TracerProvider, processor: SpanProcessor) -> None:
    """Remove one processor from ``provider``, leaving it otherwise untouched.

    The SDK offers no removal, only ``add_span_processor``; the multi-processor it
    keeps holds its children in a tuple behind a lock, which is what is rebuilt here.
    The processor is deliberately not shut down: it is handed to Laminar's
    instrumentations as well, and this only takes it off the export path.
    """
    multi = provider._active_span_processor
    with multi._lock:
        multi._span_processors = tuple(p for p in multi._span_processors if p is not processor)
