"""Make the agent loop's spans reach CloudWatch, and be findable by session.

The container runs under ``opentelemetry-instrument``, so AWS's distro owns the global
tracer provider. This module stamps ``session.id`` on spans started off the request
thread and routes OpenHands' Laminar spans through ADOT's signing exporter (otherwise
they 403). Failures here are logged and swallowed: they must never break a rollout.
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

# Resolved lazily on each use, so importing before the provider is installed is fine.
_tracer = trace.get_tracer(__name__)

# What CloudWatch's session view and CloudWatchAgentSpanCollector filter on -- not
# Laminar's ``session_id`` property, which OpenHands fills with the conversation id.
SESSION_ID_ATTRIBUTE = "session.id"

# The directory ``opentelemetry-instrument`` puts on PYTHONPATH to instrument children.
_AUTO_INSTRUMENTATION_DIR = os.path.join("opentelemetry", "instrumentation", "auto_instrumentation")

MODE_ENV = "SWE_AGENT_OPENHANDS_TRACING"
MODE_ADOT = "adot"
MODE_OFF = "off"

# The image is deployed with no environment of its own, so this default is what runs;
# set MODE_ENV to "off" on the runtime to leave the Laminar layer dormant instead.
DEFAULT_MODE = MODE_ADOT

# Any one of these being set arms OpenHands' whole observability layer
# (``should_enable_observability()``).
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

    One session id routes to one container for its whole life, so this is process-wide
    state; a second, different id breaks that assumption and is worth a warning.
    """
    global _session_id

    if session_id is None or session_id == _session_id:
        return
    if _session_id is not None:
        logger.warning("session id changed within one container: %s -> %s", _session_id, session_id)
    _session_id = session_id


class SessionIdSpanProcessor(SpanProcessor):
    """Stamps :data:`SESSION_ID_ATTRIBUTE` on every span, on any thread.

    The distro's ``BaggageSpanProcessor`` only reaches spans started on the request
    thread; this reads the process-wide id, so the agent loop's spans get it too.
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

    The inherited entry holds a ``sitecustomize.py`` that imports ``opentelemetry``, so
    every ``python`` the agent runs pollutes stderr with a ``ModuleNotFoundError`` (the
    testbed env has no ``opentelemetry``) -- noise in the trajectory, and a wrong reward
    for any test comparing stderr exactly. Where it *is* importable it is worse: the
    child gets fully instrumented and exports spans of its own.

    Only that one entry goes; children still need the rest (``/agent``). Safe because
    this process was instrumented at interpreter startup, long before this import.
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
        # An empty PYTHONPATH is not an unset one: it puts cwd on sys.path.
        del os.environ["PYTHONPATH"]
    logger.info("removed the auto-instrumentation from PYTHONPATH for child processes: %r -> %r", path, kept)


def traced_background_work(name: str, work: Callable[[], object]) -> Callable[[], object]:
    """Wrap ``work`` so what it traces belongs to the invocation that submitted it.

    Must be called on the request thread, where the invocation's server span is current:
    the span is started here and its context attached on the pool thread. The span then
    outlives the invocation by minutes, which is well-formed -- the invocation span is
    exported immediately, so the trace has a root even if this one is never exported.
    """
    span = _tracer.start_span(name)
    context = trace.set_span_in_context(span)

    def run() -> object:
        token = otel_context.attach(context)
        try:
            return work()
        except Exception as error:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            raise
        finally:
            otel_context.detach(token)
            span.end()

    return run


def configure_openhands_tracing(mode: str | None = None) -> None:
    """Settle what happens to OpenHands' Laminar layer, before it initialises.

    Must be called before anything imports ``openhands.sdk``, because that import *is*
    the initialisation (module-level ``maybe_init_laminar()`` calls). Idempotent, and in
    the routed mode it performs the import itself so the wiring lives in one place.
    """
    global _openhands_tracing_configured

    mode = mode or os.environ.get(MODE_ENV, DEFAULT_MODE)
    try:
        if not _openhands_tracing_configured:
            # The routed mode copies the global provider's processors, so the session-id
            # stamper has to be among them by now.
            configure_tracing()

            if mode == MODE_OFF:
                _disarm_observability_gate()
            elif mode == MODE_ADOT:
                _route_laminar_through_adot()
            else:
                raise ValueError(f"{MODE_ENV} must be {MODE_ADOT!r} or {MODE_OFF!r}, got {mode!r}")
            _openhands_tracing_configured = True

        if mode == MODE_ADOT:
            # Per call, not once: this one is thread-local to the rollout being started.
            _laminar_roots_under_current_span()
    except Exception:
        logger.error("failed to configure OpenHands tracing (mode=%s)", mode, exc_info=True)


def _disarm_observability_gate() -> None:
    """Leave OpenHands' observability layer dormant, and say so in the log.

    The arming variable is one the AWS distro sets at interpreter startup, so it has to
    be unset here -- and left unset, since ``should_enable_observability()`` is re-read
    by every ``@observe`` call. ADOT's exporter captured its endpoint long before this.
    """
    dropped = [key for key in _OBSERVABILITY_ENV_KEYS if os.environ.pop(key, None) is not None]
    logger.info("OpenHands tracing off: unset %s", ", ".join(dropped) or "nothing (gate was already unarmed)")


def _route_laminar_through_adot() -> None:
    """Give Laminar's spans ADOT's resource and ADOT's signing exporter.

    Three edits to the provider Laminar keeps for itself: build it with ADOT's resource
    (Laminar would name the service after ``sys.argv[0]``, i.e. uvicorn); attach ADOT's
    processors, so a Laminar span is exported exactly as an ADOT one; detach Laminar's
    own processor, which is what stops the 403 loop. Only the export path goes -- the
    processor object stays alive for the instrumentations holding it directly.
    """
    adot = _sdk_tracer_provider()
    if adot is None:
        logger.info("no SDK tracer provider; leaving OpenHands' tracing as configured")
        return

    with _laminar_resource_pinned_to(adot.resource):
        # The import is the initialisation, so it belongs inside the pin; the call after
        # it only covers openhands having been imported with observability disabled.
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

    Laminar parents ``start_span`` from a ContextVar of its own, isolated from OTel's,
    so the conversation root would otherwise start a trace of its own. Seeding that
    context must happen on the thread that creates the root, hence per rollout.
    (``Laminar.use_span_context`` is the public equivalent, but it is a context manager
    and would mean wrapping the rollout call itself.)
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

    ``set_static_params`` is the seam because ``Laminar.initialize`` overwrites the class
    attribute it writes. It cannot be repaired afterwards: an OTel ``Tracer`` copies the
    resource off its provider at creation, and Laminar creates tracers while it inits.
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
    """The global provider, if the SDK is installed on it (a no-op proxy otherwise)."""
    provider = trace.get_tracer_provider()
    return provider if isinstance(provider, TracerProvider) else None


def _span_processors(provider: TracerProvider) -> Sequence[SpanProcessor]:
    """The processors registered on ``provider``, in registration order."""
    return tuple(provider._active_span_processor._span_processors)


def _detach_span_processor(provider: TracerProvider, processor: SpanProcessor) -> None:
    """Remove one processor from ``provider``, leaving it otherwise untouched.

    The SDK offers no removal, hence reaching into the multi-processor's tuple. The
    processor is deliberately not shut down -- Laminar's instrumentations still hold it.
    """
    multi = provider._active_span_processor
    with multi._lock:
        multi._span_processors = tuple(p for p in multi._span_processors if p is not processor)
