"""Tests for the container's tracing wiring.

Two halves. The first is ordinary unit coverage of the pieces
:mod:`swe_agent_server.observability` is built from, run in this process. The second
is one end-to-end check of the routed mode, run in a *subprocess*: it initialises the
real Laminar layer against a stand-in for ADOT's provider, and both of those are
process-global, one-shot things -- the OTel global provider can only be set once and
Laminar's ``TracerWrapper`` is a singleton whose resource is fixed at construction. A
child process is what keeps that check honest (nothing else in the run has touched
either global) and keeps it from deciding what every other test in the file sees.
"""

import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from swe_agent_server import observability
from swe_agent_server.observability import (
    SESSION_ID_ATTRIBUTE,
    SessionIdSpanProcessor,
    _detach_span_processor,
    _disarm_observability_gate,
    _laminar_resource_pinned_to,
    _span_processors,
    set_session_id,
    stop_instrumenting_child_processes,
)


class SessionIdSpanProcessorTest(unittest.TestCase):
    def test_stamps_the_process_session_id_on_every_span(self):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SessionIdSpanProcessor())
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        with mock.patch.object(observability, "_session_id", "session-1"):
            with provider.get_tracer("test").start_as_current_span("work"):
                pass

        (span,) = exporter.get_finished_spans()
        self.assertEqual(span.attributes[SESSION_ID_ATTRIBUTE], "session-1")

    def test_no_session_id_no_attribute(self):
        # Before the first invocation there is nothing to stamp, and an empty
        # attribute would be worse than an absent one: it matches no query.
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SessionIdSpanProcessor())
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        with mock.patch.object(observability, "_session_id", None):
            with provider.get_tracer("test").start_as_current_span("work"):
                pass

        (span,) = exporter.get_finished_spans()
        self.assertNotIn(SESSION_ID_ATTRIBUTE, span.attributes)


class SetSessionIdTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(observability, "_session_id", None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_records_the_id(self):
        set_session_id("session-1")
        self.assertEqual(observability._session_id, "session-1")

    def test_none_does_not_clear_a_known_id(self):
        # A call with no session id (a locally driven invocation) must not undo the
        # stamping for the rest of the container's life.
        set_session_id("session-1")
        set_session_id(None)
        self.assertEqual(observability._session_id, "session-1")

    def test_a_second_session_in_one_container_warns_and_takes_the_new_id(self):
        set_session_id("session-1")
        with self.assertLogs(observability.logger, level="WARNING") as logs:
            set_session_id("session-2")

        self.assertEqual(observability._session_id, "session-2")
        self.assertIn("session id changed", logs.output[0])


class StopInstrumentingChildProcessesTest(unittest.TestCase):
    """The one PYTHONPATH entry that must not reach the agent's shell commands."""

    AUTO = "/agent/.venv/lib/python3.13/site-packages/opentelemetry/instrumentation/auto_instrumentation"

    def environ(self, pythonpath: str | None) -> dict:
        return {} if pythonpath is None else {"PYTHONPATH": pythonpath}

    def test_removes_the_auto_instrumentation_entry_and_keeps_the_rest(self):
        with mock.patch.dict(observability.os.environ, self.environ(f"{self.AUTO}:/agent"), clear=True):
            stop_instrumenting_child_processes()

            self.assertEqual(observability.os.environ["PYTHONPATH"], "/agent")

    def test_a_trailing_slash_is_still_the_same_directory(self):
        with mock.patch.dict(observability.os.environ, self.environ(f"/agent:{self.AUTO}/"), clear=True):
            stop_instrumenting_child_processes()

            self.assertEqual(observability.os.environ["PYTHONPATH"], "/agent")

    def test_nothing_else_left_means_unset_not_empty(self):
        # An empty PYTHONPATH puts the working directory on sys.path, which is a
        # different behaviour change to leak into the agent's commands.
        with mock.patch.dict(observability.os.environ, self.environ(self.AUTO), clear=True):
            stop_instrumenting_child_processes()

            self.assertNotIn("PYTHONPATH", observability.os.environ)

    def test_an_unrelated_pythonpath_is_left_alone(self):
        with mock.patch.dict(observability.os.environ, self.environ("/agent:/opt/things"), clear=True):
            stop_instrumenting_child_processes()

            self.assertEqual(observability.os.environ["PYTHONPATH"], "/agent:/opt/things")

    def test_no_pythonpath_at_all_is_not_an_error(self):
        with mock.patch.dict(observability.os.environ, {}, clear=True):
            stop_instrumenting_child_processes()

            self.assertNotIn("PYTHONPATH", observability.os.environ)


class ChildProcessInheritanceTest(unittest.TestCase):
    """What the agent's own tool calls see, in a real child interpreter.

    The failure this guards against is not raised anywhere: it is a line printed on the
    child's stderr, which in a rollout goes into the observation the model is trained on.
    So the assertion has to be made against a real subprocess, and the stand-in for the
    task environment's interpreter is one that cannot import ``opentelemetry``.
    """

    def child_stderr(self, pythonpath: str) -> str:
        # A deliberately minimal environment: what is asserted about is the inherited
        # variable, so nothing else the test runner happens to export should be in play.
        result = subprocess.run(
            [sys.executable, "-c", "print(1 + 1)"],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", **({"PYTHONPATH": pythonpath} if pythonpath else {})},
        )
        return result.stderr

    def test_a_bad_auto_instrumentation_entry_pollutes_a_child_until_it_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            # The shape of the distro's entry: a directory whose sitecustomize imports
            # something the child cannot import. Here the import is guaranteed to fail,
            # which is what a task interpreter without opentelemetry reproduces.
            auto = Path(tmp) / "opentelemetry" / "instrumentation" / "auto_instrumentation"
            auto.mkdir(parents=True)
            (auto / "sitecustomize.py").write_text("import opentelemetry_not_installed_here\n")

            polluted = self.child_stderr(str(auto))
            self.assertIn("Error in sitecustomize", polluted)

            with mock.patch.dict(observability.os.environ, {"PYTHONPATH": str(auto)}, clear=True):
                stop_instrumenting_child_processes()
                cleaned = self.child_stderr(observability.os.environ.get("PYTHONPATH", ""))

            self.assertNotIn("Error in sitecustomize", cleaned)


class DetachSpanProcessorTest(unittest.TestCase):
    def test_removes_only_the_named_processor(self):
        provider = TracerProvider()
        first, second = SessionIdSpanProcessor(), SessionIdSpanProcessor()
        provider.add_span_processor(first)
        provider.add_span_processor(second)

        _detach_span_processor(provider, first)

        self.assertEqual(_span_processors(provider), (second,))

    def test_detached_processor_stops_seeing_spans(self):
        # This is what stops Laminar's own exporter, and with it the 403 loop.
        provider = TracerProvider()
        exporter = InMemorySpanExporter()
        processor = SimpleSpanProcessor(exporter)
        provider.add_span_processor(processor)

        _detach_span_processor(provider, processor)
        with provider.get_tracer("test").start_as_current_span("work"):
            pass

        self.assertEqual(exporter.get_finished_spans(), ())


class TracedBackgroundWorkTest(unittest.TestCase):
    """The span that ties setup and a rollout to the invocation that asked for them.

    ``_tracer`` is patched rather than set globally: the module resolves it through the
    global provider, which this process must leave unset for the subprocess tests below
    to mean anything.
    """

    def setUp(self):
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        patcher = mock.patch.object(observability, "_tracer", self.provider.get_tracer("test"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def finished(self) -> dict:
        return {span.name: span for span in self.exporter.get_finished_spans()}

    def test_the_work_runs_under_a_child_of_the_calling_span(self):
        # The shape the invocation handler produces: the span is created while the
        # request's server span is current, and only then handed to a worker thread.
        with self.provider.get_tracer("test").start_as_current_span("POST /invocations") as invocation:
            work = observability.traced_background_work("rollout_start", lambda: "dumped")

        with ThreadPoolExecutor(max_workers=1) as executor:
            self.assertEqual(executor.submit(work).result(), "dumped")

        rollout = self.finished()["rollout_start"]
        self.assertEqual(rollout.parent.span_id, invocation.context.span_id)
        self.assertEqual(rollout.context.trace_id, invocation.context.trace_id)

    def test_spans_started_by_the_work_descend_from_it(self):
        # What the agent loop does, and what a rootless trace means it fails to do.
        tracer = self.provider.get_tracer("test")

        def work() -> None:
            with tracer.start_as_current_span("agent_loop"):
                pass

        with tracer.start_as_current_span("POST /invocations"):
            wrapped = observability.traced_background_work("rollout_start", work)

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(wrapped).result()

        finished = self.finished()
        self.assertEqual(finished["agent_loop"].parent.span_id, finished["rollout_start"].context.span_id)
        self.assertEqual(finished["agent_loop"].context.trace_id, finished["rollout_start"].context.trace_id)

    def test_the_worker_thread_is_left_as_it_was_found(self):
        # The pool thread is reused for the next rollout, so the attached context has
        # to come back off it.
        def work() -> object:
            return trace.get_current_span().get_span_context().span_id

        wrapped = observability.traced_background_work("rollout_start", work)
        with ThreadPoolExecutor(max_workers=1) as executor:
            inside = executor.submit(wrapped).result()
            after = executor.submit(lambda: trace.get_current_span().get_span_context().span_id).result()

        self.assertEqual(inside, self.finished()["rollout_start"].context.span_id)
        self.assertEqual(after, trace.INVALID_SPAN_ID)

    def test_a_failure_is_recorded_on_the_span_and_still_raised(self):
        error = RuntimeError("swe_unpack.sh failed")

        def work() -> None:
            raise error

        wrapped = observability.traced_background_work("rollout_setup", work)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(wrapped)
            # The status call reports this to the caller, so the exception must survive.
            self.assertIs(future.exception(), error)

        span = self.finished()["rollout_setup"]
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        self.assertEqual(span.events[0].name, "exception")
        self.assertEqual(span.events[0].attributes["exception.message"], "swe_unpack.sh failed")


class DisarmObservabilityGateTest(unittest.TestCase):
    def test_unsets_every_variable_openhands_gates_on(self):
        environ = {
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "https://xray.us-west-2.amazonaws.com/v1/traces",
            "OTEL_EXPORTER_OTLP_TRACES_HEADERS": "x-aws-log-stream=spans",
        }
        with mock.patch.dict(observability.os.environ, environ, clear=True):
            _disarm_observability_gate()

            self.assertNotIn("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", observability.os.environ)
            # Only the gate's variables go: ADOT's headers (and everything else it
            # reads) are none of this function's business.
            self.assertIn("OTEL_EXPORTER_OTLP_TRACES_HEADERS", observability.os.environ)

    def test_an_unarmed_gate_is_not_an_error(self):
        with mock.patch.dict(observability.os.environ, {}, clear=True):
            with self.assertLogs(observability.logger, level="INFO"):
                _disarm_observability_gate()


class LaminarResourcePinnedToTest(unittest.TestCase):
    """The seam that decides the resource of the provider Laminar builds.

    Exercised against the real ``TracerWrapper`` (the class attribute is what
    ``Laminar.initialize`` writes) without initialising it, so the class is left as it
    was found.
    """

    def setUp(self):
        from lmnr.opentelemetry_lib.tracing import TracerWrapper

        self.wrapper = TracerWrapper
        original = TracerWrapper.resource_attributes
        self.addCleanup(setattr, TracerWrapper, "resource_attributes", original)

    def test_pinned_attributes_win_over_laminars_own(self):
        resource = Resource.create({"service.name": "adot-service"})
        with _laminar_resource_pinned_to(resource):
            # What Laminar.initialize does: name the service after sys.argv[0].
            self.wrapper.set_static_params({"service.name": "uvicorn"}, True)

        self.assertEqual(self.wrapper.resource_attributes["service.name"], "adot-service")

    def test_the_patch_is_reverted(self):
        original = self.wrapper.set_static_params
        with _laminar_resource_pinned_to(Resource.create({"service.name": "adot-service"})):
            pass

        self.wrapper.set_static_params({"service.name": "uvicorn"}, True)
        self.assertEqual(self.wrapper.resource_attributes["service.name"], "uvicorn")
        self.assertEqual(self.wrapper.set_static_params, original)


class ConfigureOpenhandsTracingTest(unittest.TestCase):
    """Both modes, each in a clean interpreter, each asserting for itself."""

    def _run_child(self, name: str) -> None:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), name],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    def test_routed_openhands_spans_reach_the_adot_pipeline(self):
        # Four properties the routed mode exists for: Laminar's spans are exported by
        # ADOT's processors, with ADOT's resource, carrying session.id, and Laminar's
        # own exporter is off the path.
        self._run_child("route_laminar_child")

    def test_off_leaves_laminar_dormant(self):
        self._run_child("disarm_gate_child")


def _stand_in_for_the_container() -> InMemorySpanExporter:
    """Set up what the deployed container's interpreter looks like.

    That is: a real SDK tracer provider on the global, which is what
    ``opentelemetry-instrument`` leaves behind (here exporting to memory rather than to
    X-Ray), and the endpoint variable the AWS distro sets in-process -- which is the
    thing that arms OpenHands' observability gate. The endpoint is never reached:
    Laminar's exporter is detached before any span is created, and the OTLP exporters
    connect lazily.
    """
    import os

    from opentelemetry import trace

    os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"
    os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://localhost:1/v1/traces"

    exporter = InMemorySpanExporter()
    adot = TracerProvider(resource=Resource.create({"service.name": "adot-service"}))
    adot.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(adot)

    observability.configure_tracing()
    return exporter


def route_laminar_child() -> None:
    """Routed mode, end to end, against the real OpenHands and Laminar packages."""
    from opentelemetry import trace

    exporter = _stand_in_for_the_container()
    observability.set_session_id("session-1")

    def rollout() -> None:
        """What ``run_rollout`` does, on the thread it does it on.

        The tracing is configured here, on the rollout worker, because that is where
        the OpenHands import happens -- and then the shape OpenHands produces: a
        long-lived root span, re-attached so that spans started by anything else (here
        the global tracer, in the container ADOT's httpx instrumentation) are its
        children. All of it has to descend from the invocation that submitted the work.
        """
        observability.configure_openhands_tracing(mode=observability.MODE_ADOT)

        from lmnr import Laminar
        from lmnr.opentelemetry_lib.tracing import TracerWrapper

        wrapper = TracerWrapper.instance
        lmnr_provider = wrapper._tracer_provider
        assert lmnr_provider is not trace.get_tracer_provider(), "Laminar kept its own provider; the case fixed"
        assert wrapper._span_processor not in _span_processors(lmnr_provider), "Laminar's own exporter is attached"

        root = Laminar.start_span("conversation.run")
        with Laminar.use_span(root):
            with trace.get_tracer("child").start_as_current_span("POST"):
                pass
        root.end()

    with trace.get_tracer("app").start_as_current_span("POST /invocations"):
        work = observability.traced_background_work("rollout_start", rollout)
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(work).result()

    finished = {span.name: span for span in exporter.get_finished_spans()}
    expected = {"POST /invocations", "rollout_start", "conversation.run", "POST"}
    assert set(finished) == expected, f"exported: {sorted(finished)}"

    invocation, rollout_span = finished["POST /invocations"], finished["rollout_start"]
    conversation, child = finished["conversation.run"], finished["POST"]
    assert conversation.resource.attributes["service.name"] == "adot-service", conversation.resource.attributes
    assert conversation.attributes[SESSION_ID_ATTRIBUTE] == "session-1", conversation.attributes
    assert child.attributes[SESSION_ID_ATTRIBUTE] == "session-1", child.attributes
    assert child.parent.span_id == conversation.context.span_id, "the ADOT child is not parented to Laminar's root"
    assert conversation.parent.span_id == rollout_span.context.span_id, "Laminar's root is not under the rollout span"
    assert rollout_span.parent.span_id == invocation.context.span_id, "the rollout span is not under the invocation"
    assert {span.context.trace_id for span in finished.values()} == {invocation.context.trace_id}, "traces differ"

    print("OK")


def disarm_gate_child() -> None:
    """``off`` mode, end to end: OpenHands imports, and Laminar stays out of it."""
    from opentelemetry import trace

    exporter = _stand_in_for_the_container()
    observability.set_session_id("session-1")
    observability.configure_openhands_tracing(mode=observability.MODE_OFF)

    # The import that would otherwise initialise Laminar (see the routed child).
    from openhands.sdk.observability.laminar import should_enable_observability

    assert not should_enable_observability(), "the gate is still armed"

    from lmnr.opentelemetry_lib.tracing import TracerWrapper

    assert not TracerWrapper.verify_initialized(), "Laminar initialised anyway"

    # And the container's own pipeline is untouched: spans still export, still stamped.
    with trace.get_tracer("child").start_as_current_span("POST"):
        pass

    (span,) = exporter.get_finished_spans()
    assert span.attributes[SESSION_ID_ATTRIBUTE] == "session-1", span.attributes

    print("OK")


# The two subprocess halves above, by the name the parent test passes.
_CHILDREN = {
    "route_laminar_child": route_laminar_child,
    "disarm_gate_child": disarm_gate_child,
}


if __name__ == "__main__":
    # pytest never takes this path.
    (name,) = sys.argv[1:]
    _CHILDREN[name]()
