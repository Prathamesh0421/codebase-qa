"""obs/tracing.py and obs/logging.py are thin wiring, but the wiring itself
is the interesting part: a span created before configure_tracing() runs
must still nest correctly once it does (OTel's proxy-tracer contract), and
a log line emitted while a span is active must carry that span's trace_id
so the two can be correlated in Jaeger and a log aggregator by the same key.
"""

import json

import pytest
import structlog
from opentelemetry import trace

from codeqa.obs.logging import configure_logging
from codeqa.obs.tracing import configure_tracing, get_tracer


@pytest.fixture(autouse=True)
def reset_structlog():
    # structlog's global config is reset around each test so one test's
    # configure_logging call can't leak into another's. The OTel tracer
    # provider is NOT reset here -- the SDK deliberately refuses to replace
    # a real provider once set (a global, one-time, process-lifetime
    # configuration, matching how it actually behaves in production), so
    # tests are written to tolerate a real provider persisting from an
    # earlier test rather than fighting that restriction.
    yield
    structlog.reset_defaults()


class TestTracing:
    def test_get_tracer_before_configure_is_a_safe_no_op(self):
        # A span created before any exporter is configured must not raise --
        # this is what lets agents/nodes.py create spans unconditionally
        # regardless of whether it's driven by the CLI (never configured)
        # or the API (configured at startup).
        tracer = get_tracer("test")
        with tracer.start_as_current_span("locate") as span:
            span.set_attribute("codeqa.chunks_found", 3)

    def test_configure_tracing_is_idempotent(self):
        configure_tracing("codeqa-test", "http://localhost:4317")
        first_provider = trace.get_tracer_provider()
        configure_tracing("codeqa-test", "http://localhost:4317")
        assert trace.get_tracer_provider() is first_provider

    def test_no_endpoint_leaves_tracing_unconfigured(self):
        configure_tracing("codeqa-test", None)
        # No exporter wired up -- get_tracer() still returns something
        # span-creation-safe, same as the never-configured case.
        tracer = get_tracer("test")
        with tracer.start_as_current_span("locate"):
            pass


class TestLogging:
    def test_a_log_emitted_inside_a_span_carries_its_trace_id(self, capsys):
        # A NoOp span (no tracer provider configured) has an invalid,
        # all-zero trace_id by design -- configure_tracing is required here
        # to get a real, non-zero trace_id for _add_trace_id to attach.
        configure_tracing("codeqa-test", "http://localhost:4317")
        configure_logging("INFO")
        tracer = get_tracer("test")
        log = structlog.get_logger()

        with tracer.start_as_current_span("query") as span:
            expected_trace_id = format(span.get_span_context().trace_id, "032x")
            log.info("query.start", question="how does this work")

        line = json.loads(capsys.readouterr().out.strip())
        assert line["trace_id"] == expected_trace_id
        assert line["event"] == "query.start"
        assert line["question"] == "how does this work"

    def test_a_log_emitted_outside_any_span_has_no_trace_id(self, capsys):
        configure_logging("INFO")
        structlog.get_logger().info("worker.tick")

        line = json.loads(capsys.readouterr().out.strip())
        assert "trace_id" not in line

    def test_log_level_filters_below_the_configured_threshold(self, capsys):
        configure_logging("WARNING")
        log = structlog.get_logger()
        log.info("this should be filtered out")
        log.warning("this should appear")

        output = capsys.readouterr().out.strip()
        lines = [json.loads(line) for line in output.splitlines()]
        assert len(lines) == 1
        assert lines[0]["event"] == "this should appear"
