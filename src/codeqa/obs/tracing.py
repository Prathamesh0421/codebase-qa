"""OpenTelemetry wiring: one TracerProvider per process, configured once at
startup by whatever entrypoint has a Settings object (currently only
api/app.py).

Node functions in agents/nodes.py call get_tracer() and open spans
unconditionally, with no "is tracing configured" check at each call site --
before configure_tracing() runs (the CLI path, or any test), OTel's default
global tracer provider is a no-op, so those spans are created and dropped for
free. This is what lets the locate/trace/synthesize spans show up nested
under a request span in the API without agents/nodes.py needing to know
whether it's being driven by the CLI or the API.
"""

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_configured = False


def configure_tracing(service_name: str, otel_endpoint: str | None) -> None:
    """No-op if already configured (import can happen more than once under a
    reloader or in tests) or if no collector endpoint is set -- an
    unconfigured deployment gets no-op spans instead of a crash at startup,
    the same "degrade, don't crash" stance as the rest of this phase.
    """
    global _configured
    if _configured or not otel_endpoint:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    exporter = OTLPSpanExporter(endpoint=otel_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _configured = True


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)
