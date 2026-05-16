"""Structured JSON logs, tagged with the active OTel trace_id.

The tag is what makes a failed query findable in both places by the same
key: structlog gives the JSON log line, OTel gives the Jaeger trace, and
_add_trace_id is the one processor that ties a given log statement to
whichever span was current when it was emitted -- without every call site
having to thread a trace id through by hand.
"""

import logging

import structlog
from opentelemetry import trace


def _add_trace_id(logger, method_name, event_dict):
    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
    return event_dict


def configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_trace_id,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
