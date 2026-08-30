"""OpenTelemetry tracing setup.

Instruments FastAPI and outbound httpx so a single request produces a
connected trace across router -> prefill -> decode. Exporting is best-effort:
if no collector is reachable the app still runs (spans are simply dropped),
which keeps local development friction-free.
"""

from __future__ import annotations

import logging

from .config import Settings

log = logging.getLogger("mini_dynamo.telemetry")

_tracer = None


def setup_telemetry(app, settings: Settings):
    """Configure a tracer provider + OTLP exporter and instrument ``app``."""
    global _tracer
    if not settings.otel_enabled:
        log.info("OpenTelemetry disabled (OTEL_ENABLED=false)")
        return None

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create(
            {"service.name": settings.service_name, "mdyn.role": settings.role}
        )
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app)
        HTTPXClientInstrumentor().instrument()

        _tracer = trace.get_tracer("mini_dynamo")
        log.info("OpenTelemetry -> %s", settings.otel_endpoint)
        return _tracer
    except Exception as exc:  # pragma: no cover - observability is optional
        log.warning("OpenTelemetry setup failed (%s); continuing without traces", exc)
        return None


def get_tracer():
    """Return the configured tracer, or a no-op tracer if unavailable."""
    global _tracer
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace

        return trace.get_tracer("mini_dynamo")
    except Exception:  # pragma: no cover
        return _NoopTracer()


class _NoopSpan:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def set_attribute(self, *a, **k):
        pass

    def add_event(self, *a, **k):
        pass


class _NoopTracer:
    def start_as_current_span(self, *a, **k):
        return _NoopSpan()
