"""Prometheus metrics shared by all services.

Exposes the signals the benchmark and Grafana dashboards care about:

* request/stage latency histograms (p50/p95 derived in Prometheus/Grafana),
* tokens generated (rate -> tokens/sec),
* queue depth, active batch size,
* KV cache utilization, evictions, and KV-transfer time.

Call :func:`mount_metrics` to expose ``/metrics`` on a FastAPI app.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

# One registry per process; all services import these same objects.
REGISTRY = CollectorRegistry()

_LAT_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8,
)

# End-to-end and per-stage latency (seconds). label: stage = e2e|prefill|decode|transfer
REQUEST_LATENCY = Histogram(
    "mdyn_request_latency_seconds",
    "Request/stage latency in seconds.",
    labelnames=("service", "stage"),
    buckets=_LAT_BUCKETS,
    registry=REGISTRY,
)

TIME_TO_FIRST_TOKEN = Histogram(
    "mdyn_ttft_seconds",
    "Time to first token in seconds.",
    labelnames=("service",),
    buckets=_LAT_BUCKETS,
    registry=REGISTRY,
)

TOKENS_TOTAL = Counter(
    "mdyn_tokens_total",
    "Total tokens generated.",
    labelnames=("service",),
    registry=REGISTRY,
)

REQUESTS_TOTAL = Counter(
    "mdyn_requests_total",
    "Total requests handled.",
    labelnames=("service", "outcome"),  # outcome = ok|error|fallback
    registry=REGISTRY,
)

QUEUE_DEPTH = Gauge(
    "mdyn_queue_depth",
    "Number of requests waiting to be scheduled.",
    labelnames=("service",),
    registry=REGISTRY,
)

ACTIVE_BATCH = Gauge(
    "mdyn_active_batch_size",
    "Number of sequences in the running batch.",
    labelnames=("service",),
    registry=REGISTRY,
)

CACHE_UTILIZATION = Gauge(
    "mdyn_cache_utilization_ratio",
    "KV cache pool utilization (0..1).",
    labelnames=("service",),
    registry=REGISTRY,
)

CACHE_USED_MB = Gauge(
    "mdyn_cache_used_mb",
    "KV cache memory used in MB.",
    labelnames=("service",),
    registry=REGISTRY,
)

EVICTIONS_TOTAL = Counter(
    "mdyn_evictions_total",
    "Total KV cache blocks evicted.",
    labelnames=("service",),
    registry=REGISTRY,
)

KV_TRANSFER_SECONDS = Histogram(
    "mdyn_kv_transfer_seconds",
    "Simulated KV cache transfer time between stages (seconds).",
    labelnames=("service",),
    buckets=_LAT_BUCKETS,
    registry=REGISTRY,
)

ROUTING_DECISIONS = Counter(
    "mdyn_routing_decisions_total",
    "Router decisions by kind.",
    labelnames=("kind",),  # kind = cache_hit|cache_miss|fallback
    registry=REGISTRY,
)


def mount_metrics(app) -> None:
    """Add a ``/metrics`` endpoint returning the Prometheus exposition format."""

    @app.get("/metrics")
    def metrics() -> Response:  # noqa: D401
        data = generate_latest(REGISTRY)
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)
