"""Environment-driven configuration.

Every knob the platform exposes is read from an environment variable so the
same image can be run as a router, prefill, or decode node, and so memory
limits / eviction policy / latency characteristics can be tuned per-deployment
(Docker Compose, Kubernetes) without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List


def _get_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _get_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _get_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _get_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_list(key: str, default: List[str]) -> List[str]:
    raw = os.getenv(key)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class Settings:
    # --- Identity / networking -------------------------------------------
    service_name: str = "mini-dynamo"
    role: str = "router"  # router | prefill | decode
    host: str = "0.0.0.0"
    port: int = 8000
    # URL this node advertises to the router via the Redis registry.
    advertise_url: str = ""

    # --- Shared state -----------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"

    # --- Execution mode ---------------------------------------------------
    # disaggregated: prefill and decode run on separate services and KV cache
    #   is transferred between them.
    # colocated: a single node runs prefill+decode in-process (no transfer).
    mode: str = "disaggregated"

    # --- Simulated GPU memory --------------------------------------------
    gpu_memory_mb: float = 8192.0
    kv_mb_per_token: float = 0.5          # KV cache footprint per token
    eviction_policy: str = "lru"          # lru | fifo | none

    # --- KV transfer cost model ------------------------------------------
    kv_transfer_bandwidth_gbps: float = 25.0   # effective link bandwidth
    kv_transfer_fixed_ms: float = 2.0          # fixed per-transfer overhead

    # --- Mock model latency ----------------------------------------------
    prefill_base_ms: float = 15.0
    prefill_ms_per_token: float = 0.4
    decode_ms_per_token: float = 12.0
    latency_jitter: float = 0.15          # +/- fraction of jitter on latency

    # --- Continuous batching ---------------------------------------------
    max_batch_size: int = 16
    max_tokens_default: int = 64

    # --- Service discovery ------------------------------------------------
    # Static fallback lists used when the Redis registry is empty.
    prefill_urls: List[str] = field(default_factory=list)
    decode_urls: List[str] = field(default_factory=list)
    heartbeat_ttl: int = 10               # seconds; registry entries expire
    heartbeat_interval: float = 3.0       # seconds between heartbeats

    # --- Observability ----------------------------------------------------
    otel_enabled: bool = True
    otel_endpoint: str = "http://localhost:4317"
    session_ttl: int = 3600               # sticky session->backend TTL (s)

    @property
    def is_colocated(self) -> bool:
        return self.mode.lower() == "colocated"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build a Settings instance from the current environment (cached)."""
    return Settings(
        service_name=_get_str("SERVICE_NAME", "mini-dynamo"),
        role=_get_str("ROLE", "router"),
        host=_get_str("HOST", "0.0.0.0"),
        port=_get_int("PORT", 8000),
        advertise_url=_get_str("ADVERTISE_URL", ""),
        redis_url=_get_str("REDIS_URL", "redis://localhost:6379/0"),
        mode=_get_str("MODE", "disaggregated"),
        gpu_memory_mb=_get_float("GPU_MEMORY_MB", 8192.0),
        kv_mb_per_token=_get_float("KV_MB_PER_TOKEN", 0.5),
        eviction_policy=_get_str("EVICTION_POLICY", "lru"),
        kv_transfer_bandwidth_gbps=_get_float("KV_TRANSFER_BANDWIDTH_GBPS", 25.0),
        kv_transfer_fixed_ms=_get_float("KV_TRANSFER_FIXED_MS", 2.0),
        prefill_base_ms=_get_float("PREFILL_BASE_MS", 15.0),
        prefill_ms_per_token=_get_float("PREFILL_MS_PER_TOKEN", 0.4),
        decode_ms_per_token=_get_float("DECODE_MS_PER_TOKEN", 12.0),
        latency_jitter=_get_float("LATENCY_JITTER", 0.15),
        max_batch_size=_get_int("MAX_BATCH_SIZE", 16),
        max_tokens_default=_get_int("MAX_TOKENS_DEFAULT", 64),
        prefill_urls=_get_list("PREFILL_URLS", []),
        decode_urls=_get_list("DECODE_URLS", []),
        heartbeat_ttl=_get_int("HEARTBEAT_TTL", 10),
        heartbeat_interval=_get_float("HEARTBEAT_INTERVAL", 3.0),
        otel_enabled=_get_bool("OTEL_ENABLED", True),
        otel_endpoint=_get_str("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
        session_ttl=_get_int("SESSION_TTL", 3600),
    )
