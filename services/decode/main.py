"""Decode service.

Owns the decode stage. Given a KV cache produced by a prefill node it first
"transfers" that cache into local (simulated) GPU memory - paying the transfer
overhead a colocated deployment avoids - then streams generated tokens using
the continuous-batching scheduler.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.responses import StreamingResponse

from common.batching import DecodeScheduler
from common.config import get_settings
from common.kv_transfer import simulate_transfer
from common.memory_sim import MemorySimulator
from common.metrics import (
    KV_TRANSFER_SECONDS,
    REQUEST_LATENCY,
    REQUESTS_TOTAL,
    TIME_TO_FIRST_TOKEN,
    mount_metrics,
)
from common.mock_model import MockModel
from common.models import BackendInfo, DecodeRequest, HealthResponse
from common.node import start_heartbeat
from common.redis_client import RedisState
from common.sse import sse_chunk, sse_done
from common.telemetry import get_tracer, setup_telemetry

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mini_dynamo.decode")

settings = get_settings()
settings.role = "decode"
if not settings.service_name or settings.service_name == "mini-dynamo":
    settings.service_name = "decode"

model = MockModel(settings)
memory = MemorySimulator(settings.gpu_memory_mb, settings.eviction_policy)
scheduler = DecodeScheduler(settings, model, memory, service_label="decode")
state = RedisState(settings)
tracer = get_tracer()

NODE_NAME = settings.service_name


def _backend_info() -> BackendInfo:
    return BackendInfo(
        name=NODE_NAME,
        url=settings.advertise_url or f"http://{NODE_NAME}:{settings.port}",
        role="decode",
        active_batch=scheduler.active_batch,
        queue_depth=scheduler.queue_depth,
        mem_used_mb=memory.used_mb,
        mem_total_mb=memory.total_mb,
        utilization=memory.utilization,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start()
    hb = start_heartbeat(state, settings, _backend_info)
    log.info("decode node '%s' up (policy=%s, mem=%.0fMB, batch<=%d)", NODE_NAME,
             settings.eviction_policy, settings.gpu_memory_mb, settings.max_batch_size)
    yield
    hb.cancel()
    await scheduler.stop()
    await state.close()


app = FastAPI(title="Mini-Dynamo Decode", lifespan=lifespan)
setup_telemetry(app, settings)
mount_metrics(app)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        role="decode",
        name=NODE_NAME,
        mode=settings.mode,
        mem_used_mb=round(memory.used_mb, 2),
        mem_total_mb=memory.total_mb,
        utilization=round(memory.utilization, 4),
        queue_depth=scheduler.queue_depth,
        active_batch=scheduler.active_batch,
    )


@app.get("/stats")
async def stats() -> dict:
    return memory.stats()


@app.post("/v1/decode")
async def decode(req: DecodeRequest):
    """Transfer the KV cache in, then stream generated tokens (SSE)."""
    max_tokens = req.max_tokens or settings.max_tokens_default

    # Resolve KV size from Redis metadata when the caller didn't provide it.
    kv_size_mb = req.kv_size_mb
    num_prompt_tokens = req.num_prompt_tokens
    if kv_size_mb <= 0:
        meta = await state.get_kv(req.kv_cache_id)
        if meta:
            kv_size_mb = float(meta.get("size_mb", 0.0))
            num_prompt_tokens = int(meta.get("num_tokens", 0))

    async def stream():
        t0 = time.perf_counter()
        with tracer.start_as_current_span("decode.generate") as span:
            span.set_attribute("mdyn.session_id", req.session_id)
            span.set_attribute("mdyn.kv_cache_id", req.kv_cache_id)
            span.set_attribute("mdyn.prefill_backend", req.prefill_backend)
            span.set_attribute("mdyn.kv_size_mb", kv_size_mb)

            # --- KV transfer overhead (prefill node -> this decode node) ---
            with tracer.start_as_current_span("decode.kv_transfer") as tspan:
                transferred_ms = await simulate_transfer(
                    kv_size_mb,
                    settings.kv_transfer_bandwidth_gbps,
                    settings.kv_transfer_fixed_ms,
                    colocated=settings.is_colocated,
                )
                tspan.set_attribute("mdyn.transfer_ms", transferred_ms)
            KV_TRANSFER_SECONDS.labels("decode").observe(transferred_ms / 1000.0)
            REQUEST_LATENCY.labels("decode", "transfer").observe(transferred_ms / 1000.0)

            decode_req = DecodeRequest(
                session_id=req.session_id,
                kv_cache_id=req.kv_cache_id,
                num_prompt_tokens=num_prompt_tokens,
                kv_size_mb=kv_size_mb,
                max_tokens=max_tokens,
                prefill_backend=req.prefill_backend,
            )
            out = await scheduler.submit(decode_req)
            first = True
            while True:
                chunk = await out.get()
                if first and chunk.token:
                    TIME_TO_FIRST_TOKEN.labels("decode").observe(time.perf_counter() - t0)
                    first = False
                yield sse_chunk(chunk)
                if chunk.finish_reason is not None:
                    span.set_attribute("mdyn.finish_reason", chunk.finish_reason)
                    break
            REQUEST_LATENCY.labels("decode", "e2e").observe(time.perf_counter() - t0)
            REQUESTS_TOTAL.labels("decode", "ok").inc()
        yield sse_done()

    return StreamingResponse(stream(), media_type="text/event-stream")
