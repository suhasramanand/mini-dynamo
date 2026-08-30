"""Prefill service.

Owns the prefill stage: turn a prompt into a KV cache, allocate simulated GPU
memory for it, and register its metadata in Redis so the decode stage can find
and "transfer" it. Also exposes a colocated ``/v1/generate`` path (prefill +
decode in one process, no KV transfer) used as the benchmark baseline.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.responses import StreamingResponse

from common.batching import DecodeScheduler
from common.config import get_settings
from common.memory_sim import MemorySimulator, OutOfMemoryError
from common.metrics import (
    CACHE_UTILIZATION,
    EVICTIONS_TOTAL,
    REQUEST_LATENCY,
    REQUESTS_TOTAL,
    mount_metrics,
)
from common.mock_model import MockModel
from common.models import (
    DecodeRequest,
    GenerateRequest,
    HealthResponse,
    PrefillRequest,
    PrefillResponse,
)
from common.node import start_heartbeat
from common.redis_client import RedisState
from common.sse import sse_chunk, sse_done
from common.telemetry import get_tracer, setup_telemetry

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mini_dynamo.prefill")

settings = get_settings()
settings.role = "prefill"
if not settings.service_name or settings.service_name == "mini-dynamo":
    settings.service_name = "prefill"

model = MockModel(settings)
memory = MemorySimulator(settings.gpu_memory_mb, settings.eviction_policy)
# Local scheduler only used by the colocated generate path.
scheduler = DecodeScheduler(settings, model, memory, service_label="prefill")
state = RedisState(settings)
tracer = get_tracer()

NODE_NAME = settings.service_name


def _backend_info():
    from common.models import BackendInfo

    return BackendInfo(
        name=NODE_NAME,
        url=settings.advertise_url or f"http://{NODE_NAME}:{settings.port}",
        role="prefill",
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
    log.info("prefill node '%s' up (policy=%s, mem=%.0fMB)", NODE_NAME,
             settings.eviction_policy, settings.gpu_memory_mb)
    yield
    hb.cancel()
    await scheduler.stop()
    await state.close()


app = FastAPI(title="Mini-Dynamo Prefill", lifespan=lifespan)
setup_telemetry(app, settings)
mount_metrics(app)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        role="prefill",
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


@app.post("/v1/prefill", response_model=PrefillResponse)
async def prefill(req: PrefillRequest) -> PrefillResponse:
    """Compute a KV cache for ``prompt`` and register it for the decode stage."""
    t0 = time.perf_counter()
    with tracer.start_as_current_span("prefill.compute") as span:
        span.set_attribute("mdyn.session_id", req.session_id)
        result = await model.prefill(req.session_id, req.prompt)
        span.set_attribute("mdyn.prompt_tokens", result.num_prompt_tokens)
        span.set_attribute("mdyn.kv_size_mb", result.kv_size_mb)

        try:
            evicted = memory.allocate(
                result.kv_cache_id,
                req.session_id,
                result.num_prompt_tokens,
                result.kv_size_mb,
            )
        except OutOfMemoryError as exc:
            REQUESTS_TOTAL.labels("prefill", "error").inc()
            span.set_attribute("mdyn.error", str(exc))
            raise

        if evicted:
            EVICTIONS_TOTAL.labels("prefill").inc(len(evicted))

        await state.register_kv(
            result.kv_cache_id,
            req.session_id,
            result.num_prompt_tokens,
            result.kv_size_mb,
            NODE_NAME,
        )

    elapsed = time.perf_counter() - t0
    REQUEST_LATENCY.labels("prefill", "prefill").observe(elapsed)
    CACHE_UTILIZATION.labels("prefill").set(memory.utilization)
    REQUESTS_TOTAL.labels("prefill", "ok").inc()

    return PrefillResponse(
        session_id=req.session_id,
        kv_cache_id=result.kv_cache_id,
        num_prompt_tokens=result.num_prompt_tokens,
        kv_size_mb=result.kv_size_mb,
        prefill_ms=result.prefill_ms,
        backend=NODE_NAME,
        evicted=len(evicted),
    )


@app.post("/v1/generate")
async def generate_colocated(req: GenerateRequest):
    """Colocated baseline: prefill + decode in one process, no KV transfer."""
    max_tokens = req.max_tokens or settings.max_tokens_default

    async def stream():
        t0 = time.perf_counter()
        with tracer.start_as_current_span("colocated.generate") as span:
            span.set_attribute("mdyn.session_id", req.session_id)
            span.set_attribute("mdyn.mode", "colocated")
            result = await model.prefill(req.session_id, req.prompt)
            REQUEST_LATENCY.labels("prefill", "prefill").observe(result.prefill_ms / 1000.0)

            decode_req = DecodeRequest(
                session_id=req.session_id,
                kv_cache_id=result.kv_cache_id,
                num_prompt_tokens=result.num_prompt_tokens,
                kv_size_mb=result.kv_size_mb,
                max_tokens=max_tokens,
                prefill_backend=NODE_NAME,
            )
            out = await scheduler.submit(decode_req)
            first = True
            while True:
                chunk = await out.get()
                if first and chunk.token:
                    from common.metrics import TIME_TO_FIRST_TOKEN

                    TIME_TO_FIRST_TOKEN.labels("prefill").observe(time.perf_counter() - t0)
                    first = False
                yield sse_chunk(chunk)
                if chunk.finish_reason is not None:
                    break
            REQUEST_LATENCY.labels("prefill", "e2e").observe(time.perf_counter() - t0)
            REQUESTS_TOTAL.labels("prefill", "ok").inc()
        yield sse_done()

    return StreamingResponse(stream(), media_type="text/event-stream")
