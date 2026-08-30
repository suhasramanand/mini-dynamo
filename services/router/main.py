"""Router service — the client entry point.

Orchestrates a generation request across the disaggregated pipeline:

    client -> router -> prefill (compute KV) -> decode (transfer + stream)

Routing is KV-cache-aware and sticky per session, with a fallback to another
decode backend if the preferred one is unreachable. In colocated mode the
router instead forwards the whole request to a single worker that runs
prefill+decode in-process (the benchmark baseline).
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from starlette.responses import JSONResponse, StreamingResponse

from common.config import get_settings
from common.metrics import (
    REQUEST_LATENCY,
    REQUESTS_TOTAL,
    ROUTING_DECISIONS,
    TIME_TO_FIRST_TOKEN,
    mount_metrics,
)
from common.models import GenerateRequest, PrefillResponse
from common.redis_client import RedisState
from common.sse import sse, sse_done
from common.telemetry import get_tracer, setup_telemetry

from routing import NoBackendError, RoutingPolicy

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mini_dynamo.router")

settings = get_settings()
settings.role = "router"
if not settings.service_name or settings.service_name == "mini-dynamo":
    settings.service_name = "router"

state = RedisState(settings)
policy = RoutingPolicy(settings, state)
tracer = get_tracer()
client: httpx.AsyncClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
    log.info("router up (mode=%s)", settings.mode)
    yield
    await client.aclose()
    await state.close()


app = FastAPI(title="Mini-Dynamo Router", lifespan=lifespan)
setup_telemetry(app, settings)
mount_metrics(app)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "role": "router", "mode": settings.mode,
            "redis": await state.ping()}


@app.get("/backends")
async def backends() -> dict:
    """Introspect the live registry (useful for demos and debugging)."""
    return {
        "prefill": [b.model_dump() for b in await state.list_backends("prefill")],
        "decode": [b.model_dump() for b in await state.list_backends("decode")],
    }


@app.post("/v1/generate")
async def generate(req: GenerateRequest):
    if settings.is_colocated:
        return await _generate_colocated(req)
    return await _generate_disaggregated(req)


# --------------------------------------------------------------------------
# Disaggregated path: prefill -> decode
# --------------------------------------------------------------------------
async def _generate_disaggregated(req: GenerateRequest):
    max_tokens = req.max_tokens or settings.max_tokens_default

    try:
        prefill_backend = await policy.select_prefill()
    except NoBackendError as exc:
        REQUESTS_TOTAL.labels("router", "error").inc()
        return JSONResponse(status_code=503, content={"error": str(exc)})

    async def stream():
        t0 = time.perf_counter()
        with tracer.start_as_current_span("router.generate") as span:
            span.set_attribute("mdyn.session_id", req.session_id)
            span.set_attribute("mdyn.mode", "disaggregated")

            # --- 1. Prefill --------------------------------------------------
            try:
                pf_resp = await client.post(
                    f"{prefill_backend.url}/v1/prefill",
                    json={
                        "session_id": req.session_id,
                        "prompt": req.prompt,
                        "max_tokens": max_tokens,
                    },
                )
                pf_resp.raise_for_status()
                pf = PrefillResponse(**pf_resp.json())
            except Exception as exc:  # noqa: BLE001
                log.error("prefill failed: %s", exc)
                REQUESTS_TOTAL.labels("router", "error").inc()
                yield sse({"error": f"prefill failed: {exc}", "finish_reason": "error"})
                yield sse_done()
                return
            span.set_attribute("mdyn.prefill_backend", pf.backend)
            span.set_attribute("mdyn.prompt_tokens", pf.num_prompt_tokens)

            # --- 2. Decode backend selection (sticky, KV-aware) -------------
            try:
                decode_backend, decision = await policy.select_decode(req.session_id)
            except NoBackendError as exc:
                yield sse({"error": str(exc), "finish_reason": "error"})
                yield sse_done()
                return
            ROUTING_DECISIONS.labels(decision).inc()
            span.set_attribute("mdyn.decode_backend", decode_backend.name)
            span.set_attribute("mdyn.routing", decision)

            decode_payload = {
                "session_id": req.session_id,
                "kv_cache_id": pf.kv_cache_id,
                "num_prompt_tokens": pf.num_prompt_tokens,
                "kv_size_mb": pf.kv_size_mb,
                "max_tokens": max_tokens,
                "prefill_backend": pf.backend,
            }

            # --- 3. Stream decode, with fallback on connection failure ------
            attempted = set()
            first = True
            while True:
                attempted.add(decode_backend.name)
                try:
                    async for frame in _stream_decode(decode_backend.url, decode_payload):
                        if first and frame.get("token"):
                            TIME_TO_FIRST_TOKEN.labels("router").observe(
                                time.perf_counter() - t0
                            )
                            first = False
                        yield sse(frame)
                        if frame.get("finish_reason") is not None:
                            REQUEST_LATENCY.labels("router", "e2e").observe(
                                time.perf_counter() - t0
                            )
                            REQUESTS_TOTAL.labels("router", "ok").inc()
                            yield sse_done()
                            return
                    # Stream ended without a finish frame -> treat as failure.
                    raise RuntimeError("decode stream ended unexpectedly")
                except Exception as exc:  # noqa: BLE001 - trigger fallback
                    log.warning("decode via %s failed (%s); trying fallback",
                                decode_backend.name, exc)
                    fb = await policy.fallback_decode(req.session_id, decode_backend.name)
                    if fb is None or fb.name in attempted:
                        REQUESTS_TOTAL.labels("router", "error").inc()
                        yield sse({"error": f"decode failed: {exc}",
                                   "finish_reason": "error"})
                        yield sse_done()
                        return
                    ROUTING_DECISIONS.labels("fallback").inc()
                    REQUESTS_TOTAL.labels("router", "fallback").inc()
                    span.set_attribute("mdyn.fallback_to", fb.name)
                    decode_backend = fb
                    first = True

    return StreamingResponse(stream(), media_type="text/event-stream")


async def _stream_decode(url: str, payload: dict):
    """Yield parsed JSON frames from a decode backend's SSE stream."""
    async with client.stream("POST", f"{url}/v1/decode", json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


# --------------------------------------------------------------------------
# Colocated path: single worker does prefill+decode (benchmark baseline)
# --------------------------------------------------------------------------
async def _generate_colocated(req: GenerateRequest):
    max_tokens = req.max_tokens or settings.max_tokens_default
    try:
        worker = await policy.select_prefill()  # prefill nodes host the colocated path
    except NoBackendError as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})

    async def stream():
        t0 = time.perf_counter()
        first = True
        with tracer.start_as_current_span("router.generate") as span:
            span.set_attribute("mdyn.mode", "colocated")
            span.set_attribute("mdyn.worker", worker.name)
            try:
                async with client.stream(
                    "POST",
                    f"{worker.url}/v1/generate",
                    json={"session_id": req.session_id, "prompt": req.prompt,
                          "max_tokens": max_tokens, "stream": True},
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            frame = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if first and frame.get("token"):
                            TIME_TO_FIRST_TOKEN.labels("router").observe(
                                time.perf_counter() - t0
                            )
                            first = False
                        yield sse(frame)
            except Exception as exc:  # noqa: BLE001
                yield sse({"error": str(exc), "finish_reason": "error"})
            REQUEST_LATENCY.labels("router", "e2e").observe(time.perf_counter() - t0)
            REQUESTS_TOTAL.labels("router", "ok").inc()
        yield sse_done()

    return StreamingResponse(stream(), media_type="text/event-stream")
