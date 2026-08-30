"""Continuous batching scheduler for the decode stage.

Rather than waiting for a fixed window of requests, the scheduler runs a
perpetual loop. Each iteration is one simulated forward pass for the *whole*
active batch: it admits any newly-arrived sequences (up to ``max_batch_size``),
runs a single step, emits one token to every active sequence, and retires the
ones that have finished. New requests therefore join the running batch on the
very next step instead of blocking behind the current one.

Because one step advances every sequence together, aggregate throughput scales
with the batch size while per-token latency stays flat — exactly the property
continuous batching buys you on a real accelerator.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from .config import Settings
from .memory_sim import MemorySimulator, OutOfMemoryError
from .metrics import (
    ACTIVE_BATCH,
    CACHE_USED_MB,
    CACHE_UTILIZATION,
    EVICTIONS_TOTAL,
    QUEUE_DEPTH,
    TOKENS_TOTAL,
)
from .mock_model import MockModel
from .models import DecodeRequest, TokenChunk


@dataclass
class _Seq:
    req_id: str
    session_id: str
    kv_cache_id: str
    max_tokens: int
    kv_size_mb: float
    num_prompt_tokens: int
    out_queue: "asyncio.Queue[TokenChunk]"
    generated: int = 0
    started_at: float = field(default_factory=time.time)


class DecodeScheduler:
    def __init__(
        self,
        settings: Settings,
        model: MockModel,
        memory: MemorySimulator,
        service_label: str,
    ) -> None:
        self.s = settings
        self.model = model
        self.memory = memory
        self.label = service_label
        self._waiting: "asyncio.Queue[_Seq]" = asyncio.Queue()
        self._active: Dict[str, _Seq] = {}
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._task is None:
            self._running = True
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # -- introspection -----------------------------------------------------
    @property
    def queue_depth(self) -> int:
        return self._waiting.qsize()

    @property
    def active_batch(self) -> int:
        return len(self._active)

    # -- submission --------------------------------------------------------
    async def submit(self, req: DecodeRequest) -> "asyncio.Queue[TokenChunk]":
        """Enqueue a decode request; returns a queue that yields TokenChunks.

        The final chunk carries a non-null ``finish_reason``.
        """
        out: "asyncio.Queue[TokenChunk]" = asyncio.Queue()
        seq = _Seq(
            req_id=uuid.uuid4().hex,
            session_id=req.session_id,
            kv_cache_id=req.kv_cache_id,
            max_tokens=req.max_tokens or self.s.max_tokens_default,
            kv_size_mb=req.kv_size_mb,
            num_prompt_tokens=req.num_prompt_tokens,
            out_queue=out,
        )
        await self._waiting.put(seq)
        QUEUE_DEPTH.labels(self.label).set(self._waiting.qsize())
        return out

    # -- scheduler loop ----------------------------------------------------
    async def _run(self) -> None:
        idle_sleep = max(self.s.decode_ms_per_token, 5.0) / 1000.0
        while self._running:
            self._admit()
            if not self._active:
                await asyncio.sleep(idle_sleep)
                continue

            # One simulated forward pass for the entire active batch.
            step_ms = self.s.decode_ms_per_token
            if self.s.latency_jitter > 0:
                step_ms *= 1.0 + random.uniform(-self.s.latency_jitter, self.s.latency_jitter)
            await asyncio.sleep(step_ms / 1000.0)

            finished = []
            for req_id, seq in self._active.items():
                self.memory.touch(seq.kv_cache_id)
                token = self.model.next_token(seq.session_id, seq.num_prompt_tokens + seq.generated)
                seq.generated += 1
                TOKENS_TOTAL.labels(self.label).inc()
                await seq.out_queue.put(
                    TokenChunk(token=token, index=seq.generated - 1)
                )
                if seq.generated >= seq.max_tokens:
                    await seq.out_queue.put(
                        TokenChunk(index=seq.generated, finish_reason="length")
                    )
                    finished.append(req_id)

            for req_id in finished:
                seq = self._active.pop(req_id)
                self.memory.free(seq.kv_cache_id)

            self._publish_gauges()

    def _admit(self) -> None:
        """Move waiting sequences into the active batch (respecting memory)."""
        while self._waiting.qsize() > 0 and len(self._active) < self.s.max_batch_size:
            seq = self._waiting.get_nowait()
            try:
                evicted = self.memory.allocate(
                    seq.kv_cache_id, seq.session_id, seq.num_prompt_tokens, seq.kv_size_mb
                )
                if evicted:
                    EVICTIONS_TOTAL.labels(self.label).inc(len(evicted))
            except OutOfMemoryError:
                # Cannot fit even after eviction: reject this sequence.
                seq.out_queue.put_nowait(
                    TokenChunk(finish_reason="oom")
                )
                continue
            self._active[seq.req_id] = seq
        self._publish_gauges()

    def _publish_gauges(self) -> None:
        QUEUE_DEPTH.labels(self.label).set(self._waiting.qsize())
        ACTIVE_BATCH.labels(self.label).set(len(self._active))
        CACHE_UTILIZATION.labels(self.label).set(self.memory.utilization)
        CACHE_USED_MB.labels(self.label).set(self.memory.used_mb)
