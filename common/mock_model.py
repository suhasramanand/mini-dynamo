"""A mock/simulated model backend.

Fakes tokenization and token generation with realistic-looking latency so the
platform can be exercised end-to-end on a laptop with no GPU. Two operations:

* ``prefill`` — turn a prompt into a KV cache (latency grows with prompt len).
* ``decode_step`` — emit one token (fixed per-token latency + jitter).

Latencies include configurable jitter so p50/p95 histograms are meaningful.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
import uuid
from dataclasses import dataclass

from .config import Settings

# A small vocabulary so streamed output reads like plausible text.
_VOCAB = (
    "the model streams tokens through a disaggregated pipeline where prefill "
    "and decode scale independently while the router keeps each session sticky "
    "to the worker that already holds its cache so throughput stays high and "
    "latency stays low across many concurrent requests on commodity hardware"
).split()


def count_tokens(text: str) -> int:
    """Cheap deterministic 'tokenizer': whitespace words, min 1."""
    n = len(text.split())
    return max(1, n)


def new_kv_cache_id(session_id: str) -> str:
    return f"kv-{session_id[:8]}-{uuid.uuid4().hex[:8]}"


def _jittered(base_ms: float, jitter: float) -> float:
    if jitter <= 0:
        return base_ms
    return base_ms * (1.0 + random.uniform(-jitter, jitter))


@dataclass
class PrefillResult:
    kv_cache_id: str
    num_prompt_tokens: int
    kv_size_mb: float
    prefill_ms: float


class MockModel:
    """Stateless simulated backend driven by :class:`Settings`."""

    def __init__(self, settings: Settings) -> None:
        self.s = settings

    # -- prefill -----------------------------------------------------------
    def kv_size_mb(self, num_tokens: int) -> float:
        return num_tokens * self.s.kv_mb_per_token

    async def prefill(self, session_id: str, prompt: str) -> PrefillResult:
        num_tokens = count_tokens(prompt)
        latency_ms = _jittered(
            self.s.prefill_base_ms + self.s.prefill_ms_per_token * num_tokens,
            self.s.latency_jitter,
        )
        await asyncio.sleep(latency_ms / 1000.0)
        return PrefillResult(
            kv_cache_id=new_kv_cache_id(session_id),
            num_prompt_tokens=num_tokens,
            kv_size_mb=self.kv_size_mb(num_tokens),
            prefill_ms=latency_ms,
        )

    # -- decode ------------------------------------------------------------
    async def decode_step(self, session_id: str, index: int) -> str:
        """Generate one token (with per-token latency) and return its text."""
        latency_ms = _jittered(self.s.decode_ms_per_token, self.s.latency_jitter)
        await asyncio.sleep(latency_ms / 1000.0)
        return self.next_token(session_id, index)

    def next_token(self, session_id: str, index: int) -> str:
        """Deterministic pseudo-token so the same session reads consistently."""
        seed = int(hashlib.md5(f"{session_id}:{index}".encode()).hexdigest(), 16)
        word = _VOCAB[seed % len(_VOCAB)]
        # Append a space between words like a real streamed completion.
        return word + " "
