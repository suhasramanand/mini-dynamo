"""Simulated GPU memory with a configurable KV-cache eviction policy.

This models the pressure a real inference server feels: a fixed pool of HBM,
KV-cache blocks that consume it, and an eviction policy that reclaims space
when a new allocation would overflow the pool. No real GPU is required.

Thread-safe: guarded by a single lock because allocation/eviction is a short
critical section and may be touched from FastAPI's threadpool as well as the
event loop.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class OutOfMemoryError(RuntimeError):
    """Raised when an allocation cannot be satisfied even after eviction."""


@dataclass
class KVBlock:
    kv_cache_id: str
    session_id: str
    num_tokens: int
    size_mb: float
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)


class MemorySimulator:
    """A fixed-size KV cache pool with LRU / FIFO / no eviction."""

    def __init__(
        self,
        total_mb: float,
        eviction_policy: str = "lru",
    ) -> None:
        self.total_mb = float(total_mb)
        self.eviction_policy = eviction_policy.lower()
        self._blocks: Dict[str, KVBlock] = {}
        self._used_mb = 0.0
        self._lock = threading.Lock()
        # Counters exposed to metrics.
        self.evictions = 0
        self.oom_events = 0
        self.allocations = 0

    # -- introspection -----------------------------------------------------
    @property
    def used_mb(self) -> float:
        return self._used_mb

    @property
    def free_mb(self) -> float:
        return self.total_mb - self._used_mb

    @property
    def utilization(self) -> float:
        if self.total_mb <= 0:
            return 0.0
        return min(1.0, self._used_mb / self.total_mb)

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    def contains(self, kv_cache_id: str) -> bool:
        with self._lock:
            return kv_cache_id in self._blocks

    # -- core operations ---------------------------------------------------
    def allocate(
        self,
        kv_cache_id: str,
        session_id: str,
        num_tokens: int,
        size_mb: float,
    ) -> List[str]:
        """Reserve ``size_mb`` for a KV cache block.

        Returns the list of ``kv_cache_id``s evicted to make room.
        Raises :class:`OutOfMemoryError` if the request cannot be satisfied.
        """
        if size_mb > self.total_mb:
            self.oom_events += 1
            raise OutOfMemoryError(
                f"block {size_mb:.1f}MB exceeds pool {self.total_mb:.1f}MB"
            )

        with self._lock:
            evicted: List[str] = []
            # Re-allocating an existing id: free the old footprint first.
            if kv_cache_id in self._blocks:
                self._used_mb -= self._blocks.pop(kv_cache_id).size_mb

            while self._used_mb + size_mb > self.total_mb:
                victim = self._pick_victim()
                if victim is None:
                    self.oom_events += 1
                    raise OutOfMemoryError(
                        f"cannot free {size_mb:.1f}MB (policy={self.eviction_policy})"
                    )
                self._used_mb -= self._blocks.pop(victim).size_mb
                evicted.append(victim)
                self.evictions += 1

            self._blocks[kv_cache_id] = KVBlock(
                kv_cache_id=kv_cache_id,
                session_id=session_id,
                num_tokens=num_tokens,
                size_mb=size_mb,
            )
            self._used_mb += size_mb
            self.allocations += 1
            return evicted

    def touch(self, kv_cache_id: str) -> bool:
        """Mark a block as recently used (for LRU). Returns True if present."""
        with self._lock:
            block = self._blocks.get(kv_cache_id)
            if block is None:
                return False
            block.last_access = time.time()
            return True

    def free(self, kv_cache_id: str) -> bool:
        with self._lock:
            block = self._blocks.pop(kv_cache_id, None)
            if block is None:
                return False
            self._used_mb -= block.size_mb
            return True

    def get(self, kv_cache_id: str) -> Optional[KVBlock]:
        with self._lock:
            return self._blocks.get(kv_cache_id)

    # -- eviction policy ---------------------------------------------------
    def _pick_victim(self) -> Optional[str]:
        """Choose a block to evict per the configured policy. Caller holds lock."""
        if not self._blocks or self.eviction_policy == "none":
            return None
        if self.eviction_policy == "fifo":
            # Oldest created wins.
            return min(self._blocks.items(), key=lambda kv: kv[1].created_at)[0]
        # Default: LRU — least recently accessed.
        return min(self._blocks.items(), key=lambda kv: kv[1].last_access)[0]

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_mb": self.total_mb,
                "used_mb": round(self._used_mb, 2),
                "free_mb": round(self.total_mb - self._used_mb, 2),
                "utilization": round(self.utilization, 4),
                "num_blocks": len(self._blocks),
                "evictions": self.evictions,
                "oom_events": self.oom_events,
                "allocations": self.allocations,
                "policy": self.eviction_policy,
            }
