"""Routing policy: KV-cache-aware sticky selection with least-loaded fallback.

The router keeps conversations *sticky*: the first turn of a session picks a
decode backend and records it in Redis; later turns of the same session are
sent back to that same backend because it (may) still hold the KV cache - a
cache hit avoids re-prefill and re-transfer. If the sticky backend has
disappeared from the registry (crashed / heartbeat expired) the router falls
back to the least-loaded live backend and re-pins the session.
"""

from __future__ import annotations

import itertools
from typing import List, Optional, Tuple

from common.config import Settings
from common.models import BackendInfo
from common.redis_client import RedisState


class NoBackendError(RuntimeError):
    pass


def _load_score(b: BackendInfo) -> float:
    """Lower is better: prefer idle, low-utilization backends."""
    return b.queue_depth + b.active_batch + b.utilization


class RoutingPolicy:
    def __init__(self, settings: Settings, state: RedisState) -> None:
        self.s = settings
        self.state = state
        self._rr_prefill = itertools.count()
        self._rr_decode = itertools.count()

    # -- candidate discovery ----------------------------------------------
    async def _candidates(self, role: str) -> List[BackendInfo]:
        backends = await self.state.list_backends(role)
        if backends:
            return backends
        # Static fallback (no registry entries yet): synthesize from config.
        urls = self.s.prefill_urls if role == "prefill" else self.s.decode_urls
        return [
            BackendInfo(name=f"{role}-{i}", url=url, role=role)
            for i, url in enumerate(urls)
        ]

    # -- prefill selection -------------------------------------------------
    async def select_prefill(self) -> BackendInfo:
        candidates = await self._candidates("prefill")
        if not candidates:
            raise NoBackendError("no prefill backends available")
        # Least-loaded, ties broken round-robin for even spread.
        candidates.sort(key=_load_score)
        best = candidates[0]
        tied = [c for c in candidates if _load_score(c) == _load_score(best)]
        return tied[next(self._rr_prefill) % len(tied)]

    # -- decode selection (sticky) ----------------------------------------
    async def select_decode(
        self, session_id: str
    ) -> Tuple[BackendInfo, str]:
        """Return (backend, decision) where decision is cache_hit|cache_miss."""
        candidates = await self._candidates("decode")
        if not candidates:
            raise NoBackendError("no decode backends available")
        by_name = {c.name: c for c in candidates}

        pinned = await self.state.get_session_backend(session_id)
        if pinned and pinned in by_name:
            return by_name[pinned], "cache_hit"

        # Cache miss (new session, or the pinned backend is gone): least-loaded.
        candidates.sort(key=_load_score)
        chosen = candidates[0]
        await self.state.set_session_backend(session_id, chosen.name)
        return chosen, "cache_miss"

    async def fallback_decode(
        self, session_id: str, exclude: str
    ) -> Optional[BackendInfo]:
        """Pick a live decode backend other than ``exclude`` and re-pin."""
        candidates = [
            c for c in await self._candidates("decode") if c.name != exclude
        ]
        if not candidates:
            return None
        candidates.sort(key=_load_score)
        chosen = candidates[0]
        await self.state.set_session_backend(session_id, chosen.name)
        return chosen
