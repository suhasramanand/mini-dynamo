"""Redis-backed shared state for Mini-Dynamo.

Three responsibilities:

1. **Service registry** — prefill/decode nodes self-register with a TTL
   heartbeat so the router can discover them dynamically and detect failures.
2. **Sticky session map** — ``session_id -> decode backend`` so follow-up turns
   in a conversation land on the worker that already holds the KV cache.
3. **KV metadata registry** — ``kv_cache_id -> {size, tokens, backend}`` so the
   decode stage knows how much to "transfer" and where the cache lives.

All access is async (``redis.asyncio``). Every method degrades gracefully if
Redis is briefly unavailable rather than crashing the request path.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

import redis.asyncio as aioredis

from .config import Settings
from .models import BackendInfo

_REGISTRY_PREFIX = "mdyn:backend:"      # + role:name  -> JSON BackendInfo
_SESSION_PREFIX = "mdyn:session:"       # + session_id -> decode backend name
_KV_PREFIX = "mdyn:kv:"                 # + kv_cache_id -> JSON metadata


class RedisState:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._r: aioredis.Redis = aioredis.from_url(
            settings.redis_url, decode_responses=True
        )

    async def close(self) -> None:
        await self._r.aclose()

    async def ping(self) -> bool:
        try:
            return bool(await self._r.ping())
        except Exception:
            return False

    # -- service registry --------------------------------------------------
    def _registry_key(self, role: str, name: str) -> str:
        return f"{_REGISTRY_PREFIX}{role}:{name}"

    async def register_backend(self, info: BackendInfo) -> None:
        """Publish/refresh this node in the registry with a TTL heartbeat."""
        key = self._registry_key(info.role, info.name)
        await self._r.set(key, info.model_dump_json(), ex=self.s.heartbeat_ttl)

    async def list_backends(self, role: str) -> List[BackendInfo]:
        """Return all live backends for a role (expired heartbeats drop out)."""
        pattern = f"{_REGISTRY_PREFIX}{role}:*"
        out: List[BackendInfo] = []
        async for key in self._r.scan_iter(match=pattern):
            raw = await self._r.get(key)
            if raw:
                try:
                    out.append(BackendInfo.model_validate_json(raw))
                except Exception:
                    continue
        return out

    # -- sticky sessions ---------------------------------------------------
    async def get_session_backend(self, session_id: str) -> Optional[str]:
        return await self._r.get(f"{_SESSION_PREFIX}{session_id}")

    async def set_session_backend(self, session_id: str, backend: str) -> None:
        await self._r.set(
            f"{_SESSION_PREFIX}{session_id}", backend, ex=self.s.session_ttl
        )

    # -- KV metadata -------------------------------------------------------
    async def register_kv(
        self,
        kv_cache_id: str,
        session_id: str,
        num_tokens: int,
        size_mb: float,
        backend: str,
    ) -> None:
        meta = {
            "kv_cache_id": kv_cache_id,
            "session_id": session_id,
            "num_tokens": num_tokens,
            "size_mb": size_mb,
            "backend": backend,
            "created_at": time.time(),
        }
        await self._r.set(
            f"{_KV_PREFIX}{kv_cache_id}", json.dumps(meta), ex=self.s.session_ttl
        )

    async def get_kv(self, kv_cache_id: str) -> Optional[Dict]:
        raw = await self._r.get(f"{_KV_PREFIX}{kv_cache_id}")
        return json.loads(raw) if raw else None
