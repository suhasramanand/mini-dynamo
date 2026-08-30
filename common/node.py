"""Helpers for prefill/decode worker nodes: self-registration heartbeat.

Each worker advertises itself in the Redis registry on a fixed interval with a
TTL slightly longer than the interval. If a worker dies, its entry expires and
the router stops routing to it - this is the failure detection that the router's
fallback path relies on.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from .config import Settings
from .models import BackendInfo
from .redis_client import RedisState

log = logging.getLogger("mini_dynamo.node")

# A StatsProvider returns the current live BackendInfo for this node.
StatsProvider = Callable[[], BackendInfo]


async def heartbeat_loop(
    state: RedisState, settings: Settings, stats: StatsProvider
) -> None:
    """Register this node repeatedly until cancelled."""
    while True:
        try:
            await state.register_backend(stats())
        except Exception as exc:  # pragma: no cover - transient redis blips
            log.warning("heartbeat failed: %s", exc)
        await asyncio.sleep(settings.heartbeat_interval)


def start_heartbeat(
    state: RedisState, settings: Settings, stats: StatsProvider
) -> asyncio.Task:
    return asyncio.create_task(heartbeat_loop(state, settings, stats))
