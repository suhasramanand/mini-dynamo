"""Model the cost of moving a KV cache between the prefill and decode stages.

In a real disaggregated deployment the KV cache produced by prefill must be
shipped to the decode worker (NVLink / RDMA / TCP). That transfer is pure
overhead that a colocated deployment never pays. We approximate it as:

    time_ms = fixed_ms + size_bytes / bandwidth_bytes_per_ms

so cost grows with the cache size (i.e. with prompt length) on top of a fixed
per-transfer setup cost.
"""

from __future__ import annotations

import asyncio


def transfer_ms(
    size_mb: float,
    bandwidth_gbps: float,
    fixed_ms: float,
    colocated: bool = False,
) -> float:
    """Return the simulated transfer time in milliseconds.

    ``bandwidth_gbps`` is gigabits per second. Colocated execution keeps the
    KV cache in local memory, so the network transfer cost is zero.
    """
    if colocated or size_mb <= 0:
        return 0.0
    # gigabits/s -> megabytes/ms:  Gb/s * (1000 Mb/Gb) / (8 b/B) / (1000 ms/s)
    #             = Gb/s * 0.125 MB/ms
    mb_per_ms = bandwidth_gbps * 0.125
    if mb_per_ms <= 0:
        return fixed_ms
    return fixed_ms + size_mb / mb_per_ms


async def simulate_transfer(
    size_mb: float,
    bandwidth_gbps: float,
    fixed_ms: float,
    colocated: bool = False,
) -> float:
    """Sleep for the computed transfer time and return the elapsed ms."""
    ms = transfer_ms(size_mb, bandwidth_gbps, fixed_ms, colocated)
    if ms > 0:
        await asyncio.sleep(ms / 1000.0)
    return ms
