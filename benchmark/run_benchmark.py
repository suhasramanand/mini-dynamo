"""Benchmark harness: colocated vs disaggregated execution.

Drives the same workload against two endpoints:

* disaggregated  — the router in disaggregated mode (prefill and decode are
  separate services; the KV cache is transferred between them),
* colocated      — the router in colocated mode (a single worker runs prefill
  and decode in-process; no KV transfer).

It also measures the KV-transfer overhead directly by sampling the decode
nodes' Prometheus ``mdyn_kv_transfer_seconds`` histogram across the run.

Run it inside the compose network so it can reach the internal service names:

    docker compose run --rm --no-deps -v "$PWD:/work" -w /work prefill \
        python benchmark/run_benchmark.py

Results are written as JSON (in benchmark/results/) and Markdown (stdout).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
from typing import List, Tuple

import httpx

# Make the shared load client importable whether run from repo root or /work.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from load_client import run_load  # noqa: E402
from report import render_markdown  # noqa: E402  (same dir)


def _parse_metric_total(text: str, metric: str) -> float:
    """Sum the values of all series for ``metric`` in a Prometheus payload."""
    total = 0.0
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith(metric):
            continue
        # Line looks like:  metric{labels} value   (or  metric value)
        head, _, value = line.rpartition(" ")
        name = head.split("{", 1)[0].strip()
        if name != metric:
            continue
        try:
            total += float(value)
        except ValueError:
            continue
    return total


async def _sample_transfer(
    client: httpx.AsyncClient, metrics_urls: List[str]
) -> Tuple[float, float]:
    """Return (sum_seconds, count) of KV transfers across decode nodes."""
    total_sum = 0.0
    total_count = 0.0
    for url in metrics_urls:
        try:
            text = (await client.get(url, timeout=5.0)).text
        except Exception:
            continue
        total_sum += _parse_metric_total(text, "mdyn_kv_transfer_seconds_sum")
        total_count += _parse_metric_total(text, "mdyn_kv_transfer_seconds_count")
    return total_sum, total_count


async def run_benchmark(args) -> dict:
    async with httpx.AsyncClient() as client:
        # Warmup so both paths have live batches and JIT-free steady state.
        await run_load(args.disagg_url, args.warmup, args.concurrency,
                       args.max_tokens, args.prompt_tokens, args.sessions, "warm-d-")
        await run_load(args.colo_url, args.warmup, args.concurrency,
                       args.max_tokens, args.prompt_tokens, args.sessions, "warm-c-")

        # Disaggregated run, bracketed by KV-transfer metric samples.
        s0, c0 = await _sample_transfer(client, args.decode_metrics)
        disagg = await run_load(args.disagg_url, args.requests, args.concurrency,
                                args.max_tokens, args.prompt_tokens, args.sessions,
                                "disagg-")
        s1, c1 = await _sample_transfer(client, args.decode_metrics)

        # Colocated run.
        colo = await run_load(args.colo_url, args.requests, args.concurrency,
                              args.max_tokens, args.prompt_tokens, args.sessions,
                              "colo-")

    transfers = c1 - c0
    measured_mean_ms = round((s1 - s0) / transfers * 1000, 2) if transfers > 0 else 0.0
    share = (
        measured_mean_ms / disagg["mean_latency_ms"] * 100
        if disagg["mean_latency_ms"] else 0.0
    )

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "requests": args.requests,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "prompt_tokens": args.prompt_tokens,
            "sessions": args.sessions,
            "disagg_url": args.disagg_url,
            "colo_url": args.colo_url,
        },
        "colocated": colo,
        "disaggregated": disagg,
        "kv_transfer": {
            "measured_mean_ms": measured_mean_ms,
            "transfers": int(transfers),
            "share_of_latency_pct": round(share, 1),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--disagg-url", default="http://router:8000")
    p.add_argument("--colo-url", default="http://router-colo:8000")
    p.add_argument("--decode-metrics", nargs="*",
                   default=["http://decode1:8000/metrics", "http://decode2:8000/metrics"])
    p.add_argument("--requests", type=int, default=60)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--prompt-tokens", type=int, default=64)
    p.add_argument("--sessions", type=int, default=12)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--json-out", default="")
    args = p.parse_args()

    results = asyncio.run(run_benchmark(args))

    json_out = args.json_out or os.path.join(
        _ROOT, "benchmark", "results",
        f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(json_out), exist_ok=True)
    with open(json_out, "w") as fh:
        json.dump(results, fh, indent=2)

    print(render_markdown(results))
    print(f"\n<!-- JSON written to {json_out} -->", file=sys.stderr)


if __name__ == "__main__":
    main()
