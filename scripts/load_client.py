"""A small async load client for the router.

Sends N generation requests at a target concurrency, consumes the SSE stream,
and reports latency (p50/p95), time-to-first-token, and tokens/sec. Also used
by the benchmark harness.

    python scripts/load_client.py --url http://localhost:8000 \
        --requests 50 --concurrency 8 --max-tokens 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import List

import httpx


@dataclass
class Result:
    ttft: float = 0.0
    total: float = 0.0
    tokens: int = 0
    ok: bool = True


@dataclass
class Summary:
    results: List[Result] = field(default_factory=list)

    def add(self, r: Result) -> None:
        self.results.append(r)

    def report(self) -> dict:
        ok = [r for r in self.results if r.ok]
        if not ok:
            return {"requests": len(self.results), "ok": 0}
        lat = sorted(r.total for r in ok)
        ttft = sorted(r.ttft for r in ok)
        total_tokens = sum(r.tokens for r in ok)
        wall = max(r.total for r in ok)
        return {
            "requests": len(self.results),
            "ok": len(ok),
            "p50_latency_ms": round(_pct(lat, 50) * 1000, 1),
            "p95_latency_ms": round(_pct(lat, 95) * 1000, 1),
            "mean_latency_ms": round(statistics.mean(lat) * 1000, 1),
            "p50_ttft_ms": round(_pct(ttft, 50) * 1000, 1),
            "p95_ttft_ms": round(_pct(ttft, 95) * 1000, 1),
            "total_tokens": total_tokens,
            "tokens_per_sec": round(total_tokens / wall, 1) if wall > 0 else 0.0,
        }


def _pct(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


async def one_request(
    client: httpx.AsyncClient, url: str, session_id: str, prompt: str, max_tokens: int
) -> Result:
    r = Result()
    t0 = time.perf_counter()
    first = True
    try:
        async with client.stream(
            "POST",
            f"{url}/v1/generate",
            json={"session_id": session_id, "prompt": prompt,
                  "max_tokens": max_tokens, "stream": True},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                frame = json.loads(data)
                if frame.get("error"):
                    r.ok = False
                    break
                if frame.get("token"):
                    if first:
                        r.ttft = time.perf_counter() - t0
                        first = False
                    r.tokens += 1
    except Exception:
        r.ok = False
    r.total = time.perf_counter() - t0
    return r


async def run(args) -> dict:
    summary = Summary()
    sem = asyncio.Semaphore(args.concurrency)
    prompt = " ".join(["token"] * args.prompt_tokens)

    async with httpx.AsyncClient(timeout=120.0) as client:
        async def worker(i: int):
            async with sem:
                sid = f"{args.session_prefix}{i % args.sessions}"
                summary.add(await one_request(client, args.url, sid, prompt, args.max_tokens))

        await asyncio.gather(*(worker(i) for i in range(args.requests)))
    return summary.report()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--requests", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--prompt-tokens", type=int, default=32)
    p.add_argument("--sessions", type=int, default=10,
                   help="number of distinct session ids (drives cache hits)")
    p.add_argument("--session-prefix", default="load-")
    args = p.parse_args()

    report = asyncio.run(run(args))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
