"""Render benchmark results as a Markdown report."""

from __future__ import annotations

from typing import Dict

from tabulate import tabulate


def _pct_delta(base: float, other: float) -> str:
    if base == 0:
        return "n/a"
    return f"{(other - base) / base * 100:+.1f}%"


def render_markdown(results: Dict) -> str:
    cfg = results["config"]
    colo = results["colocated"]
    disagg = results["disaggregated"]
    kv = results["kv_transfer"]

    lines = []
    lines.append("# Mini-Dynamo Benchmark: Colocated vs Disaggregated")
    lines.append("")
    lines.append(f"Generated: {results['generated_at']}")
    lines.append("")

    lines.append("## Workload")
    lines.append("")
    lines.append(
        tabulate(
            [
                ["Requests", cfg["requests"]],
                ["Concurrency", cfg["concurrency"]],
                ["Prompt tokens", cfg["prompt_tokens"]],
                ["Max output tokens", cfg["max_tokens"]],
                ["Distinct sessions", cfg["sessions"]],
            ],
            headers=["Parameter", "Value"],
            tablefmt="github",
        )
    )
    lines.append("")

    lines.append("## Results")
    lines.append("")
    rows = [
        ["p50 latency (ms)", colo["p50_latency_ms"], disagg["p50_latency_ms"],
         _pct_delta(colo["p50_latency_ms"], disagg["p50_latency_ms"])],
        ["p95 latency (ms)", colo["p95_latency_ms"], disagg["p95_latency_ms"],
         _pct_delta(colo["p95_latency_ms"], disagg["p95_latency_ms"])],
        ["mean latency (ms)", colo["mean_latency_ms"], disagg["mean_latency_ms"],
         _pct_delta(colo["mean_latency_ms"], disagg["mean_latency_ms"])],
        ["p50 TTFT (ms)", colo["p50_ttft_ms"], disagg["p50_ttft_ms"],
         _pct_delta(colo["p50_ttft_ms"], disagg["p50_ttft_ms"])],
        ["p95 TTFT (ms)", colo["p95_ttft_ms"], disagg["p95_ttft_ms"],
         _pct_delta(colo["p95_ttft_ms"], disagg["p95_ttft_ms"])],
        ["tokens/sec", colo["tokens_per_sec"], disagg["tokens_per_sec"],
         _pct_delta(colo["tokens_per_sec"], disagg["tokens_per_sec"])],
    ]
    lines.append(
        tabulate(
            rows,
            headers=["Metric", "Colocated", "Disaggregated", "Δ (disagg vs colo)"],
            tablefmt="github",
        )
    )
    lines.append("")

    lines.append("## KV transfer overhead")
    lines.append("")
    lines.append(
        tabulate(
            [
                ["Measured mean transfer / request (ms)", kv["measured_mean_ms"]],
                ["Transfers observed", kv["transfers"]],
                ["Share of disaggregated mean latency",
                 f"{kv['share_of_latency_pct']:.1f}%"],
            ],
            headers=["Metric", "Value"],
            tablefmt="github",
        )
    )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Disaggregation separates prefill and decode into independently "
        "scalable services at the cost of transferring the KV cache between "
        "them. The KV transfer overhead above is the price paid for that "
        "separation; in exchange, prefill and decode capacity can be scaled "
        "independently and decode nodes serve requests with continuous "
        "batching. The colocated configuration avoids the transfer but couples "
        "the two stages onto a single worker."
    )
    lines.append("")
    return "\n".join(lines)
