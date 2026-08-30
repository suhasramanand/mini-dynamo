# Mini-Dynamo Benchmark: Colocated vs Disaggregated

Representative run on a laptop (Docker Compose, simulated backend, no GPU).
Reproduce with `make bench`.

Generated: 2026-08-30T01:07:42+00:00

## Workload

| Parameter         |   Value |
|-------------------|---------|
| Requests          |      60 |
| Concurrency       |       8 |
| Prompt tokens     |      64 |
| Max output tokens |      32 |
| Distinct sessions |      12 |

## Results

| Metric            |   Colocated |   Disaggregated | Δ (disagg vs colo)   |
|-------------------|-------------|-----------------|----------------------|
| p50 latency (ms)  |       556   |           588.2 | +5.8%                |
| p95 latency (ms)  |       605.1 |           633.2 | +4.6%                |
| mean latency (ms) |       558.6 |           589   | +5.4%                |
| p50 TTFT (ms)     |        76.7 |            97.2 | +26.7%               |
| p95 TTFT (ms)     |        93.9 |           118.3 | +26.0%               |
| tokens/sec        |      3167.8 |          2994.6 | -5.5%                |

## KV transfer overhead

| Metric                                | Value   |
|---------------------------------------|---------|
| Measured mean transfer / request (ms) | 12.24   |
| Transfers observed                    | 60      |
| Share of disaggregated mean latency   | 2.1%    |

The measured transfer (12.24 ms) matches the cost model for this workload:
64 prompt tokens × 0.5 MB/token = 32 MB, transferred at 25 Gbps (3.125 MB/ms)
plus a 2 ms fixed cost = 12.24 ms.

## Interpretation

Disaggregation separates prefill and decode into independently scalable services at the cost of transferring the KV cache between them. The KV transfer overhead above is the price paid for that separation; in exchange, prefill and decode capacity can be scaled independently and decode nodes serve requests with continuous batching. The colocated configuration avoids the transfer but couples the two stages onto a single worker.
