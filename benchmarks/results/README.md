# V1 Engine benchmark results

This directory stores raw JSON documents produced by the frozen V1 reference
benchmark. Do not hand-edit results or copy measurements from human-readable
logs.

The harness is pinned to `Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca`, Tesla T4, FP16, greedy sampling,
seed 0, the `reference` attention operator, and offline Hub loading. Populate the
standard Hugging Face cache before running it; benchmark execution always uses
`local_files_only=True`.

Run each workload from the repository root with the project virtual environment:

```bash
for workload in prefill-16 prefill-128 prefill-512 decode mixed cache-pressure
do
  .venv/bin/python benchmarks/benchmark_engine.py \
    --workload "$workload" \
    --output "benchmarks/results/reference-${workload}.json"
done
```

Each invocation performs two warmup repetitions and five measured repetitions.
The output is one JSON document with four reproducibility sections
(`model`, `hardware`, `runtime`, and `capacity`), the exact tokenized workload,
an aggregate `summary`, and raw per-repetition request timings.

All durations use seconds, token counts use tokens, and memory/capacity values
use bytes or unformatted page counts. `ttft` and `itl` contain a sample count,
median, and linearly interpolated p90. A latency value is JSON `null` when the
workload produces no corresponding samples (for example, ITL for a one-token
prefill). Throughput and elapsed time include only explicitly synchronized
`Engine.step()` calls; model loading and warmup are excluded. CUDA peak allocated
memory is the absolute allocator peak during a measured repetition. Peak KV
utilization is the maximum fraction of non-null physical pages allocated at an
execution boundary. The cache-pressure result also records the number of
preempted requests observed in existing scheduler plans.

Before accepting a reference baseline, run every workload at least twice and
inspect the raw repetitions for order-of-magnitude instability. Normal GPU noise
is reported rather than converted into a CI threshold.

## Recorded reference baseline

The baseline below was recorded on 2026-07-23 from commit `e212f86` on a Tesla
T4 with 15,360 MiB memory, driver 580.126.09, Python 3.12.3, PyTorch
2.8.0+cu128, and CUDA 12.8. Each displayed pair is `run-1 / run-2`; each run
contains five measured repetitions after two warmups.

| workload | prompt tokens/s | output tokens/s | total tokens/s | median TTFT seconds | median ITL seconds | CUDA peak bytes | peak KV utilization | preemptions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill-16 | 267.238 / 267.735 | 16.702 / 16.733 | 283.940 / 284.468 | 0.059972 / 0.059719 | null / null | 1,318,894,592 | 0.0159 | 0 / 0 |
| prefill-128 | 2,099.545 / 2,093.746 | 16.403 / 16.357 | 2,115.947 / 2,110.103 | 0.060842 / 0.060876 | null / null | 1,325,962,240 | 0.1270 | 0 / 0 |
| prefill-512 | 6,151.438 / 6,172.573 | 12.015 / 12.056 | 6,163.453 / 6,184.629 | 0.081647 / 0.082561 | null / null | 1,383,567,360 | 0.5079 | 0 / 0 |
| decode | 1.042 / 1.050 | 16.673 / 16.806 | 17.715 / 17.856 | 0.060102 / 0.059687 | 0.059968 / 0.059518 | 1,262,371,840 | 0.2903 | 0 / 0 |
| mixed | 35.333 / 35.371 | 16.887 / 16.905 | 52.220 / 52.276 | 0.110095 / 0.109447 | 0.059507 / 0.059457 | 1,267,226,624 | 0.2903 | 0 / 0 |
| cache-pressure | 10.045 / 10.023 | 20.089 / 20.045 | 30.134 / 30.068 | 0.080070 / 0.080083 | 0.078652 / 0.078467 | 1,222,400,000 | 1.0000 | 5 / 5 |

The largest run-to-run output-throughput change was 0.797% (decode). Per-run
elapsed-time coefficients of variation were at most 3.92%, so no
order-of-magnitude instability was observed. Cache pressure reached full
non-null page utilization and triggered one preemption per measured repetition.
