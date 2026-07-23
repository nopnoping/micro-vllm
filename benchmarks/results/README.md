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
