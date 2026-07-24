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

## Phase 1 SDPA feasibility spike

The non-registered spike in `benchmarks/spike_sdpa.py` reuses the reference
paged-cache write/gather helpers and measures one Qwen3-0.6B attention layer.
It uses 16 query heads, 8 KV heads, head dimension 128, FP16, a 512-token
context, manual GQA expansion, and an explicit boolean visibility mask. Each
pair below is `run-1 / run-2`, with 10 warmups and 100 synchronized
measurements per run.

| case | reference median ms | SDPA median ms | speedup | max absolute error | selected kernel |
| --- | ---: | ---: | ---: | ---: | --- |
| prefill, Q=512 | 2.012 / 1.975 | 0.878 / 0.862 | 2.292x / 2.292x | 0.001953 | efficient attention |
| decode, Q=1 | 1.037 / 1.032 | 0.824 / 0.821 | 1.259x / 1.257x | 0.000244 | efficient attention |

Both cases satisfy `atol=0.05`, `rtol=0.05`. PyTorch selected
`aten::_scaled_dot_product_efficient_attention`; forcing efficient attention or
math succeeds, while Flash Attention and cuDNN attention are unavailable on the
T4 for these masked shapes. The isolated SDPA core produces one Dynamo graph
with no break. The complete paged path produces four graphs and three breaks
because the existing gather helper calls `Tensor.item()`.

The repeated benefit is sufficient to proceed to a formal, explicitly selected
operator. This is an operator-level single-layer measurement, not evidence of
the final end-to-end Engine speedup; parity and Engine benchmarks remain
required before calling the backend optimized.

The active Phase 1 experiment machine changed on 2026-07-24 to an NVIDIA
GeForce RTX 5090 with 32,607 MiB memory and compute capability 12.0. PyTorch
remained 2.8.0+cu128 with CUDA 12.8, so the same shapes, seed, dtype, tolerance,
warmup count, and measurement count were rerun without changing the operator
code. The T4 measurements above remain the frozen Phase 0/initial-spike record;
the following numbers are a separate hardware baseline.

| case | reference median ms | SDPA median ms | speedup | max absolute error | selected kernel |
| --- | ---: | ---: | ---: | ---: | --- |
| prefill, Q=512 | 0.324 / 0.326 | 0.242 / 0.242 | 1.340x / 1.346x | 0.001953 | efficient attention |
| decode, Q=1 | 0.335 / 0.330 | 0.286 / 0.287 | 1.169x / 1.148x | 0.000122 | efficient attention |

On the 5090, forcing Flash Attention still fails because this spike passes a
non-null visibility mask. Forced cuDNN attention succeeds for prefill but does
not support the length-one decode query, while efficient attention and math
succeed for both cases. The isolated core and complete paged path retain the
same `1 graph / 0 breaks` and `4 graphs / 3 breaks` structure observed on the
T4. The smaller relative speedup on the faster GPU is a reason to measure the
write, gather, input preparation, and attention core separately before choosing
the first custom Triton target.

### RTX 5090 stage breakdown

Run the breakdown as a module so it can reuse the exact feasibility-spike
shapes and SDPA helpers:

```bash
.venv/bin/python -m benchmarks.spike_attention_breakdown \
  --expected-gpu-name "NVIDIA GeForce RTX 5090" \
  --output benchmarks/results/spikes/attention-breakdown-rtx5090-run-1.json
```

The table reports isolated median latency in milliseconds as
`run-1 / run-2`. Every isolated sample has its own CUDA synchronization, so
these medians rank candidate optimization targets but must not be added together
or treated as a percentage decomposition of the full operator.

| case | full paged SDPA | KV write | paged gather | request scalar/view extraction | dense SDPA preparation | SDPA core | output copy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill, Q=512 | 0.245 / 0.244 | 0.018 / 0.018 | 0.083 / 0.082 | 0.030 / 0.030 | 0.051 / 0.051 | 0.064 / 0.064 | 0.011 / 0.011 |
| decode, Q=1 | 0.239 / 0.238 | 0.017 / 0.017 | 0.083 / 0.081 | 0.030 / 0.030 | 0.051 / 0.051 | 0.059 / 0.058 | 0.010 / 0.011 |

Paged gather is the largest isolated stage, and dense input preparation plus
GPU-to-CPU scalar extraction are also material relative to the SDPA core. A
custom dense attention core would therefore optimize the wrong boundary. The
Triton learning path starts with batched page write and page-address mapping,
then moves the page loads into one ragged paged-attention kernel. The final path
must load K/V through the block table for every request concurrently and avoid
the CPU request loop, gathered K/V tensors, GQA expansion, and explicit dense
mask.

The breakdown harness reports a lower decode full-path median than the earlier
feasibility harness (`0.238-0.239 ms` versus `0.286-0.287 ms`). Both pairs are
internally repeatable, but the surrounding profiler/backend probes and GPU clock
state differ. These sub-millisecond absolute timings are therefore
harness-local; later kernel comparisons must interleave complete operator paths
under one shared warmup and measurement loop.

### Triton paged path step 1: batched page write

The first Triton step replaces the two reference `index_copy_` calls with one
two-dimensional launch. Grid axis 0 covers every flattened token in the batch;
axis 1 covers every KV head. Each program loads `slot_mapping[token]`, derives
the physical block and offset, and copies one K head plus one V head across the
head dimension. No request loop or request-specific launch is used.

Run the step as:

```bash
.venv/bin/python -m benchmarks.spike_triton_paged \
  --expected-gpu-name "NVIDIA GeForce RTX 5090" \
  --output benchmarks/results/spikes/triton-page-write-rtx5090-run-1.json
```

The two raw runs use 100 alternating warmup calls, 20 measured samples, and
1,000 launches per CUDA-event sample. Latency is the median per launch.

| case | reference write microseconds | Triton write microseconds | Triton speedup | exact cache parity |
| --- | ---: | ---: | ---: | --- |
| prefill, 512 tokens | 12.496 / 12.501 | 19.881 / 20.397 | 0.629x / 0.613x | yes / yes |
| decode, 1 token | 15.362 / 15.200 | 21.091 / 20.788 | 0.728x / 0.731x | yes / yes |

The standalone Triton wrapper is slower end to end even though it launches one
kernel instead of two. The same JSON documents contain a 100-operation
`torch.profiler` sample. For prefill, the two reference CUDA kernels total
`4.262 / 4.278 us` per write and the Triton kernel uses
`2.680 / 2.676 us`, a device-time reduction of about `1.59x`. For decode, the
corresponding values are `3.165 / 3.161 us` and `0.837 / 0.839 us`, a reduction
of about `3.77x`. Python-to-Triton dispatch nevertheless dominates the observed
`20-21 us` operator call, so a smaller device kernel does not make this isolated
wrapper faster.

Step 1 is accepted for address correctness, not as a standalone optimization.
The result argues against adding a separately launched Triton write to the
runtime unchanged. The fused design must amortize dispatch over paged attention,
or read current scheduled K/V directly while preserving the required cache
write side effect.
