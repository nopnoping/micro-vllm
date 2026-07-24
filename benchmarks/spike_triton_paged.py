"""Build the Triton paged-attention path one address operation at a time.

Step 1 implements only batched K/V cache writes. One launch covers every
flattened input token and KV head, while the existing slot mapping remains the
source of truth for physical cache placement. The spike is not registered as a
runtime attention operator.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
from pathlib import Path
import platform
import sys
from typing import Any

import torch
import triton
import triton.language as tl

from benchmarks.spike_sdpa import (
    DTYPE,
    RANDOM_SEED,
    SPIKE_SHAPES,
    SpikeCase,
    _latency_summary,
    _make_case,
    _shape_document,
    _validate_target_gpu,
)
from micro_vllm.attention.attention import AttentionMetadata
from micro_vllm.attention.operators.reference import write_kv_cache


SCHEMA_VERSION = "micro-vllm.triton-paged-step1.v1"


@triton.jit
def _write_kv_cache_kernel(
    key_ptr,
    value_ptr,
    cache_ptr,
    slot_mapping_ptr,
    key_token_stride,
    key_head_stride,
    key_dim_stride,
    value_token_stride,
    value_head_stride,
    value_dim_stride,
    cache_plane_stride,
    cache_block_stride,
    cache_offset_stride,
    cache_head_stride,
    cache_dim_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    token_index = tl.program_id(0)
    kv_head_index = tl.program_id(1)
    dim_offsets = tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offsets < HEAD_DIM

    physical_slot = tl.load(slot_mapping_ptr + token_index)
    physical_block = physical_slot // BLOCK_SIZE
    block_offset = physical_slot % BLOCK_SIZE

    key_offsets = (
        token_index * key_token_stride
        + kv_head_index * key_head_stride
        + dim_offsets * key_dim_stride
    )
    value_offsets = (
        token_index * value_token_stride
        + kv_head_index * value_head_stride
        + dim_offsets * value_dim_stride
    )
    cache_offsets = (
        physical_block * cache_block_stride
        + block_offset * cache_offset_stride
        + kv_head_index * cache_head_stride
        + dim_offsets * cache_dim_stride
    )

    key = tl.load(key_ptr + key_offsets, mask=dim_mask)
    value = tl.load(value_ptr + value_offsets, mask=dim_mask)
    tl.store(cache_ptr + cache_offsets, key, mask=dim_mask)
    tl.store(
        cache_ptr + cache_plane_stride + cache_offsets,
        value,
        mask=dim_mask,
    )


def triton_write_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_kv_cache: torch.Tensor,
    metadata: AttentionMetadata,
) -> None:
    """Write all flattened batch tokens with one two-dimensional launch."""

    head_dim = key.shape[-1]
    block_dim = triton.next_power_of_2(head_dim)
    num_warps = 4 if block_dim >= 64 else 1
    grid = (key.shape[0], key.shape[1])
    _write_kv_cache_kernel[grid](
        key,
        value,
        layer_kv_cache,
        metadata.slot_mapping,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        layer_kv_cache.stride(0),
        layer_kv_cache.stride(1),
        layer_kv_cache.stride(2),
        layer_kv_cache.stride(3),
        layer_kv_cache.stride(4),
        HEAD_DIM=head_dim,
        BLOCK_SIZE=metadata.block_size,
        BLOCK_DIM=block_dim,
        num_warps=num_warps,
    )


def _accuracy(case: SpikeCase, device: torch.device) -> dict[str, Any]:
    reference_cache = case.layer_kv_cache.clone()
    triton_cache = case.layer_kv_cache.clone()
    write_kv_cache(
        case.key,
        case.value,
        reference_cache,
        case.metadata,
    )
    triton_write_kv_cache(
        case.key,
        case.value,
        triton_cache,
        case.metadata,
    )
    torch.cuda.synchronize(device)
    difference = (
        reference_cache.to(torch.float32) - triton_cache.to(torch.float32)
    ).abs()
    return {
        "exact": torch.equal(reference_cache, triton_cache),
        "max_absolute_error": difference.max().item(),
    }


def _measure_launch_batch(
    operation: Callable[[], object],
    *,
    launches_per_sample: int,
) -> float:
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    for _ in range(launches_per_sample):
        operation()
    finished.record()
    finished.synchronize()
    return started.elapsed_time(finished) / 1000.0 / launches_per_sample


def _measure_pair(
    reference: Callable[[], object],
    triton_operation: Callable[[], object],
    *,
    warmup_iterations: int,
    measurement_iterations: int,
    launches_per_sample: int,
) -> dict[str, dict[str, int | float] | float]:
    for _ in range(warmup_iterations):
        reference()
        triton_operation()
    torch.cuda.synchronize()

    samples: dict[str, list[float]] = {
        "reference": [],
        "triton": [],
    }
    operations = {
        "reference": reference,
        "triton": triton_operation,
    }
    for sample_index in range(measurement_iterations):
        order = (
            ("reference", "triton")
            if sample_index % 2 == 0
            else ("triton", "reference")
        )
        for name in order:
            samples[name].append(
                _measure_launch_batch(
                    operations[name],
                    launches_per_sample=launches_per_sample,
                )
            )

    reference_summary = _latency_summary(samples["reference"])
    triton_summary = _latency_summary(samples["triton"])
    return {
        "reference": reference_summary,
        "triton": triton_summary,
        "median_speedup": (
            reference_summary["median_seconds"]
            / triton_summary["median_seconds"]
        ),
    }


def _profile_cuda_kernels(
    operation: Callable[[], object],
    device: torch.device,
    *,
    iterations: int,
) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    with profile(
        activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
    ) as profiler:
        for _ in range(iterations):
            operation()
        torch.cuda.synchronize(device)

    kernels = []
    total_device_time_microseconds = 0.0
    for event in profiler.key_averages():
        device_time = getattr(event, "self_device_time_total", None)
        if device_time is None:
            device_time = event.self_cuda_time_total
        if device_time <= 0 or "kernel" not in event.key.lower():
            continue
        total_device_time_microseconds += device_time
        kernels.append(
            {
                "name": event.key,
                "count": event.count,
                "total_device_seconds": device_time / 1_000_000.0,
                "mean_device_seconds": (
                    device_time / event.count / 1_000_000.0
                ),
            }
        )
    return {
        "iterations": iterations,
        "kernels": kernels,
        "device_seconds_per_operation": (
            total_device_time_microseconds / iterations / 1_000_000.0
        ),
    }


def _case_result(
    case: SpikeCase,
    device: torch.device,
    *,
    warmup_iterations: int,
    measurement_iterations: int,
    launches_per_sample: int,
) -> dict[str, Any]:
    accuracy = _accuracy(case, device)
    if not accuracy["exact"]:
        raise RuntimeError(
            f"{case.shape.name} Triton page write mismatched reference: "
            f"{accuracy}"
        )

    reference_cache = case.layer_kv_cache.clone()
    triton_cache = case.layer_kv_cache.clone()

    def reference() -> None:
        write_kv_cache(
            case.key,
            case.value,
            reference_cache,
            case.metadata,
        )

    def triton_operation() -> None:
        triton_write_kv_cache(
            case.key,
            case.value,
            triton_cache,
            case.metadata,
        )

    return {
        "shape": _shape_document(case.shape),
        "accuracy": accuracy,
        "latency": _measure_pair(
            reference,
            triton_operation,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            launches_per_sample=launches_per_sample,
        ),
        "cuda_kernel_profile": {
            "reference": _profile_cuda_kernels(
                reference,
                device,
                iterations=100,
            ),
            "triton": _profile_cuda_kernels(
                triton_operation,
                device,
                iterations=100,
            ),
        },
    }


def run_step1(
    *,
    warmup_iterations: int,
    measurement_iterations: int,
    launches_per_sample: int,
    expected_gpu_name: str | None = None,
) -> dict[str, Any]:
    """Run batched page-write parity and latency cases."""

    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")
    if launches_per_sample <= 0:
        raise ValueError("launches_per_sample must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("the Triton paged spike requires a CUDA GPU")

    device = torch.device("cuda", torch.cuda.current_device())
    gpu_name = torch.cuda.get_device_name(device)
    _validate_target_gpu(gpu_name, expected_gpu_name)
    properties = torch.cuda.get_device_properties(device)

    cases = {}
    for index, shape in enumerate(SPIKE_SHAPES):
        case = _make_case(
            shape,
            device=device,
            dtype=DTYPE,
            seed=RANDOM_SEED + index,
        )
        cases[shape.name] = _case_result(
            case,
            device,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            launches_per_sample=launches_per_sample,
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "step": "batched_page_write",
        "hardware": {
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "compute_capability": (
                f"{properties.major}.{properties.minor}"
            ),
        },
        "runtime": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "triton_version": triton.__version__,
            "random_seed": RANDOM_SEED,
        },
        "measurement": {
            "warmup_iterations": warmup_iterations,
            "measurement_iterations": measurement_iterations,
            "launches_per_sample": launches_per_sample,
            "timer": "cuda_events_over_repeated_launches",
            "sample_order": "alternating_reference_and_triton",
        },
        "cases": cases,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup-iterations", type=int, default=100)
    parser.add_argument("--measurement-iterations", type=int, default=20)
    parser.add_argument("--launches-per-sample", type=int, default=1000)
    parser.add_argument(
        "--expected-gpu-name",
        help="fail instead of recording results if a different GPU is selected",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_step1(
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
        launches_per_sample=args.launches_per_sample,
        expected_gpu_name=args.expected_gpu_name,
    )
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for case_name, case in result["cases"].items():
        latency = case["latency"]
        reference_us = latency["reference"]["median_seconds"] * 1_000_000
        triton_us = latency["triton"]["median_seconds"] * 1_000_000
        print(
            f"{case_name}: reference={reference_us:.3f} us, "
            f"triton={triton_us:.3f} us, "
            f"speedup={latency['median_speedup']:.3f}x"
        )
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
