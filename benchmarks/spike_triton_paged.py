"""Build the Triton paged-attention path one address operation at a time.

Step 1 writes every flattened input token and KV head in one launch. Step 2
gathers every request through its block table in one launch, materializing a
padded teaching output that will not enter the final runtime path. Step 3
measures the registered direct-paged operator, which reuses the same page
address calculation without a gathered K/V tensor.
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
    ATOL,
    DTYPE,
    RANDOM_SEED,
    RTOL,
    SPIKE_SHAPES,
    SpikeCase,
    _latency_summary,
    _make_case,
    _shape_document,
    _validate_target_gpu,
)
from micro_vllm.attention.attention import AttentionMetadata
from micro_vllm.attention.operators.reference import (
    ReferenceAttentionOperator,
    gather_kv_cache,
    write_kv_cache,
)
from micro_vllm.attention.operators.sdpa import paged_sdpa_attention
from micro_vllm.attention.operators.triton_paged import (
    triton_paged_attention,
    triton_write_kv_cache,
)


SCHEMA_VERSION = "micro-vllm.triton-paged-step1.v1"


@triton.jit
def _gather_kv_cache_kernel(
    cache_ptr,
    query_context_start_positions_ptr,
    query_start_locations_ptr,
    context_lengths_ptr,
    block_table_ptr,
    output_ptr,
    cache_plane_stride,
    cache_block_stride,
    cache_offset_stride,
    cache_head_stride,
    cache_dim_stride,
    block_table_request_stride,
    block_table_block_stride,
    output_plane_stride,
    output_request_stride,
    output_token_stride,
    output_head_stride,
    output_dim_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    request_index = tl.program_id(0)
    context_offset = tl.program_id(1)
    kv_head_index = tl.program_id(2)
    dim_offsets = tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offsets < HEAD_DIM

    query_start = tl.load(query_start_locations_ptr + request_index)
    context_start = tl.load(
        query_context_start_positions_ptr + query_start
    )
    context_length = tl.load(context_lengths_ptr + request_index)
    logical_position = context_start + context_offset
    token_mask = logical_position < context_length

    logical_block = logical_position // BLOCK_SIZE
    physical_block = tl.load(
        block_table_ptr
        + request_index * block_table_request_stride
        + logical_block * block_table_block_stride,
        mask=token_mask,
        other=0,
    )
    block_offset = logical_position % BLOCK_SIZE
    cache_offsets = (
        physical_block * cache_block_stride
        + block_offset * cache_offset_stride
        + kv_head_index * cache_head_stride
        + dim_offsets * cache_dim_stride
    )
    output_offsets = (
        request_index * output_request_stride
        + context_offset * output_token_stride
        + kv_head_index * output_head_stride
        + dim_offsets * output_dim_stride
    )
    element_mask = token_mask & dim_mask

    key = tl.load(cache_ptr + cache_offsets, mask=element_mask)
    value = tl.load(
        cache_ptr + cache_plane_stride + cache_offsets,
        mask=element_mask,
    )
    tl.store(output_ptr + output_offsets, key, mask=element_mask)
    tl.store(
        output_ptr + output_plane_stride + output_offsets,
        value,
        mask=element_mask,
    )


def triton_gather_kv_cache(
    layer_kv_cache: torch.Tensor,
    metadata: AttentionMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather every request into one padded tensor with one kernel launch."""

    num_requests = metadata.num_requests
    context_capacity = metadata.block_table.shape[1] * metadata.block_size
    num_kv_heads = layer_kv_cache.shape[3]
    head_dim = layer_kv_cache.shape[4]
    output = torch.empty(
        (
            2,
            num_requests,
            context_capacity,
            num_kv_heads,
            head_dim,
        ),
        dtype=layer_kv_cache.dtype,
        device=layer_kv_cache.device,
    )
    block_dim = triton.next_power_of_2(head_dim)
    num_warps = 4 if block_dim >= 64 else 1
    grid = (num_requests, context_capacity, num_kv_heads)
    _gather_kv_cache_kernel[grid](
        layer_kv_cache,
        metadata.query_context_start_positions,
        metadata.query_start_locations,
        metadata.context_lengths,
        metadata.block_table,
        output,
        layer_kv_cache.stride(0),
        layer_kv_cache.stride(1),
        layer_kv_cache.stride(2),
        layer_kv_cache.stride(3),
        layer_kv_cache.stride(4),
        metadata.block_table.stride(0),
        metadata.block_table.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        output.stride(4),
        HEAD_DIM=head_dim,
        BLOCK_SIZE=metadata.block_size,
        BLOCK_DIM=block_dim,
        num_warps=num_warps,
    )
    return output[0], output[1]


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


def _gather_accuracy(
    case: SpikeCase,
    device: torch.device,
) -> dict[str, Any]:
    reference = gather_kv_cache(case.layer_kv_cache, case.metadata)
    triton_key, triton_value = triton_gather_kv_cache(
        case.layer_kv_cache,
        case.metadata,
    )
    torch.cuda.synchronize(device)

    exact = True
    max_absolute_error = 0.0
    for request_index, (reference_key, reference_value) in enumerate(
        reference
    ):
        context_tokens = reference_key.shape[0]
        actual_key = triton_key[request_index, :context_tokens]
        actual_value = triton_value[request_index, :context_tokens]
        exact = exact and torch.equal(actual_key, reference_key)
        exact = exact and torch.equal(actual_value, reference_value)
        key_error = (
            actual_key.to(torch.float32) - reference_key.to(torch.float32)
        ).abs().max().item()
        value_error = (
            actual_value.to(torch.float32)
            - reference_value.to(torch.float32)
        ).abs().max().item()
        max_absolute_error = max(
            max_absolute_error,
            key_error,
            value_error,
        )
    return {
        "exact": exact,
        "max_absolute_error": max_absolute_error,
    }


def _attention_accuracy(
    case: SpikeCase,
    device: torch.device,
) -> dict[str, Any]:
    reference_cache = case.layer_kv_cache.clone()
    sdpa_cache = case.layer_kv_cache.clone()
    triton_cache = case.layer_kv_cache.clone()
    reference = ReferenceAttentionOperator()(
        case.query,
        case.key,
        case.value,
        reference_cache,
        case.metadata,
    )
    sdpa = paged_sdpa_attention(
        case.query,
        case.key,
        case.value,
        sdpa_cache,
        case.metadata,
    )
    triton_output = triton_paged_attention(
        case.query,
        case.key,
        case.value,
        triton_cache,
        case.metadata,
    )
    torch.cuda.synchronize(device)

    reference_float = reference.to(torch.float32)

    def comparison(
        output: torch.Tensor,
        cache: torch.Tensor,
    ) -> dict[str, Any]:
        difference = (reference_float - output.to(torch.float32)).abs()
        relative_difference = difference / reference_float.abs().clamp_min(
            1e-6
        )
        return {
            "allclose": torch.allclose(
                reference,
                output,
                atol=ATOL,
                rtol=RTOL,
            ),
            "cache_exact": torch.equal(reference_cache, cache),
            "max_absolute_error": difference.max().item(),
            "max_relative_error": relative_difference.max().item(),
        }

    sdpa_comparison = comparison(sdpa, sdpa_cache)
    triton_comparison = comparison(triton_output, triton_cache)
    return {
        "atol": ATOL,
        "rtol": RTOL,
        "allclose": (
            sdpa_comparison["allclose"]
            and sdpa_comparison["cache_exact"]
            and triton_comparison["allclose"]
            and triton_comparison["cache_exact"]
        ),
        "sdpa": sdpa_comparison,
        "triton": triton_comparison,
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


def _measure_interleaved(
    operations: dict[str, Callable[[], object]],
    *,
    warmup_iterations: int,
    measurement_iterations: int,
    launches_per_sample: int,
) -> dict[str, Any]:
    """Measure every complete path under one rotating sample order."""

    operation_names = tuple(operations)
    for _ in range(warmup_iterations):
        for operation in operations.values():
            operation()
    torch.cuda.synchronize()

    samples = {name: [] for name in operation_names}
    for sample_index in range(measurement_iterations):
        rotation = sample_index % len(operation_names)
        order = operation_names[rotation:] + operation_names[:rotation]
        for name in order:
            samples[name].append(
                _measure_launch_batch(
                    operations[name],
                    launches_per_sample=launches_per_sample,
                )
            )

    summaries = {
        name: _latency_summary(samples[name])
        for name in operation_names
    }
    reference_median = summaries["reference"]["median_seconds"]
    sdpa_median = summaries["sdpa"]["median_seconds"]
    triton_median = summaries["triton"]["median_seconds"]
    return {
        **summaries,
        "median_speedup": reference_median / triton_median,
        "median_speedups": {
            "reference_over_sdpa": reference_median / sdpa_median,
            "reference_over_triton": reference_median / triton_median,
            "sdpa_over_triton": sdpa_median / triton_median,
        },
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
        if device_time <= 0 or event.key.startswith(
            (
                "aten::",
                "cuda",
                "Activity Buffer",
                "Runtime Triggered",
                "Lazy Function",
            )
        ):
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


def _gather_case_result(
    case: SpikeCase,
    device: torch.device,
    *,
    warmup_iterations: int,
    measurement_iterations: int,
    launches_per_sample: int,
) -> dict[str, Any]:
    accuracy = _gather_accuracy(case, device)
    if not accuracy["exact"]:
        raise RuntimeError(
            f"{case.shape.name} Triton page gather mismatched reference: "
            f"{accuracy}"
        )

    def reference() -> object:
        return gather_kv_cache(case.layer_kv_cache, case.metadata)

    def triton_operation() -> object:
        return triton_gather_kv_cache(
            case.layer_kv_cache,
            case.metadata,
        )

    return {
        "shape": _shape_document(case.shape),
        "padded_context_capacity": (
            case.metadata.block_table.shape[1] * case.metadata.block_size
        ),
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


def _attention_case_result(
    case: SpikeCase,
    device: torch.device,
    *,
    warmup_iterations: int,
    measurement_iterations: int,
    launches_per_sample: int,
) -> dict[str, Any]:
    accuracy = _attention_accuracy(case, device)
    if not accuracy["allclose"]:
        raise RuntimeError(
            f"{case.shape.name} attention comparison mismatched "
            f"reference: {accuracy}"
        )

    reference_cache = case.layer_kv_cache.clone()
    sdpa_cache = case.layer_kv_cache.clone()
    triton_cache = case.layer_kv_cache.clone()
    reference_operator = ReferenceAttentionOperator()

    def reference() -> object:
        return reference_operator(
            case.query,
            case.key,
            case.value,
            reference_cache,
            case.metadata,
        )

    def sdpa_operation() -> object:
        return paged_sdpa_attention(
            case.query,
            case.key,
            case.value,
            sdpa_cache,
            case.metadata,
        )

    def triton_operation() -> object:
        return triton_paged_attention(
            case.query,
            case.key,
            case.value,
            triton_cache,
            case.metadata,
        )

    return {
        "shape": _shape_document(case.shape),
        "accuracy": accuracy,
        "latency": _measure_interleaved(
            {
                "reference": reference,
                "sdpa": sdpa_operation,
                "triton": triton_operation,
            },
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            launches_per_sample=launches_per_sample,
        ),
        "cuda_kernel_profile": {
            "reference": _profile_cuda_kernels(
                reference,
                device,
                iterations=20,
            ),
            "sdpa": _profile_cuda_kernels(
                sdpa_operation,
                device,
                iterations=20,
            ),
            "triton": _profile_cuda_kernels(
                triton_operation,
                device,
                iterations=20,
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


def run_step2(
    *,
    warmup_iterations: int,
    measurement_iterations: int,
    launches_per_sample: int,
    expected_gpu_name: str | None = None,
) -> dict[str, Any]:
    """Run one-launch padded page-gather parity and latency cases."""

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
        cases[shape.name] = _gather_case_result(
            case,
            device,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            launches_per_sample=launches_per_sample,
        )

    return {
        "schema_version": "micro-vllm.triton-paged-step2.v1",
        "step": "batched_page_gather",
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
            "materialization": "padded_learning_output_only",
        },
        "cases": cases,
    }


def run_step3(
    *,
    warmup_iterations: int,
    measurement_iterations: int,
    launches_per_sample: int,
    expected_gpu_name: str | None = None,
) -> dict[str, Any]:
    """Run fused ragged paged-attention parity and latency cases."""

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
        cases[shape.name] = _attention_case_result(
            case,
            device,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            launches_per_sample=launches_per_sample,
        )

    return {
        "schema_version": "micro-vllm.triton-paged-step3.v2",
        "step": "fused_ragged_paged_attention",
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
            "sample_order": "rotating_reference_sdpa_triton",
            "compared_complete_paths": [
                "reference",
                "paged_sdpa",
                "triton_paged",
            ],
            "path_intermediate_gather_tensor": {
                "reference": True,
                "paged_sdpa": True,
                "triton_paged": False,
            },
            "triton_request_mapping": (
                "packed_query_block_gpu_binary_search"
            ),
            "triton_attention_accumulation": "fp32_online_softmax",
        },
        "cases": cases,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=("page-write", "page-gather", "paged-attention"),
        default="page-write",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup-iterations", type=int, default=100)
    parser.add_argument("--measurement-iterations", type=int, default=20)
    parser.add_argument("--launches-per-sample", type=int)
    parser.add_argument(
        "--expected-gpu-name",
        help="fail instead of recording results if a different GPU is selected",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    runners = {
        "page-write": run_step1,
        "page-gather": run_step2,
        "paged-attention": run_step3,
    }
    default_launches_per_sample = {
        "page-write": 1000,
        "page-gather": 100,
        "paged-attention": 10,
    }
    runner = runners[args.step]
    launches_per_sample = args.launches_per_sample
    if launches_per_sample is None:
        launches_per_sample = default_launches_per_sample[args.step]
    result = runner(
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
        launches_per_sample=launches_per_sample,
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
        sdpa_text = ""
        if "sdpa" in latency:
            sdpa_us = latency["sdpa"]["median_seconds"] * 1_000_000
            sdpa_over_triton = latency["median_speedups"][
                "sdpa_over_triton"
            ]
            sdpa_text = (
                f"sdpa={sdpa_us:.3f} us, "
                f"sdpa/triton={sdpa_over_triton:.3f}x, "
            )
        print(
            f"{result['step']} {case_name}: "
            f"reference={reference_us:.3f} us, "
            f"{sdpa_text}"
            f"triton={triton_us:.3f} us, "
            f"speedup={latency['median_speedup']:.3f}x"
        )
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
