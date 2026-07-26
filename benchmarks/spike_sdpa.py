"""Measure the registered PyTorch SDPA transition operator on one CUDA GPU.

The operator uses the reference path's existing paged-cache write and gather
helpers, then calls ``torch.nn.functional.scaled_dot_product_attention`` with
Qwen3-0.6B attention shapes for one prefill and one decode case.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any
import warnings

import torch

from micro_vllm.attention.attention import AttentionMetadata, map_cache_slots
from micro_vllm.attention.operators.reference import (
    ReferenceAttentionOperator,
    gather_kv_cache,
    write_kv_cache,
)
from micro_vllm.attention.operators.sdpa import (
    paged_sdpa_attention,
    prepare_sdpa_inputs as _prepare_sdpa_inputs,
    sdpa_core as _sdpa_core,
    visible_mask as _visible_mask,
)


SCHEMA_VERSION = "micro-vllm.sdpa-feasibility.v1"
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
RANDOM_SEED = 0
DTYPE = torch.float16
ATOL = 5e-2
RTOL = 5e-2


@dataclass(frozen=True, slots=True)
class SpikeShape:
    """One fixed paged-attention shape used by the feasibility spike."""

    name: str
    query_length: int
    context_length: int
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    block_size: int = 16


@dataclass(slots=True)
class SpikeCase:
    """Device tensors and metadata for one fixed shape."""

    shape: SpikeShape
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    layer_kv_cache: torch.Tensor
    metadata: AttentionMetadata


SPIKE_SHAPES = (
    SpikeShape(
        name="prefill",
        query_length=512,
        context_length=512,
    ),
    SpikeShape(
        name="decode",
        query_length=1,
        context_length=512,
    ),
)


def _make_case(
    shape: SpikeShape,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    seed: int,
) -> SpikeCase:
    """Create deterministic paged-cache inputs for one prefill/decode shape."""

    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)

    num_blocks = math.ceil(shape.context_length / shape.block_size)
    first_query_position = shape.context_length - shape.query_length
    query_positions = torch.arange(
        first_query_position,
        shape.context_length,
        dtype=torch.int64,
        device=device,
    )
    block_table = torch.arange(
        num_blocks,
        dtype=torch.int64,
        device=device,
    ).unsqueeze(0)
    metadata = AttentionMetadata(
        query_positions=query_positions,
        query_context_start_positions=torch.zeros_like(query_positions),
        query_start_locations=torch.tensor(
            (0, shape.query_length),
            dtype=torch.int64,
            device=device,
        ),
        context_lengths=torch.tensor(
            (shape.context_length,),
            dtype=torch.int64,
            device=device,
        ),
        block_table=block_table,
        slot_mapping=map_cache_slots(
            query_positions,
            block_table[0],
            shape.block_size,
        ),
        block_size=shape.block_size,
    )
    return SpikeCase(
        shape=shape,
        query=torch.randn(
            shape.query_length,
            shape.num_attention_heads,
            shape.head_dim,
            dtype=dtype,
            device=device,
        ),
        key=torch.randn(
            shape.query_length,
            shape.num_key_value_heads,
            shape.head_dim,
            dtype=dtype,
            device=device,
        ),
        value=torch.randn(
            shape.query_length,
            shape.num_key_value_heads,
            shape.head_dim,
            dtype=dtype,
            device=device,
        ),
        layer_kv_cache=torch.randn(
            2,
            num_blocks,
            shape.block_size,
            shape.num_key_value_heads,
            shape.head_dim,
            dtype=dtype,
            device=device,
        ),
        metadata=metadata,
    )


def _case_arguments(
    case: SpikeCase,
    cache: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    AttentionMetadata,
]:
    return (
        case.query,
        case.key,
        case.value,
        cache,
        case.metadata,
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return (
        ordered[lower_index] * (1.0 - weight)
        + ordered[upper_index] * weight
    )


def _latency_summary(values: Sequence[float]) -> dict[str, int | float]:
    return {
        "count": len(values),
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "p90_seconds": _percentile(values, 0.9),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def _measure_operator(
    operator: Callable[
        [
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            AttentionMetadata,
        ],
        torch.Tensor,
    ],
    case: SpikeCase,
    device: torch.device,
    *,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, int | float]:
    cache = case.layer_kv_cache.clone()
    for _ in range(warmup_iterations):
        operator(*_case_arguments(case, cache))
    torch.cuda.synchronize(device)

    samples: list[float] = []
    for _ in range(measurement_iterations):
        started = time.perf_counter()
        operator(*_case_arguments(case, cache))
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - started)
    return _latency_summary(samples)


def _accuracy(case: SpikeCase, device: torch.device) -> dict[str, float | bool]:
    reference_cache = case.layer_kv_cache.clone()
    sdpa_cache = case.layer_kv_cache.clone()
    reference_output = ReferenceAttentionOperator()(
        *_case_arguments(case, reference_cache)
    )
    sdpa_output = paged_sdpa_attention(*_case_arguments(case, sdpa_cache))
    torch.cuda.synchronize(device)

    difference = (
        reference_output.to(torch.float32) - sdpa_output.to(torch.float32)
    ).abs()
    relative_difference = difference / reference_output.to(
        torch.float32
    ).abs().clamp_min(1e-6)
    close = torch.allclose(
        reference_output,
        sdpa_output,
        atol=ATOL,
        rtol=RTOL,
    )
    return {
        "atol": ATOL,
        "rtol": RTOL,
        "allclose": close,
        "max_absolute_error": difference.max().item(),
        "max_relative_error": relative_difference.max().item(),
    }


def _core_inputs(case: SpikeCase) -> tuple[torch.Tensor, ...]:
    cache = case.layer_kv_cache.clone()
    write_kv_cache(
        case.key,
        case.value,
        cache,
        case.metadata,
    )
    cached_key, cached_value = gather_kv_cache(cache, case.metadata)[0]
    return _prepare_sdpa_inputs(
        case.query,
        cached_key,
        cached_value,
        case.metadata.query_positions,
        case.metadata.query_context_start_positions,
        int(case.metadata.query_context_start_positions[0].item()),
    )


def _profile_kernel(
    case: SpikeCase,
    device: torch.device,
) -> list[str]:
    from torch.profiler import ProfilerActivity, profile

    cache = case.layer_kv_cache.clone()
    with profile(
        activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
        record_shapes=True,
    ) as profiler:
        paged_sdpa_attention(*_case_arguments(case, cache))
        torch.cuda.synchronize(device)
    return sorted(
        event.key
        for event in profiler.key_averages()
        if "scaled_dot_product" in event.key
    )


def _probe_backends(
    case: SpikeCase,
    device: torch.device,
) -> dict[str, Any]:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    inputs = _core_inputs(case)
    backends = {
        "flash_attention": SDPBackend.FLASH_ATTENTION,
        "efficient_attention": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
        "cudnn_attention": SDPBackend.CUDNN_ATTENTION,
    }
    results: dict[str, Any] = {}
    for name, backend in backends.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                with sdpa_kernel(backend):
                    _sdpa_core(*inputs)
                    torch.cuda.synchronize(device)
            except Exception as error:
                results[name] = {
                    "available": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "warnings": [str(item.message) for item in caught],
                }
            else:
                results[name] = {
                    "available": True,
                    "error_type": None,
                    "error": None,
                    "warnings": [str(item.message) for item in caught],
                }
    return results


def _explain_graph(
    function: Callable[..., torch.Tensor],
    arguments: tuple[Any, ...],
) -> dict[str, Any]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            explanation = torch._dynamo.explain(function)(*arguments)
        except Exception as error:
            return {
                "succeeded": False,
                "graph_count": None,
                "graph_break_count": None,
                "break_reasons": [],
                "error_type": type(error).__name__,
                "error": str(error),
                "warnings": [str(item.message) for item in caught],
            }
    return {
        "succeeded": True,
        "graph_count": explanation.graph_count,
        "graph_break_count": explanation.graph_break_count,
        "break_reasons": [
            getattr(reason, "reason", str(reason))
            for reason in explanation.break_reasons
        ],
        "error_type": None,
        "error": None,
        "warnings": [str(item.message) for item in caught],
    }


def _graph_results(case: SpikeCase) -> dict[str, Any]:
    core_inputs = _core_inputs(case)
    paged_cache = case.layer_kv_cache.clone()

    def paged_function(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cache: torch.Tensor,
    ) -> torch.Tensor:
        return paged_sdpa_attention(
            query,
            key,
            value,
            cache,
            case.metadata,
        )

    return {
        "sdpa_core": _explain_graph(_sdpa_core, core_inputs),
        "paged_operator": _explain_graph(
            paged_function,
            (case.query, case.key, case.value, paged_cache),
        ),
    }


def _shape_document(shape: SpikeShape) -> dict[str, int | str]:
    return {
        "name": shape.name,
        "query_length": shape.query_length,
        "context_length": shape.context_length,
        "num_attention_heads": shape.num_attention_heads,
        "num_key_value_heads": shape.num_key_value_heads,
        "head_dim": shape.head_dim,
        "block_size": shape.block_size,
        "kv_page_count": math.ceil(shape.context_length / shape.block_size),
    }


def _validate_target_gpu(
    actual_gpu_name: str,
    expected_gpu_name: str | None,
) -> None:
    """Reject an accidentally selected GPU when the caller pins a target."""

    if expected_gpu_name is not None and actual_gpu_name != expected_gpu_name:
        raise RuntimeError(
            f"expected GPU {expected_gpu_name!r}, found {actual_gpu_name!r}"
        )


def run_spike(
    *,
    warmup_iterations: int,
    measurement_iterations: int,
    expected_gpu_name: str | None = None,
) -> dict[str, Any]:
    """Run both fixed cases and return one JSON-ready feasibility document."""

    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("the SDPA feasibility spike requires a CUDA GPU")

    device = torch.device("cuda", torch.cuda.current_device())
    gpu_name = torch.cuda.get_device_name(device)
    _validate_target_gpu(gpu_name, expected_gpu_name)

    case_results: dict[str, Any] = {}
    for index, shape in enumerate(SPIKE_SHAPES):
        case = _make_case(
            shape,
            device=device,
            dtype=DTYPE,
            seed=RANDOM_SEED + index,
        )
        accuracy = _accuracy(case, device)
        if not accuracy["allclose"]:
            raise RuntimeError(
                f"{shape.name} SDPA output exceeded frozen tolerance: {accuracy}"
            )
        reference_latency = _measure_operator(
            ReferenceAttentionOperator(),
            case,
            device,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
        )
        sdpa_latency = _measure_operator(
            paged_sdpa_attention,
            case,
            device,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
        )
        case_results[shape.name] = {
            "shape": _shape_document(shape),
            "accuracy": accuracy,
            "latency": {
                "reference": reference_latency,
                "sdpa": sdpa_latency,
                "median_speedup": (
                    reference_latency["median_seconds"]
                    / sdpa_latency["median_seconds"]
                ),
            },
            "profiled_sdpa_operators": _profile_kernel(case, device),
            "forced_backend_availability": _probe_backends(case, device),
            "graphs": _graph_results(case),
        }

    properties = torch.cuda.get_device_properties(device)
    return {
        "schema_version": SCHEMA_VERSION,
        "model_shape": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "dtype": str(DTYPE),
            "manual_gqa_expansion": True,
            "paged_cache_helpers": [
                "write_kv_cache",
                "gather_kv_cache",
            ],
        },
        "hardware": {
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
        },
        "runtime": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "random_seed": RANDOM_SEED,
            "global_sdpa_flags": {
                "flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
                "efficient_enabled": (
                    torch.backends.cuda.mem_efficient_sdp_enabled()
                ),
                "math_enabled": torch.backends.cuda.math_sdp_enabled(),
                "cudnn_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
            },
        },
        "measurement": {
            "warmup_iterations": warmup_iterations,
            "measurement_iterations": measurement_iterations,
            "timer": "perf_counter_with_per_iteration_cuda_synchronize",
        },
        "cases": case_results,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--measurement-iterations", type=int, default=100)
    parser.add_argument(
        "--expected-gpu-name",
        help="fail instead of recording results if a different GPU is selected",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_spike(
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
        expected_gpu_name=args.expected_gpu_name,
    )
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, case in result["cases"].items():
        latency = case["latency"]
        print(
            f"{name}: {latency['median_speedup']:.3f}x median speedup, "
            f"kernels={case['profiled_sdpa_operators']}"
        )
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
