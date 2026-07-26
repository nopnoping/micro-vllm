"""Isolate the stages around the Phase 1 paged SDPA feasibility path.

Each component is warmed up and timed independently with a CUDA synchronize
after every sample. These isolated latencies identify promising optimization
targets, but their medians are not additive: every component pays its own
Python-call, launch, and synchronization overhead. The full paged SDPA latency
remains the end-to-end operator measurement.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import torch

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
from micro_vllm.attention.operators.reference import (
    gather_kv_cache,
    write_kv_cache,
)
from micro_vllm.attention.operators.sdpa import (
    paged_sdpa_attention,
    prepare_sdpa_inputs as _prepare_sdpa_inputs,
    sdpa_core as _sdpa_core,
)


SCHEMA_VERSION = "micro-vllm.attention-breakdown.v1"


@dataclass(frozen=True, slots=True)
class RequestInputs:
    """One request's views and scalar bounds before dense SDPA preparation."""

    query: torch.Tensor
    cached_key: torch.Tensor
    cached_value: torch.Tensor
    query_positions: torch.Tensor
    query_context_starts: torch.Tensor
    context_start: int
    output_start: int
    output_end: int


def _extract_request_inputs(
    case: SpikeCase,
    histories: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    request_index: int,
) -> RequestInputs:
    """Reproduce the per-request scalar extraction and flattened slicing."""

    query_start = case.metadata.query_start_locations[request_index].item()
    query_end = case.metadata.query_start_locations[request_index + 1].item()
    context_start = case.metadata.query_context_start_positions[
        query_start
    ].item()
    cached_key, cached_value = histories[request_index]
    return RequestInputs(
        query=case.query[query_start:query_end],
        cached_key=cached_key,
        cached_value=cached_value,
        query_positions=case.metadata.query_positions[query_start:query_end],
        query_context_starts=(
            case.metadata.query_context_start_positions[query_start:query_end]
        ),
        context_start=context_start,
        output_start=query_start,
        output_end=query_end,
    )


def _measure_operation(
    operation: Callable[[], object],
    device: torch.device,
    *,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, int | float]:
    for _ in range(warmup_iterations):
        operation()
    torch.cuda.synchronize(device)

    samples: list[float] = []
    for _ in range(measurement_iterations):
        started = time.perf_counter()
        operation()
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - started)
    return _latency_summary(samples)


def _case_breakdown(
    case: SpikeCase,
    device: torch.device,
    *,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, Any]:
    cache = case.layer_kv_cache.clone()
    write_kv_cache(case.key, case.value, cache, case.metadata)
    histories = gather_kv_cache(cache, case.metadata)
    request_inputs = _extract_request_inputs(case, histories, 0)
    sdpa_inputs = _prepare_sdpa_inputs(
        request_inputs.query,
        request_inputs.cached_key,
        request_inputs.cached_value,
        request_inputs.query_positions,
        request_inputs.query_context_starts,
        request_inputs.context_start,
    )
    attention = _sdpa_core(*sdpa_inputs)
    output = torch.empty_like(case.query)
    torch.cuda.synchronize(device)

    def write() -> None:
        write_kv_cache(case.key, case.value, cache, case.metadata)

    def gather() -> object:
        return gather_kv_cache(cache, case.metadata)

    def extract_request() -> object:
        return _extract_request_inputs(case, histories, 0)

    def prepare() -> object:
        return _prepare_sdpa_inputs(
            request_inputs.query,
            request_inputs.cached_key,
            request_inputs.cached_value,
            request_inputs.query_positions,
            request_inputs.query_context_starts,
            request_inputs.context_start,
        )

    def attention_core() -> object:
        return _sdpa_core(*sdpa_inputs)

    def copy_output() -> None:
        output[
            request_inputs.output_start : request_inputs.output_end
        ].copy_(attention.squeeze(0).transpose(0, 1))

    def full_operator() -> object:
        return paged_sdpa_attention(
            case.query,
            case.key,
            case.value,
            cache,
            case.metadata,
        )

    components = {
        "write_kv_cache": write,
        "gather_kv_cache": gather,
        "extract_request_metadata_and_views": extract_request,
        "prepare_sdpa_inputs": prepare,
        "sdpa_core": attention_core,
        "copy_output": copy_output,
    }
    return {
        "shape": _shape_document(case.shape),
        "full_paged_sdpa": _measure_operation(
            full_operator,
            device,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
        ),
        "isolated_components": {
            name: _measure_operation(
                operation,
                device,
                warmup_iterations=warmup_iterations,
                measurement_iterations=measurement_iterations,
            )
            for name, operation in components.items()
        },
    }


def run_breakdown(
    *,
    warmup_iterations: int,
    measurement_iterations: int,
    expected_gpu_name: str | None = None,
) -> dict[str, Any]:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("the attention breakdown requires a CUDA GPU")

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
        cases[shape.name] = _case_breakdown(
            case,
            device,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
        )

    return {
        "schema_version": SCHEMA_VERSION,
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
            "random_seed": RANDOM_SEED,
        },
        "measurement": {
            "warmup_iterations": warmup_iterations,
            "measurement_iterations": measurement_iterations,
            "timer": "isolated_perf_counter_with_per_sample_cuda_synchronize",
            "isolated_component_medians_are_additive": False,
        },
        "cases": cases,
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
    result = run_breakdown(
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
    for case_name, case in result["cases"].items():
        full_ms = case["full_paged_sdpa"]["median_seconds"] * 1000
        component_text = ", ".join(
            f"{name}={latency['median_seconds'] * 1000:.3f} ms"
            for name, latency in case["isolated_components"].items()
        )
        print(f"{case_name}: full={full_ms:.3f} ms; {component_text}")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
