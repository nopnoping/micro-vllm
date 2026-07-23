"""Run the frozen V1 tokenized Engine reference benchmark.

The benchmark intentionally bypasses the tokenizer and drives ``Engine.step``
directly. Model initialization and warmup are excluded from measured latency.
Every measurement repetition uses a fresh Scheduler/KVCacheManager pair while
reusing the initialized model and physical KV buffers.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

import torch

from micro_vllm.config import CacheConfig, LLMConfig, ModelConfig, SchedulerConfig
from micro_vllm.core.kv_cache_manager import KVCacheManager
from micro_vllm.core.scheduler import Scheduler
from micro_vllm.core.scheduler_output import SchedulerOutput
from micro_vllm.engine.engine import Engine
from micro_vllm.engine.output import RequestOutput
from micro_vllm.executor.interface import ModelRunnerOutput
from micro_vllm.executor.single_process import SingleProcessExecutor
from micro_vllm.sampling.params import SamplingParams
from micro_vllm.worker.worker import Worker


SCHEMA_VERSION = "micro-vllm.engine-benchmark.v1"
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
ATTENTION_OPERATOR = "reference"
RANDOM_SEED = 0
DTYPE = torch.float16
BLOCK_SIZE = 16
_VOCAB_SIZE = 151_936


@dataclass(frozen=True, slots=True)
class RequestSpec:
    """One fixed tokenized request and its admission point."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    admit_after_steps: int = 0


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    """Frozen scheduler, cache, and request inputs for one benchmark case."""

    name: str
    description: str
    max_num_batched_tokens: int
    max_num_seqs: int
    max_model_len: int
    num_blocks: int
    enable_prefix_cache: bool
    requests: tuple[RequestSpec, ...]


@dataclass(frozen=True, slots=True)
class RequestMeasurement:
    """Token timing observed from cumulative RequestOutput snapshots."""

    request_id: str
    prompt_tokens: int
    output_tokens: int
    ttft_seconds: float | None
    itl_seconds: tuple[float, ...]
    finished: bool


@dataclass(frozen=True, slots=True)
class RepetitionMeasurement:
    """Raw measurements for one fresh logical-engine repetition."""

    elapsed_seconds: float
    cuda_peak_allocated_bytes: int
    peak_used_kv_pages: int
    preempted_request_count: int
    requests: tuple[RequestMeasurement, ...]


def _fixed_tokens(length: int, *, offset: int) -> tuple[int, ...]:
    """Return stable in-vocabulary IDs without tokenizer or random input."""

    return tuple(
        ((offset + token_index * 7_919) % (_VOCAB_SIZE - 1)) + 1
        for token_index in range(length)
    )


WORKLOADS: dict[str, WorkloadSpec] = {
    "prefill-16": WorkloadSpec(
        name="prefill-16",
        description="single request, 16 prompt tokens, one output token",
        max_num_batched_tokens=512,
        max_num_seqs=1,
        max_model_len=32,
        num_blocks=64,
        enable_prefix_cache=False,
        requests=(RequestSpec("prefill", _fixed_tokens(16, offset=11), 1),),
    ),
    "prefill-128": WorkloadSpec(
        name="prefill-128",
        description="single request, 128 prompt tokens, one output token",
        max_num_batched_tokens=512,
        max_num_seqs=1,
        max_model_len=160,
        num_blocks=64,
        enable_prefix_cache=False,
        requests=(RequestSpec("prefill", _fixed_tokens(128, offset=23), 1),),
    ),
    "prefill-512": WorkloadSpec(
        name="prefill-512",
        description="single request, 512 prompt tokens, one output token",
        max_num_batched_tokens=512,
        max_num_seqs=1,
        max_model_len=544,
        num_blocks=64,
        enable_prefix_cache=False,
        requests=(RequestSpec("prefill", _fixed_tokens(512, offset=37), 1),),
    ),
    "decode": WorkloadSpec(
        name="decode",
        description="short prompt followed by a fixed 128-token decode",
        max_num_batched_tokens=128,
        max_num_seqs=1,
        max_model_len=160,
        num_blocks=32,
        enable_prefix_cache=False,
        requests=(RequestSpec("decode", _fixed_tokens(8, offset=41), 128),),
    ),
    "mixed": WorkloadSpec(
        name="mixed",
        description="running decode plus a newly admitted 128-token prefill",
        max_num_batched_tokens=128,
        max_num_seqs=2,
        max_model_len=160,
        num_blocks=32,
        enable_prefix_cache=False,
        requests=(
            RequestSpec("decode", _fixed_tokens(8, offset=53), 64),
            RequestSpec(
                "prefill",
                _fixed_tokens(128, offset=67),
                1,
                admit_after_steps=1,
            ),
        ),
    ),
    "cache-pressure": WorkloadSpec(
        name="cache-pressure",
        description="two decodes exceed shared page capacity and force preemption",
        max_num_batched_tokens=64,
        max_num_seqs=2,
        max_model_len=128,
        num_blocks=10,
        enable_prefix_cache=False,
        requests=(
            RequestSpec("first", _fixed_tokens(32, offset=79), 64),
            RequestSpec("second", _fixed_tokens(32, offset=97), 64),
        ),
    ),
}


class _MeasuringExecutor:
    """Observe allocated logical pages immediately before model execution."""

    def __init__(
        self,
        executor: SingleProcessExecutor,
        cache_manager: KVCacheManager,
        num_blocks: int,
    ) -> None:
        self._executor = executor
        self._cache_manager = cache_manager
        self._num_data_blocks = num_blocks - 1
        self.peak_used_kv_pages = 0
        self.preempted_request_count = 0

    def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput:
        used_pages = self._num_data_blocks - self._cache_manager.num_free_blocks
        self.peak_used_kv_pages = max(self.peak_used_kv_pages, used_pages)
        self.preempted_request_count += len(plan.preempted_request_ids)
        return self._executor.execute(plan)

    def shutdown(self) -> None:
        self._executor.shutdown()


@dataclass(slots=True)
class _RequestClock:
    admitted_at_seconds: float
    prompt_tokens: int
    output_tokens: int = 0
    first_token_at_seconds: float | None = None
    last_token_at_seconds: float | None = None
    itl_seconds: tuple[float, ...] = ()
    finished: bool = False

    def observe(self, output: RequestOutput, now_seconds: float) -> None:
        new_token_count = len(output.output_token_ids) - self.output_tokens
        if new_token_count < 0:
            raise RuntimeError("RequestOutput token count moved backwards")
        for _ in range(new_token_count):
            if self.first_token_at_seconds is None:
                self.first_token_at_seconds = now_seconds
            elif self.last_token_at_seconds is not None:
                self.itl_seconds += (now_seconds - self.last_token_at_seconds,)
            self.last_token_at_seconds = now_seconds
        self.output_tokens = len(output.output_token_ids)
        self.finished = output.finished

    def snapshot(self, request_id: str) -> RequestMeasurement:
        ttft_seconds = None
        if self.first_token_at_seconds is not None:
            ttft_seconds = (
                self.first_token_at_seconds - self.admitted_at_seconds
            )
        return RequestMeasurement(
            request_id=request_id,
            prompt_tokens=self.prompt_tokens,
            output_tokens=self.output_tokens,
            ttft_seconds=ttft_seconds,
            itl_seconds=self.itl_seconds,
            finished=self.finished,
        )


def _synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _sampling_params(request: RequestSpec) -> SamplingParams:
    return SamplingParams(
        max_tokens=request.max_tokens,
        ignore_eos=True,
        temperature=0.0,
        seed=RANDOM_SEED,
    )


def _run_repetition(
    worker: Worker,
    config: LLMConfig,
    workload: WorkloadSpec,
    device: torch.device,
    *,
    measure_memory: bool,
) -> RepetitionMeasurement:
    """Drive a fresh logical Engine and time only synchronized Engine.step calls."""

    cache_manager = KVCacheManager(config, worker.kv_cache_layout)
    scheduler = Scheduler(config, cache_manager)
    executor = _MeasuringExecutor(
        SingleProcessExecutor(worker),
        cache_manager,
        worker.kv_cache_layout.num_blocks,
    )
    engine = Engine(scheduler, executor)
    pending = list(workload.requests)
    clocks: dict[str, _RequestClock] = {}
    elapsed_seconds = 0.0
    completed_steps = 0

    def admit_ready_requests() -> None:
        ready = [
            request
            for request in pending
            if request.admit_after_steps == completed_steps
        ]
        for request in ready:
            engine.add_request(
                request.request_id,
                list(request.prompt_token_ids),
                _sampling_params(request),
            )
            clocks[request.request_id] = _RequestClock(
                admitted_at_seconds=elapsed_seconds,
                prompt_tokens=len(request.prompt_token_ids),
            )
            pending.remove(request)

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    _synchronize(device)
    if measure_memory:
        torch.cuda.reset_peak_memory_stats(device)
    admit_ready_requests()
    max_steps = 10 * sum(
        len(request.prompt_token_ids) + request.max_tokens
        for request in workload.requests
    )

    while engine.has_unfinished_requests() or pending:
        if completed_steps >= max_steps:
            raise RuntimeError(
                f"workload {workload.name!r} exceeded {max_steps} Engine steps"
            )
        started = time.perf_counter()
        outputs = engine.step()
        _synchronize(device)
        elapsed_seconds += time.perf_counter() - started
        completed_steps += 1
        for output in outputs:
            clocks[output.request_id].observe(output, elapsed_seconds)
        admit_ready_requests()

    peak_memory = (
        torch.cuda.max_memory_allocated(device) if measure_memory else 0
    )
    requests = tuple(
        clocks[request.request_id].snapshot(request.request_id)
        for request in workload.requests
    )
    if not all(request.finished for request in requests):
        raise RuntimeError(f"workload {workload.name!r} did not finish")
    return RepetitionMeasurement(
        elapsed_seconds=elapsed_seconds,
        cuda_peak_allocated_bytes=peak_memory,
        peak_used_kv_pages=executor.peak_used_kv_pages,
        preempted_request_count=executor.preempted_request_count,
        requests=requests,
    )


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile, or None for no samples."""

    if not values:
        return None
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


def _latency_summary(values: Sequence[float]) -> dict[str, int | float | None]:
    return {
        "count": len(values),
        "median_seconds": statistics.median(values) if values else None,
        "p90_seconds": _percentile(values, 0.9),
    }


def _request_document(request: RequestMeasurement) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "prompt_tokens": request.prompt_tokens,
        "output_tokens": request.output_tokens,
        "ttft_seconds": request.ttft_seconds,
        "itl_seconds": list(request.itl_seconds),
        "finished": request.finished,
    }


def _repetition_document(
    index: int,
    repetition: RepetitionMeasurement,
) -> dict[str, Any]:
    return {
        "index": index,
        "elapsed_seconds": repetition.elapsed_seconds,
        "cuda_peak_allocated_bytes": repetition.cuda_peak_allocated_bytes,
        "peak_used_kv_pages": repetition.peak_used_kv_pages,
        "preempted_request_count": repetition.preempted_request_count,
        "requests": [
            _request_document(request) for request in repetition.requests
        ],
    }


def _physical_kv_bytes(config: LLMConfig, worker: Worker) -> int:
    values = config.model_config.values
    num_attention_heads = int(values["num_attention_heads"])
    num_key_value_heads = int(
        values.get("num_key_value_heads", num_attention_heads)
    )
    head_dim = int(
        values.get("head_dim", int(values["hidden_size"]) // num_attention_heads)
    )
    layout = worker.kv_cache_layout
    element_size = torch.empty((), dtype=config.model_config.dtype).element_size()
    return (
        layout.page_layer_count
        * 2
        * layout.num_blocks
        * layout.block_size
        * num_key_value_heads
        * head_dim
        * element_size
    )


def _workload_document(
    workload: WorkloadSpec,
    warmup_repetitions: int,
    measurement_repetitions: int,
) -> dict[str, Any]:
    return {
        "name": workload.name,
        "description": workload.description,
        "warmup_repetitions": warmup_repetitions,
        "measurement_repetitions": measurement_repetitions,
        "requests": [
            {
                "request_id": request.request_id,
                "prompt_token_ids": list(request.prompt_token_ids),
                "max_tokens": request.max_tokens,
                "admit_after_steps": request.admit_after_steps,
            }
            for request in workload.requests
        ],
    }


def _build_document(
    config: LLMConfig,
    worker: Worker,
    workload: WorkloadSpec,
    warmup_repetitions: int,
    repetitions: Sequence[RepetitionMeasurement],
    device: torch.device,
) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    elapsed_seconds = sum(item.elapsed_seconds for item in repetitions)
    requests = [request for item in repetitions for request in item.requests]
    prompt_tokens = sum(request.prompt_tokens for request in requests)
    output_tokens = sum(request.output_tokens for request in requests)
    ttft_values = [
        request.ttft_seconds
        for request in requests
        if request.ttft_seconds is not None
    ]
    itl_values = [value for request in requests for value in request.itl_seconds]
    layout = worker.kv_cache_layout
    num_data_pages = layout.num_blocks - 1
    peak_used_pages = max(item.peak_used_kv_pages for item in repetitions)

    return {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "local_files_only": True,
            "cache_dir": None,
            "dtype": str(DTYPE),
            "attention_operator": ATTENTION_OPERATOR,
        },
        "hardware": {
            "device_type": device.type,
            "device_index": device.index,
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "random_seed": RANDOM_SEED,
            "sampling": "greedy",
            "execution_mode": "eager",
        },
        "capacity": {
            "block_size_tokens": layout.block_size,
            "requested_kv_page_count": workload.num_blocks,
            "kv_page_count": layout.num_blocks,
            "kv_data_page_count": num_data_pages,
            "physical_kv_bytes": _physical_kv_bytes(config, worker),
            "physical_kv_buffer_count": layout.page_layer_count,
            "gpu_memory_utilization": config.cache_config.gpu_memory_utilization,
            "max_num_batched_tokens": workload.max_num_batched_tokens,
            "max_num_seqs": workload.max_num_seqs,
            "max_model_len": workload.max_model_len,
            "enable_prefix_cache": workload.enable_prefix_cache,
        },
        "workload": _workload_document(
            workload,
            warmup_repetitions,
            len(repetitions),
        ),
        "summary": {
            "elapsed_seconds": elapsed_seconds,
            "completed_requests": len(requests),
            "request_throughput_per_second": len(requests) / elapsed_seconds,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
            "prompt_token_throughput_per_second": prompt_tokens / elapsed_seconds,
            "output_token_throughput_per_second": output_tokens / elapsed_seconds,
            "total_token_throughput_per_second": (
                prompt_tokens + output_tokens
            )
            / elapsed_seconds,
            "ttft": _latency_summary(ttft_values),
            "itl": _latency_summary(itl_values),
            "cuda_peak_allocated_bytes": max(
                item.cuda_peak_allocated_bytes for item in repetitions
            ),
            "peak_used_kv_pages": peak_used_pages,
            "peak_kv_utilization": peak_used_pages / num_data_pages,
            "preempted_request_count": sum(
                item.preempted_request_count for item in repetitions
            ),
        },
        "repetitions": [
            _repetition_document(index, repetition)
            for index, repetition in enumerate(repetitions)
        ],
    }


def _make_config(workload: WorkloadSpec) -> LLMConfig:
    return LLMConfig(
        model_config=ModelConfig(
            model=MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            attention_operator=ATTENTION_OPERATOR,
            device="cuda",
            dtype=DTYPE,
        ),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=workload.max_num_batched_tokens,
            max_num_seqs=workload.max_num_seqs,
            max_model_len=workload.max_model_len,
        ),
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.9,
            num_blocks=workload.num_blocks,
            enable_prefix_cache=workload.enable_prefix_cache,
        ),
    )


def run_benchmark(
    workload: WorkloadSpec,
    *,
    warmup_repetitions: int,
    measurement_repetitions: int,
) -> dict[str, Any]:
    """Initialize the pinned model and return one JSON-ready result document."""

    if warmup_repetitions < 0:
        raise ValueError("warmup_repetitions must be non-negative")
    if measurement_repetitions <= 0:
        raise ValueError("measurement_repetitions must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("the V1 reference benchmark requires a CUDA GPU")

    device = torch.device("cuda", torch.cuda.current_device())
    gpu_name = torch.cuda.get_device_name(device)
    if gpu_name != "Tesla T4":
        raise RuntimeError(
            f"the frozen V1 baseline requires Tesla T4, found {gpu_name!r}"
        )

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    config = _make_config(workload)

    worker = Worker(config)
    try:
        for _ in range(warmup_repetitions):
            _run_repetition(
                worker,
                config,
                workload,
                device,
                measure_memory=False,
            )
        repetitions = tuple(
            _run_repetition(
                worker,
                config,
                workload,
                device,
                measure_memory=True,
            )
            for _ in range(measurement_repetitions)
        )
        return _build_document(
            config,
            worker,
            workload,
            warmup_repetitions,
            repetitions,
            device,
        )
    finally:
        worker.shutdown()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True, choices=tuple(WORKLOADS))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup-repetitions", type=int, default=2)
    parser.add_argument("--measurement-repetitions", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_benchmark(
        WORKLOADS[args.workload],
        warmup_repetitions=args.warmup_repetitions,
        measurement_repetitions=args.measurement_repetitions,
    )
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = result["summary"]
    print(
        f"{args.workload}: {summary['elapsed_seconds']:.6f}s, "
        f"{summary['output_token_throughput_per_second']:.3f} output tokens/s, "
        f"wrote {output_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
