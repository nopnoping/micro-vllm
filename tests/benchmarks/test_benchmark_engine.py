"""Device-independent checks for the frozen Engine benchmark contract."""

import math

from benchmarks.benchmark_engine import (
    BLOCK_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    SCHEMA_VERSION,
    WORKLOADS,
    RequestMeasurement,
    _RequestClock,
    _latency_summary,
    _request_document,
)
from micro_vllm.engine.output import RequestOutput


def test_frozen_workloads_cover_phase_zero_scenarios() -> None:
    assert set(WORKLOADS) == {
        "prefill-16",
        "prefill-128",
        "prefill-512",
        "decode",
        "mixed",
        "cache-pressure",
    }
    assert MODEL_ID == "Qwen/Qwen3-0.6B"
    assert MODEL_REVISION == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert SCHEMA_VERSION == "micro-vllm.engine-benchmark.v1"
    assert BLOCK_SIZE == 16

    assert [
        len(WORKLOADS[name].requests[0].prompt_token_ids)
        for name in ("prefill-16", "prefill-128", "prefill-512")
    ] == [16, 128, 512]
    assert WORKLOADS["decode"].requests[0].max_tokens == 128
    assert WORKLOADS["mixed"].requests[1].admit_after_steps == 1
    cache_pressure = WORKLOADS["cache-pressure"]
    num_data_pages = cache_pressure.num_blocks - 1
    pages_per_request = [
        math.ceil(
            (len(request.prompt_token_ids) + request.max_tokens) / BLOCK_SIZE
        )
        for request in cache_pressure.requests
    ]
    assert all(page_count <= num_data_pages for page_count in pages_per_request)
    assert sum(pages_per_request) > num_data_pages


def test_latency_summary_uses_seconds_and_all_samples() -> None:
    assert _latency_summary(()) == {
        "count": 0,
        "median_seconds": None,
        "p90_seconds": None,
    }
    assert _latency_summary((0.1, 0.2, 0.3)) == {
        "count": 3,
        "median_seconds": 0.2,
        "p90_seconds": 0.28,
    }


def test_request_document_keeps_raw_token_timings() -> None:
    document = _request_document(
        RequestMeasurement(
            request_id="decode",
            prompt_tokens=8,
            output_tokens=3,
            ttft_seconds=0.5,
            itl_seconds=(0.1, 0.2),
            finished=True,
        )
    )

    assert document == {
        "request_id": "decode",
        "prompt_tokens": 8,
        "output_tokens": 3,
        "ttft_seconds": 0.5,
        "itl_seconds": [0.1, 0.2],
        "finished": True,
    }


def test_request_clock_derives_ttft_and_itl_from_cumulative_outputs() -> None:
    clock = _RequestClock(admitted_at_seconds=0.2, prompt_tokens=8)
    clock.observe(
        RequestOutput("decode", (11,), finished=False),
        now_seconds=0.7,
    )
    clock.observe(
        RequestOutput("decode", (11, 12), finished=True),
        now_seconds=0.9,
    )

    measurement = clock.snapshot("decode")
    assert math.isclose(measurement.ttft_seconds or 0.0, 0.5)
    assert len(measurement.itl_seconds) == 1
    assert math.isclose(measurement.itl_seconds[0], 0.2)
    assert measurement.finished is True
