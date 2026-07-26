"""Device-independent checks for the registered SDPA transition operator."""

import math

import pytest
import torch

from benchmarks.spike_sdpa import (
    SPIKE_SHAPES,
    SpikeShape,
    _latency_summary,
    _make_case,
    _validate_target_gpu,
    _visible_mask,
)
from micro_vllm.attention.operators.reference import ReferenceAttentionOperator
from micro_vllm.attention.operators.sdpa import SDPAAttentionOperator


def test_spike_shapes_match_frozen_qwen_prefill_and_decode() -> None:
    assert [
        (shape.name, shape.query_length, shape.context_length)
        for shape in SPIKE_SHAPES
    ] == [
        ("prefill", 512, 512),
        ("decode", 1, 512),
    ]
    assert all(shape.num_attention_heads == 16 for shape in SPIKE_SHAPES)
    assert all(shape.num_key_value_heads == 8 for shape in SPIKE_SHAPES)
    assert all(shape.head_dim == 128 for shape in SPIKE_SHAPES)


def test_visible_mask_combines_context_start_and_causal_bound() -> None:
    mask = _visible_mask(
        query_positions=torch.tensor((2, 3)),
        query_context_starts=torch.tensor((1, 0)),
        context_positions=torch.arange(4),
    )

    assert mask.tolist() == [
        [False, True, True, False],
        [True, True, True, True],
    ]


def test_paged_sdpa_matches_reference_for_tiny_gqa_cases() -> None:
    shapes = (
        SpikeShape(
            name="prefill",
            query_length=4,
            context_length=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            block_size=2,
        ),
        SpikeShape(
            name="decode",
            query_length=1,
            context_length=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            block_size=2,
        ),
    )
    for index, shape in enumerate(shapes):
        case = _make_case(
            shape,
            device="cpu",
            dtype=torch.float32,
            seed=index,
        )
        reference = ReferenceAttentionOperator()(
            case.query,
            case.key,
            case.value,
            case.layer_kv_cache.clone(),
            case.metadata,
        )
        actual = SDPAAttentionOperator()(
            case.query,
            case.key,
            case.value,
            case.layer_kv_cache.clone(),
            case.metadata,
        )
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)


def test_latency_summary_uses_all_samples() -> None:
    summary = _latency_summary((0.1, 0.2, 0.3))

    assert summary["count"] == 3
    assert math.isclose(summary["mean_seconds"], 0.2)
    assert summary["median_seconds"] == 0.2
    assert math.isclose(summary["p90_seconds"], 0.28)
    assert summary["min_seconds"] == 0.1
    assert summary["max_seconds"] == 0.3


def test_target_gpu_check_is_optional_but_rejects_mismatch() -> None:
    _validate_target_gpu("NVIDIA GeForce RTX 5090", None)
    _validate_target_gpu(
        "NVIDIA GeForce RTX 5090",
        "NVIDIA GeForce RTX 5090",
    )

    with pytest.raises(RuntimeError, match="expected GPU 'Tesla T4'"):
        _validate_target_gpu("NVIDIA GeForce RTX 5090", "Tesla T4")
