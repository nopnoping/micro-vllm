"""GPU checks for the stepwise Triton paged-attention spike."""

import pytest
import torch


pytest.importorskip("triton")

from benchmarks.spike_triton_paged import (
    triton_gather_kv_cache,
    triton_write_kv_cache,
)
from micro_vllm.attention.attention import AttentionMetadata
from micro_vllm.attention.operators.reference import write_kv_cache


def _write_metadata(
    slot_mapping: torch.Tensor,
    *,
    block_size: int,
) -> AttentionMetadata:
    num_tokens = slot_mapping.numel()
    return AttentionMetadata(
        query_positions=torch.arange(
            num_tokens,
            dtype=torch.int64,
            device=slot_mapping.device,
        ),
        query_context_start_positions=torch.zeros(
            num_tokens,
            dtype=torch.int64,
            device=slot_mapping.device,
        ),
        query_start_locations=torch.tensor(
            (0, num_tokens - 1, num_tokens),
            dtype=torch.int64,
            device=slot_mapping.device,
        ),
        context_lengths=torch.tensor(
            (num_tokens - 1, 1),
            dtype=torch.int64,
            device=slot_mapping.device,
        ),
        block_table=torch.zeros(
            (2, 2),
            dtype=torch.int64,
            device=slot_mapping.device,
        ),
        slot_mapping=slot_mapping,
        block_size=block_size,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("num_kv_heads", "head_dim"),
    [
        pytest.param(2, 7, id="masked-head-dimension"),
        pytest.param(8, 128, id="qwen3-shape"),
    ],
)
def test_triton_page_write_matches_reference_across_physical_pages(
    num_kv_heads: int,
    head_dim: int,
) -> None:
    device = torch.device("cuda")
    block_size = 4
    slot_mapping = torch.tensor(
        (7, 2, 8),
        dtype=torch.int64,
        device=device,
    )
    metadata = _write_metadata(slot_mapping, block_size=block_size)
    key_storage = torch.randn(
        (3, num_kv_heads, head_dim * 2),
        dtype=torch.float16,
        device=device,
    )
    key = key_storage[..., ::2]
    value = torch.randn(
        (3, num_kv_heads, head_dim),
        dtype=torch.float16,
        device=device,
    )
    initial_cache = torch.randn(
        (2, 3, block_size, num_kv_heads, head_dim),
        dtype=torch.float16,
        device=device,
    )
    reference_cache = initial_cache.clone()
    triton_cache = initial_cache.clone()

    write_kv_cache(key, value, reference_cache, metadata)
    triton_write_kv_cache(key, value, triton_cache, metadata)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(triton_cache, reference_cache, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("num_kv_heads", "head_dim"),
    [
        pytest.param(2, 7, id="masked-head-dimension"),
        pytest.param(8, 128, id="qwen3-shape"),
    ],
)
def test_triton_page_gather_matches_ragged_multi_request_reference(
    num_kv_heads: int,
    head_dim: int,
) -> None:
    from micro_vllm.attention.operators.reference import gather_kv_cache

    device = torch.device("cuda")
    block_size = 2
    layer_kv_cache = torch.arange(
        2 * 6 * block_size * num_kv_heads * head_dim,
        dtype=torch.float32,
        device=device,
    ).reshape(2, 6, block_size, num_kv_heads, head_dim).to(torch.float16)
    metadata = AttentionMetadata(
        query_positions=torch.tensor(
            (3, 4, 2),
            dtype=torch.int64,
            device=device,
        ),
        query_context_start_positions=torch.tensor(
            (1, 1, 0),
            dtype=torch.int64,
            device=device,
        ),
        query_start_locations=torch.tensor(
            (0, 2, 3),
            dtype=torch.int64,
            device=device,
        ),
        context_lengths=torch.tensor(
            (5, 3),
            dtype=torch.int64,
            device=device,
        ),
        block_table=torch.tensor(
            (
                (3, 1, 5),
                (4, 0, -1),
            ),
            dtype=torch.int64,
            device=device,
        ),
        slot_mapping=torch.tensor(
            (3, 10, 0),
            dtype=torch.int64,
            device=device,
        ),
        block_size=block_size,
    )

    reference = gather_kv_cache(layer_kv_cache, metadata)
    actual_key, actual_value = triton_gather_kv_cache(
        layer_kv_cache,
        metadata,
    )
    torch.cuda.synchronize(device)

    assert actual_key.shape == (2, 6, num_kv_heads, head_dim)
    assert actual_value.shape == (2, 6, num_kv_heads, head_dim)
    for request_index, (expected_key, expected_value) in enumerate(reference):
        context_tokens = expected_key.shape[0]
        torch.testing.assert_close(
            actual_key[request_index, :context_tokens],
            expected_key,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual_value[request_index, :context_tokens],
            expected_value,
            rtol=0,
            atol=0,
        )
