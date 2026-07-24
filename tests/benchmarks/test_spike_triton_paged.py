"""GPU checks for the stepwise Triton paged-attention spike."""

import pytest
import torch


pytest.importorskip("triton")

from benchmarks.spike_triton_paged import triton_write_kv_cache
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
