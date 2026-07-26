"""Semantic parity tests for the registered Triton paged operator."""

from collections.abc import Sequence

import pytest
import torch


pytest.importorskip("triton")

from micro_vllm.attention.attention import Attention, map_cache_slots
from micro_vllm.attention.spec import (
    AttentionSpec,
    ChunkedLocalSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
)


BLOCK_SIZE = 3
BLOCK_TABLE = (6, 1, 4)
NUM_BLOCKS = 7
NUM_QUERY_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 16
ATOL = 5e-2
RTOL = 5e-2


def _make_inputs(
    num_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.random.fork_rng(devices=(torch.cuda.current_device(),)):
        torch.manual_seed(0)
        query = torch.randn(
            (num_tokens, NUM_QUERY_HEADS, HEAD_DIM),
            dtype=torch.float16,
            device=device,
        )
        key = torch.randn(
            (num_tokens, NUM_KV_HEADS, HEAD_DIM),
            dtype=torch.float16,
            device=device,
        )
        value = torch.randn(
            (num_tokens, NUM_KV_HEADS, HEAD_DIM),
            dtype=torch.float16,
            device=device,
        )
    cache = torch.full(
        (
            2,
            NUM_BLOCKS,
            BLOCK_SIZE,
            NUM_KV_HEADS,
            HEAD_DIM,
        ),
        torch.nan,
        dtype=torch.float16,
        device=device,
    )
    return query, key, value, cache


def _run_chunks(
    attention: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cache: torch.Tensor,
    chunks: Sequence[tuple[int, int]],
) -> torch.Tensor:
    block_table = torch.tensor(
        (BLOCK_TABLE,),
        dtype=torch.int64,
        device=query.device,
    )
    outputs = []
    for start, end in chunks:
        positions = torch.arange(
            start,
            end,
            dtype=torch.int64,
            device=query.device,
        )
        metadata = attention.build_metadata(
            query_positions=positions,
            query_start_locations=torch.tensor(
                (0, end - start),
                dtype=torch.int64,
                device=query.device,
            ),
            context_lengths=torch.tensor(
                (end,),
                dtype=torch.int64,
                device=query.device,
            ),
            block_table=block_table,
            slot_mapping=map_cache_slots(
                positions,
                block_table[0],
                BLOCK_SIZE,
            ),
            block_size=BLOCK_SIZE,
        )
        outputs.append(
            attention(
                query[start:end],
                key[start:end],
                value[start:end],
                cache,
                metadata,
            )
        )
    return torch.cat(outputs)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_full_prefill_and_chunked_decode_match_reference() -> None:
    device = torch.device("cuda")
    query, key, value, initial_cache = _make_inputs(8, device)
    full_chunks = ((0, 8),)
    incremental_chunks = ((0, 3), (3, 6), (6, 7), (7, 8))

    reference_cache = initial_cache.clone()
    expected = _run_chunks(
        Attention(FullAttentionSpec(), "reference"),
        query,
        key,
        value,
        reference_cache,
        full_chunks,
    )
    full_cache = initial_cache.clone()
    full_output = _run_chunks(
        Attention(FullAttentionSpec(), "triton_paged"),
        query,
        key,
        value,
        full_cache,
        full_chunks,
    )
    incremental_cache = initial_cache.clone()
    incremental_output = _run_chunks(
        Attention(FullAttentionSpec(), "triton_paged"),
        query,
        key,
        value,
        incremental_cache,
        incremental_chunks,
    )

    torch.testing.assert_close(full_output, expected, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(
        incremental_output,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )
    torch.testing.assert_close(
        full_cache,
        reference_cache,
        atol=0,
        rtol=0,
        equal_nan=True,
    )
    torch.testing.assert_close(
        incremental_cache,
        reference_cache,
        atol=0,
        rtol=0,
        equal_nan=True,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    "attention_spec",
    [
        pytest.param(SlidingWindowSpec(3), id="sliding-window"),
        pytest.param(ChunkedLocalSpec(4), id="chunked-local"),
    ],
)
def test_bounded_chunked_prefill_and_decode_match_reference(
    attention_spec: AttentionSpec,
) -> None:
    device = torch.device("cuda")
    query, key, value, initial_cache = _make_inputs(8, device)
    chunks = ((0, 3), (3, 6), (6, 7), (7, 8))

    reference_cache = initial_cache.clone()
    expected = _run_chunks(
        Attention(attention_spec, "reference"),
        query,
        key,
        value,
        reference_cache,
        chunks,
    )
    triton_cache = initial_cache.clone()
    actual = _run_chunks(
        Attention(attention_spec, "triton_paged"),
        query,
        key,
        value,
        triton_cache,
        chunks,
    )

    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(
        triton_cache,
        reference_cache,
        atol=0,
        rtol=0,
        equal_nan=True,
    )
