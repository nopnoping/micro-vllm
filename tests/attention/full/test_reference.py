"""Tests for the reference full-attention operator and cache helpers."""

import math

import pytest
import torch

from micro_vllm.attention.base import PAD_BLOCK_ID
from micro_vllm.attention.full.metadata import FullAttentionMetadata
from micro_vllm.attention.full.operator import FullAttentionOperator
from micro_vllm.attention.full.reference import (
    ReferenceFullAttentionOperator,
    gather_kv_cache,
    write_kv_cache,
)


class RecordingOperator(FullAttentionOperator):
    def __init__(self) -> None:
        self.arguments: tuple[object, ...] | None = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_kv_cache: torch.Tensor,
        metadata: FullAttentionMetadata,
        /,
    ) -> torch.Tensor:
        self.arguments = (query, key, value, layer_kv_cache, metadata)
        return torch.empty_like(query)


def invoke_operator(
    operator: FullAttentionOperator,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_kv_cache: torch.Tensor,
    metadata: FullAttentionMetadata,
) -> torch.Tensor:
    return operator(query, key, value, layer_kv_cache, metadata)


def test_full_attention_operator_receives_tensor_data_and_metadata() -> None:
    query = torch.empty((2, 4, 16))
    key = torch.empty((2, 2, 16))
    value = torch.empty((2, 2, 16))
    layer_kv_cache = torch.empty((2, 8, 4, 2, 16))
    metadata = FullAttentionMetadata(
        query_positions=torch.tensor([0, 1], dtype=torch.int64),
        query_start_locations=torch.tensor([0, 2], dtype=torch.int64),
        context_lengths=torch.tensor([2], dtype=torch.int64),
        block_table=torch.tensor([[0]], dtype=torch.int64),
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
        block_size=4,
    )
    operator = RecordingOperator()

    output = invoke_operator(
        operator,
        query,
        key,
        value,
        layer_kv_cache,
        metadata,
    )

    expected_arguments = (
        query,
        key,
        value,
        layer_kv_cache,
        metadata,
    )
    assert operator.arguments is not None
    assert all(
        actual is expected
        for actual, expected in zip(operator.arguments, expected_arguments)
    )
    assert output.shape == query.shape
    assert output.dtype is query.dtype
    assert output.device == query.device


def make_write_metadata(slot_mapping: list[int]) -> FullAttentionMetadata:
    num_tokens = len(slot_mapping)
    return FullAttentionMetadata(
        query_positions=torch.arange(num_tokens, dtype=torch.int64),
        query_start_locations=torch.tensor(
            [0, num_tokens],
            dtype=torch.int64,
        ),
        context_lengths=torch.tensor([num_tokens], dtype=torch.int64),
        block_table=torch.tensor([[0, 1]], dtype=torch.int64),
        slot_mapping=torch.tensor(slot_mapping, dtype=torch.int64),
        block_size=4,
    )


def test_kv_write_uses_flattened_physical_slots() -> None:
    layer_kv_cache = torch.zeros((2, 2, 4, 2, 2))
    key = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    value = key + 10
    metadata = make_write_metadata([1, 6])

    write_kv_cache(key, value, layer_kv_cache, metadata)

    flattened_cache = layer_kv_cache.view(2, 8, 2, 2)
    assert torch.equal(flattened_cache[0, 1], key[0])
    assert torch.equal(flattened_cache[0, 6], key[1])
    assert torch.equal(flattened_cache[1, 1], value[0])
    assert torch.equal(flattened_cache[1, 6], value[1])
    assert torch.count_nonzero(
        flattened_cache[:, [0, 2, 3, 4, 5, 7]]
    ).item() == 0


def test_kv_write_does_not_overwrite_shared_prefix_slots() -> None:
    layer_kv_cache = torch.zeros((2, 3, 4, 1, 2))
    layer_kv_cache[:, 0].fill_(9)
    key = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    value = key + 10
    metadata = FullAttentionMetadata(
        query_positions=torch.tensor([4, 5], dtype=torch.int64),
        query_start_locations=torch.tensor([0, 2], dtype=torch.int64),
        context_lengths=torch.tensor([6], dtype=torch.int64),
        block_table=torch.tensor([[0, 2]], dtype=torch.int64),
        slot_mapping=torch.tensor([8, 9], dtype=torch.int64),
        block_size=4,
    )

    write_kv_cache(key, value, layer_kv_cache, metadata)

    assert torch.all(layer_kv_cache[:, 0] == 9)
    flattened_cache = layer_kv_cache.view(2, 12, 1, 2)
    assert torch.equal(flattened_cache[0, 8:10], key)
    assert torch.equal(flattened_cache[1, 8:10], value)


def test_kv_gather_handles_each_request_independently() -> None:
    layer_kv_cache = torch.full((2, 5, 2, 1, 1), 99.0)
    flattened_cache = layer_kv_cache.view(2, 10, 1, 1)
    flattened_cache[0, [6, 7, 2, 8]] = torch.tensor(
        [[[10.0]], [[11.0]], [[12.0]], [[20.0]]]
    )
    flattened_cache[1, [6, 7, 2, 8]] = torch.tensor(
        [[[110.0]], [[111.0]], [[112.0]], [[120.0]]]
    )
    metadata = FullAttentionMetadata(
        query_positions=torch.tensor([1, 2, 0], dtype=torch.int64),
        query_start_locations=torch.tensor([0, 2, 3], dtype=torch.int64),
        context_lengths=torch.tensor([3, 1], dtype=torch.int64),
        block_table=torch.tensor(
            [[3, 1, PAD_BLOCK_ID], [4, PAD_BLOCK_ID, PAD_BLOCK_ID]],
            dtype=torch.int64,
        ),
        slot_mapping=torch.tensor([7, 2, 8], dtype=torch.int64),
        block_size=2,
    )

    histories = gather_kv_cache(layer_kv_cache, metadata)

    assert [keys.flatten().tolist() for keys, _ in histories] == [
        [10.0, 11.0, 12.0],
        [20.0],
    ]
    assert [values.flatten().tolist() for _, values in histories] == [
        [110.0, 111.0, 112.0],
        [120.0],
    ]


def test_kv_gather_sees_cached_history_and_current_prefill_chunk() -> None:
    layer_kv_cache = torch.zeros((2, 6, 2, 1, 1))
    flattened_cache = layer_kv_cache.view(2, 12, 1, 1)
    flattened_cache[0, 4:6] = torch.tensor([[[1.0]], [[2.0]]])
    flattened_cache[1, 4:6] = torch.tensor([[[11.0]], [[12.0]]])
    key = torch.tensor([[[3.0]], [[4.0]]])
    value = torch.tensor([[[13.0]], [[14.0]]])
    metadata = FullAttentionMetadata(
        query_positions=torch.tensor([2, 3], dtype=torch.int64),
        query_start_locations=torch.tensor([0, 2], dtype=torch.int64),
        context_lengths=torch.tensor([4], dtype=torch.int64),
        block_table=torch.tensor([[2, 5]], dtype=torch.int64),
        slot_mapping=torch.tensor([10, 11], dtype=torch.int64),
        block_size=2,
    )

    write_kv_cache(key, value, layer_kv_cache, metadata)
    ((gathered_key, gathered_value),) = gather_kv_cache(
        layer_kv_cache,
        metadata,
    )

    assert gathered_key.flatten().tolist() == [1.0, 2.0, 3.0, 4.0]
    assert gathered_value.flatten().tolist() == [11.0, 12.0, 13.0, 14.0]


def test_kv_gather_decode_reads_the_complete_context() -> None:
    layer_kv_cache = torch.zeros((2, 6, 2, 1, 1))
    flattened_cache = layer_kv_cache.view(2, 12, 1, 1)
    flattened_cache[0, [4, 5, 10, 11]] = torch.tensor(
        [[[1.0]], [[2.0]], [[3.0]], [[4.0]]]
    )
    flattened_cache[1, [4, 5, 10, 11]] = torch.tensor(
        [[[11.0]], [[12.0]], [[13.0]], [[14.0]]]
    )
    metadata = FullAttentionMetadata(
        query_positions=torch.tensor([4], dtype=torch.int64),
        query_start_locations=torch.tensor([0, 1], dtype=torch.int64),
        context_lengths=torch.tensor([5], dtype=torch.int64),
        block_table=torch.tensor([[2, 5, 1]], dtype=torch.int64),
        slot_mapping=torch.tensor([2], dtype=torch.int64),
        block_size=2,
    )
    key = torch.tensor([[[5.0]]])
    value = torch.tensor([[[15.0]]])

    write_kv_cache(key, value, layer_kv_cache, metadata)
    ((gathered_key, gathered_value),) = gather_kv_cache(
        layer_kv_cache,
        metadata,
    )

    assert gathered_key.flatten().tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert gathered_value.flatten().tolist() == [
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
    ]


def dense_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    head_group_size = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(head_group_size, dim=1).to(torch.float32)
    value = value.repeat_interleave(head_group_size, dim=1).to(torch.float32)
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.einsum(
        "qhd,khd->hqk",
        query.to(torch.float32),
        key,
    )
    causal_mask = torch.ones(
        query.shape[0],
        key.shape[0],
        dtype=torch.bool,
        device=query.device,
    ).triu(diagonal=1)
    scores.masked_fill_(causal_mask.unsqueeze(0), -torch.inf)
    weights = torch.softmax(scores * scale, dim=-1)
    return torch.einsum("hqk,khd->qhd", weights, value).to(query.dtype)


def make_attention_metadata(
    query_positions: list[list[int]],
    block_tables: list[list[int]],
    *,
    block_size: int,
    device: torch.device | str = "cpu",
) -> FullAttentionMetadata:
    flattened_positions = [
        position
        for request_positions in query_positions
        for position in request_positions
    ]
    query_start_locations = [0]
    for request_positions in query_positions:
        query_start_locations.append(
            query_start_locations[-1] + len(request_positions)
        )

    max_num_blocks = max(len(block_table) for block_table in block_tables)
    padded_block_tables = [
        block_table + [PAD_BLOCK_ID] * (max_num_blocks - len(block_table))
        for block_table in block_tables
    ]
    slot_mapping = [
        block_table[position // block_size] * block_size
        + position % block_size
        for request_positions, block_table in zip(
            query_positions,
            block_tables,
        )
        for position in request_positions
    ]
    return FullAttentionMetadata(
        query_positions=torch.tensor(
            flattened_positions,
            dtype=torch.int64,
            device=device,
        ),
        query_start_locations=torch.tensor(
            query_start_locations,
            dtype=torch.int64,
            device=device,
        ),
        context_lengths=torch.tensor(
            [positions[-1] + 1 for positions in query_positions],
            dtype=torch.int64,
            device=device,
        ),
        block_table=torch.tensor(
            padded_block_tables,
            dtype=torch.int64,
            device=device,
        ),
        slot_mapping=torch.tensor(
            slot_mapping,
            dtype=torch.int64,
            device=device,
        ),
        block_size=block_size,
    )


def make_attention_tensors(
    num_tokens: int,
    *,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    head_dim: int = 3,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.linspace(
        -0.8,
        0.9,
        steps=num_tokens * num_attention_heads * head_dim,
        device=device,
    ).reshape(num_tokens, num_attention_heads, head_dim)
    key = torch.linspace(
        0.7,
        -0.6,
        steps=num_tokens * num_key_value_heads * head_dim,
        device=device,
    ).reshape(num_tokens, num_key_value_heads, head_dim)
    value = torch.linspace(
        -1.0,
        1.0,
        steps=num_tokens * num_key_value_heads * head_dim,
        device=device,
    ).reshape(num_tokens, num_key_value_heads, head_dim)
    return query.to(dtype), key.to(dtype), value.to(dtype)


@pytest.mark.parametrize("num_key_value_heads", [4, 2, 1])
def test_full_causal_prefill_matches_dense_mha_gqa_and_mqa(
    num_key_value_heads: int,
) -> None:
    query = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2) / 10
    key = (
        torch.arange(3 * num_key_value_heads * 2, dtype=torch.float32).reshape(
            3,
            num_key_value_heads,
            2,
        )
        / 20
    )
    value = key + 1
    layer_kv_cache = torch.zeros((2, 3, 2, num_key_value_heads, 2))
    metadata = FullAttentionMetadata(
        query_positions=torch.tensor([0, 1, 2], dtype=torch.int64),
        query_start_locations=torch.tensor([0, 3], dtype=torch.int64),
        context_lengths=torch.tensor([3], dtype=torch.int64),
        block_table=torch.tensor([[2, 0]], dtype=torch.int64),
        slot_mapping=torch.tensor([4, 5, 0], dtype=torch.int64),
        block_size=2,
    )

    output = ReferenceFullAttentionOperator()(
        query,
        key,
        value,
        layer_kv_cache,
        metadata,
    )

    expected = dense_causal_attention(query, key, value)
    assert torch.allclose(output, expected, atol=1e-6, rtol=1e-6)
    assert output.dtype is query.dtype


def test_reference_attention_restores_low_precision_dtype() -> None:
    query = torch.tensor([[[1.0, 2.0]]], dtype=torch.float16)
    key = torch.tensor([[[2.0, 1.0]]], dtype=torch.float16)
    value = torch.tensor([[[3.0, 4.0]]], dtype=torch.float16)
    layer_kv_cache = torch.zeros((2, 1, 2, 1, 2), dtype=torch.float16)
    metadata = FullAttentionMetadata(
        query_positions=torch.tensor([0], dtype=torch.int64),
        query_start_locations=torch.tensor([0, 1], dtype=torch.int64),
        context_lengths=torch.tensor([1], dtype=torch.int64),
        block_table=torch.tensor([[0]], dtype=torch.int64),
        slot_mapping=torch.tensor([0], dtype=torch.int64),
        block_size=2,
    )

    output = ReferenceFullAttentionOperator()(
        query,
        key,
        value,
        layer_kv_cache,
        metadata,
    )

    assert output.dtype is torch.float16
    assert torch.equal(output, value)


def test_single_token_self_attention_matches_dense_reference() -> None:
    query, key, value = make_attention_tensors(
        1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    layer_kv_cache = torch.zeros((2, 2, 2, 2, 3))
    metadata = make_attention_metadata(
        [[0]],
        [[1]],
        block_size=2,
    )

    output = ReferenceFullAttentionOperator()(
        query,
        key,
        value,
        layer_kv_cache,
        metadata,
    )

    torch.testing.assert_close(
        output,
        dense_causal_attention(query, key, value),
        atol=1e-6,
        rtol=1e-6,
    )


def test_chunked_prefill_matches_full_prefill_and_dense_reference() -> None:
    query, key, value = make_attention_tensors(6)
    block_table = [4, 1, 3]
    full_cache = torch.zeros((2, 5, 2, 2, 3))
    full_output = ReferenceFullAttentionOperator()(
        query,
        key,
        value,
        full_cache,
        make_attention_metadata(
            [list(range(6))],
            [block_table],
            block_size=2,
        ),
    )

    chunked_cache = torch.zeros_like(full_cache)
    chunked_outputs = []
    for start, end in ((0, 2), (2, 4), (4, 6)):
        chunked_outputs.append(
            ReferenceFullAttentionOperator()(
                query[start:end],
                key[start:end],
                value[start:end],
                chunked_cache,
                make_attention_metadata(
                    [list(range(start, end))],
                    [block_table],
                    block_size=2,
                ),
            )
        )
    chunked_output = torch.cat(chunked_outputs)
    expected = dense_causal_attention(query, key, value)

    torch.testing.assert_close(full_output, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(chunked_output, expected, atol=1e-6, rtol=1e-6)


def test_prefill_then_decode_matches_full_sequence_attention() -> None:
    query, key, value = make_attention_tensors(5)
    block_table = [3, 0, 4]
    full_cache = torch.zeros((2, 5, 2, 2, 3))
    full_output = ReferenceFullAttentionOperator()(
        query,
        key,
        value,
        full_cache,
        make_attention_metadata(
            [list(range(5))],
            [block_table],
            block_size=2,
        ),
    )

    decode_cache = torch.zeros_like(full_cache)
    incremental_outputs = [
        ReferenceFullAttentionOperator()(
            query[:3],
            key[:3],
            value[:3],
            decode_cache,
            make_attention_metadata(
                [[0, 1, 2]],
                [block_table],
                block_size=2,
            ),
        )
    ]
    for position in (3, 4):
        incremental_outputs.append(
            ReferenceFullAttentionOperator()(
                query[position : position + 1],
                key[position : position + 1],
                value[position : position + 1],
                decode_cache,
                make_attention_metadata(
                    [[position]],
                    [block_table],
                    block_size=2,
                ),
            )
        )
    incremental_output = torch.cat(incremental_outputs)
    expected = dense_causal_attention(query, key, value)

    torch.testing.assert_close(full_output, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        incremental_output,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_mixed_prefill_and_decode_match_independent_requests() -> None:
    query_a, key_a, value_a = make_attention_tensors(2)
    query_b, key_b, value_b = make_attention_tensors(3)
    query_b = query_b + 0.3
    key_b = key_b - 0.2
    value_b = value_b + 0.5
    block_table_a = [0]
    block_table_b = [2, 3]

    mixed_cache = torch.zeros((2, 4, 2, 2, 3))
    ReferenceFullAttentionOperator()(
        query_b[:2],
        key_b[:2],
        value_b[:2],
        mixed_cache,
        make_attention_metadata(
            [[0, 1]],
            [block_table_b],
            block_size=2,
        ),
    )
    mixed_output = ReferenceFullAttentionOperator()(
        torch.cat((query_a, query_b[2:])),
        torch.cat((key_a, key_b[2:])),
        torch.cat((value_a, value_b[2:])),
        mixed_cache,
        make_attention_metadata(
            [[0, 1], [2]],
            [block_table_a, block_table_b],
            block_size=2,
        ),
    )

    cache_a = torch.zeros_like(mixed_cache)
    output_a = ReferenceFullAttentionOperator()(
        query_a,
        key_a,
        value_a,
        cache_a,
        make_attention_metadata(
            [[0, 1]],
            [block_table_a],
            block_size=2,
        ),
    )
    cache_b = torch.zeros_like(mixed_cache)
    ReferenceFullAttentionOperator()(
        query_b[:2],
        key_b[:2],
        value_b[:2],
        cache_b,
        make_attention_metadata(
            [[0, 1]],
            [block_table_b],
            block_size=2,
        ),
    )
    output_b = ReferenceFullAttentionOperator()(
        query_b[2:],
        key_b[2:],
        value_b[2:],
        cache_b,
        make_attention_metadata(
            [[2]],
            [block_table_b],
            block_size=2,
        ),
    )

    torch.testing.assert_close(mixed_output[:2], output_a, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(mixed_output[2:], output_b, atol=1e-6, rtol=1e-6)


def test_non_contiguous_physical_blocks_do_not_change_attention() -> None:
    query, key, value = make_attention_tensors(6)
    contiguous_cache = torch.zeros((2, 5, 2, 2, 3))
    contiguous_output = ReferenceFullAttentionOperator()(
        query,
        key,
        value,
        contiguous_cache,
        make_attention_metadata(
            [list(range(6))],
            [[0, 1, 2]],
            block_size=2,
        ),
    )

    non_contiguous_cache = torch.zeros_like(contiguous_cache)
    non_contiguous_output = ReferenceFullAttentionOperator()(
        query,
        key,
        value,
        non_contiguous_cache,
        make_attention_metadata(
            [list(range(6))],
            [[4, 1, 3]],
            block_size=2,
        ),
    )

    torch.testing.assert_close(
        non_contiguous_output,
        contiguous_output,
        atol=1e-6,
        rtol=1e-6,
    )


def test_reused_block_does_not_expose_stale_values_beyond_new_context() -> None:
    old_query, old_key, old_value = make_attention_tensors(
        4,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    layer_kv_cache = torch.zeros((2, 2, 4, 1, 3))
    ReferenceFullAttentionOperator()(
        old_query,
        old_key,
        old_value,
        layer_kv_cache,
        make_attention_metadata(
            [list(range(4))],
            [[0]],
            block_size=4,
        ),
    )
    stale_key = old_key[1].clone()
    stale_value = old_value[1].clone()

    query, key, value = make_attention_tensors(
        1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    query = query + 0.4
    key = key - 0.3
    value = value + 0.8
    output = ReferenceFullAttentionOperator()(
        query,
        key,
        value,
        layer_kv_cache,
        make_attention_metadata(
            [[0]],
            [[0]],
            block_size=4,
        ),
    )

    flattened_cache = layer_kv_cache.view(2, 8, 1, 3)
    assert torch.equal(flattened_cache[0, 1], stale_key)
    assert torch.equal(flattened_cache[1, 1], stale_value)
    torch.testing.assert_close(
        output,
        dense_causal_attention(query, key, value),
        atol=1e-6,
        rtol=1e-6,
    )


def test_shared_prefix_produces_identical_history_attention() -> None:
    prefix_query, prefix_key, prefix_value = make_attention_tensors(
        2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    layer_kv_cache = torch.zeros((2, 3, 2, 1, 3))
    ReferenceFullAttentionOperator()(
        prefix_query,
        prefix_key,
        prefix_value,
        layer_kv_cache,
        make_attention_metadata(
            [[0, 1]],
            [[0]],
            block_size=2,
        ),
    )
    query, key, value = make_attention_tensors(
        1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    query = query + 0.25
    key = key - 0.15
    value = value + 0.35

    output = ReferenceFullAttentionOperator()(
        torch.cat((query, query)),
        torch.cat((key, key)),
        torch.cat((value, value)),
        layer_kv_cache,
        make_attention_metadata(
            [[2], [2]],
            [[0, 1], [0, 2]],
            block_size=2,
        ),
    )
    expected = dense_causal_attention(
        torch.cat((prefix_query, query)),
        torch.cat((prefix_key, key)),
        torch.cat((prefix_value, value)),
    )[-1]

    torch.testing.assert_close(output[0], expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(output[1], expected, atol=1e-6, rtol=1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        pytest.param(torch.float16, 2e-3, 2e-3, id="float16"),
        pytest.param(
            torch.bfloat16,
            2e-2,
            2e-2,
            id="bfloat16",
            marks=pytest.mark.skipif(
                not torch.cuda.is_bf16_supported(),
                reason="CUDA device does not support bfloat16",
            ),
        ),
    ],
)
def test_gpu_low_precision_matches_dense_reference(
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    device = torch.device("cuda")
    query, key, value = make_attention_tensors(
        5,
        head_dim=8,
        dtype=dtype,
        device=device,
    )
    layer_kv_cache = torch.zeros(
        (2, 5, 2, 2, 8),
        dtype=dtype,
        device=device,
    )
    output = ReferenceFullAttentionOperator()(
        query,
        key,
        value,
        layer_kv_cache,
        make_attention_metadata(
            [list(range(5))],
            [[3, 0, 4]],
            block_size=2,
            device=device,
        ),
    )

    torch.testing.assert_close(
        output,
        dense_causal_attention(query, key, value),
        atol=atol,
        rtol=rtol,
    )
