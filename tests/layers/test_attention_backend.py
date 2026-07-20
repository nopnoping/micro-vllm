"""Tests for the attention backend contract and cache helpers."""

import math

import pytest
import torch

from micro_vllm.layers.attention_backend import (
    AttentionBackend,
    AttentionMetadata,
    PAD_BLOCK_ID,
    ReferenceAttentionBackend,
    gather_kv_cache,
    write_kv_cache,
)


class RecordingBackend:
    def __init__(self) -> None:
        self.arguments: tuple[object, ...] | None = None

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_kv_cache: torch.Tensor,
        metadata: AttentionMetadata,
        /,
    ) -> torch.Tensor:
        self.arguments = (query, key, value, layer_kv_cache, metadata)
        return torch.empty_like(query)


def invoke_backend(
    backend: AttentionBackend,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_kv_cache: torch.Tensor,
    metadata: AttentionMetadata,
) -> torch.Tensor:
    return backend(query, key, value, layer_kv_cache, metadata)


def test_attention_backend_receives_only_tensor_data_and_metadata() -> None:
    query = torch.empty((2, 4, 16))
    key = torch.empty((2, 2, 16))
    value = torch.empty((2, 2, 16))
    layer_kv_cache = torch.empty((2, 8, 4, 2, 16))
    metadata = AttentionMetadata(
        query_positions=torch.tensor([0, 1], dtype=torch.int64),
        query_start_locations=torch.tensor([0, 2], dtype=torch.int64),
        context_lengths=torch.tensor([2], dtype=torch.int64),
        block_table=torch.tensor([[0]], dtype=torch.int64),
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
        block_size=4,
    )
    backend = RecordingBackend()

    output = invoke_backend(
        backend,
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
    assert backend.arguments is not None
    assert all(
        actual is expected
        for actual, expected in zip(backend.arguments, expected_arguments)
    )
    assert output.shape == query.shape
    assert output.dtype is query.dtype
    assert output.device == query.device


def make_write_metadata(slot_mapping: list[int]) -> AttentionMetadata:
    num_tokens = len(slot_mapping)
    return AttentionMetadata(
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
    metadata = AttentionMetadata(
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


def test_kv_gather_handles_each_request_and_query_independently() -> None:
    layer_kv_cache = torch.full((2, 5, 2, 1, 1), 99.0)
    flattened_cache = layer_kv_cache.view(2, 10, 1, 1)
    flattened_cache[0, [6, 7, 2, 8]] = torch.tensor(
        [[[10.0]], [[11.0]], [[12.0]], [[20.0]]]
    )
    flattened_cache[1, [6, 7, 2, 8]] = torch.tensor(
        [[[110.0]], [[111.0]], [[112.0]], [[120.0]]]
    )
    metadata = AttentionMetadata(
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
        [10.0, 11.0],
        [10.0, 11.0, 12.0],
        [20.0],
    ]
    assert [values.flatten().tolist() for _, values in histories] == [
        [110.0, 111.0],
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
    metadata = AttentionMetadata(
        query_positions=torch.tensor([2, 3], dtype=torch.int64),
        query_start_locations=torch.tensor([0, 2], dtype=torch.int64),
        context_lengths=torch.tensor([4], dtype=torch.int64),
        block_table=torch.tensor([[2, 5]], dtype=torch.int64),
        slot_mapping=torch.tensor([10, 11], dtype=torch.int64),
        block_size=2,
    )

    write_kv_cache(key, value, layer_kv_cache, metadata)
    histories = gather_kv_cache(layer_kv_cache, metadata)

    assert [keys.flatten().tolist() for keys, _ in histories] == [
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0, 4.0],
    ]
    assert [values.flatten().tolist() for _, values in histories] == [
        [11.0, 12.0, 13.0],
        [11.0, 12.0, 13.0, 14.0],
    ]


def test_kv_gather_decode_reads_the_complete_context() -> None:
    layer_kv_cache = torch.zeros((2, 6, 2, 1, 1))
    flattened_cache = layer_kv_cache.view(2, 12, 1, 1)
    flattened_cache[0, [4, 5, 10, 11]] = torch.tensor(
        [[[1.0]], [[2.0]], [[3.0]], [[4.0]]]
    )
    flattened_cache[1, [4, 5, 10, 11]] = torch.tensor(
        [[[11.0]], [[12.0]], [[13.0]], [[14.0]]]
    )
    metadata = AttentionMetadata(
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
    outputs = []
    scale = 1.0 / math.sqrt(query.shape[-1])
    for position, query_row in enumerate(query):
        scores = torch.einsum(
            "hd,thd->ht",
            query_row.to(torch.float32),
            key[: position + 1],
        )
        weights = torch.softmax(scores * scale, dim=-1)
        outputs.append(
            torch.einsum("ht,thd->hd", weights, value[: position + 1])
        )
    return torch.stack(outputs).to(query.dtype)


@pytest.mark.parametrize("num_key_value_heads", [4, 2, 1])
def test_reference_attention_matches_dense_mha_gqa_and_mqa(
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
    metadata = AttentionMetadata(
        query_positions=torch.tensor([0, 1, 2], dtype=torch.int64),
        query_start_locations=torch.tensor([0, 3], dtype=torch.int64),
        context_lengths=torch.tensor([3], dtype=torch.int64),
        block_table=torch.tensor([[2, 0]], dtype=torch.int64),
        slot_mapping=torch.tensor([4, 5, 0], dtype=torch.int64),
        block_size=2,
    )

    output = ReferenceAttentionBackend()(
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
    metadata = AttentionMetadata(
        query_positions=torch.tensor([0], dtype=torch.int64),
        query_start_locations=torch.tensor([0, 1], dtype=torch.int64),
        context_lengths=torch.tensor([1], dtype=torch.int64),
        block_table=torch.tensor([[0]], dtype=torch.int64),
        slot_mapping=torch.tensor([0], dtype=torch.int64),
        block_size=2,
    )

    output = ReferenceAttentionBackend()(
        query,
        key,
        value,
        layer_kv_cache,
        metadata,
    )

    assert output.dtype is torch.float16
    assert torch.equal(output, value)


def test_reference_attention_rejects_incompatible_head_counts() -> None:
    query = torch.empty((1, 3, 2))
    key = torch.empty((1, 2, 2))
    value = torch.empty_like(key)
    layer_kv_cache = torch.empty((2, 1, 2, 2, 2))
    metadata = AttentionMetadata(
        query_positions=torch.tensor([0], dtype=torch.int64),
        query_start_locations=torch.tensor([0, 1], dtype=torch.int64),
        context_lengths=torch.tensor([1], dtype=torch.int64),
        block_table=torch.tensor([[0]], dtype=torch.int64),
        slot_mapping=torch.tensor([0], dtype=torch.int64),
        block_size=2,
    )

    with pytest.raises(
        ValueError,
        match="num_attention_heads must be divisible by num_key_value_heads",
    ):
        ReferenceAttentionBackend()(
            query,
            key,
            value,
            layer_kv_cache,
            metadata,
        )
