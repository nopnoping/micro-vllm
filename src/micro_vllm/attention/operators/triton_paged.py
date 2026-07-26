"""Triton attention over flattened queries and a paged KV cache."""

import torch
import triton
import triton.language as tl

from micro_vllm.attention.attention import AttentionMetadata
from micro_vllm.attention.operators.base import AttentionOperator


@triton.jit
def _write_kv_cache_kernel(
    key_ptr,
    value_ptr,
    cache_ptr,
    slot_mapping_ptr,
    key_token_stride,
    key_head_stride,
    key_dim_stride,
    value_token_stride,
    value_head_stride,
    value_dim_stride,
    cache_plane_stride,
    cache_block_stride,
    cache_offset_stride,
    cache_head_stride,
    cache_dim_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    token_index = tl.program_id(0)
    kv_head_index = tl.program_id(1)
    dim_offsets = tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offsets < HEAD_DIM

    physical_slot = tl.load(slot_mapping_ptr + token_index)
    valid_slot = physical_slot >= 0
    physical_slot = tl.where(valid_slot, physical_slot, 0)
    physical_block = physical_slot // BLOCK_SIZE
    block_offset = physical_slot % BLOCK_SIZE

    key_offsets = (
        token_index * key_token_stride
        + kv_head_index * key_head_stride
        + dim_offsets * key_dim_stride
    )
    value_offsets = (
        token_index * value_token_stride
        + kv_head_index * value_head_stride
        + dim_offsets * value_dim_stride
    )
    cache_offsets = (
        physical_block * cache_block_stride
        + block_offset * cache_offset_stride
        + kv_head_index * cache_head_stride
        + dim_offsets * cache_dim_stride
    )

    key = tl.load(key_ptr + key_offsets, mask=dim_mask)
    value = tl.load(value_ptr + value_offsets, mask=dim_mask)
    store_mask = valid_slot & dim_mask
    tl.store(cache_ptr + cache_offsets, key, mask=store_mask)
    tl.store(
        cache_ptr + cache_plane_stride + cache_offsets,
        value,
        mask=store_mask,
    )


@triton.jit
def _find_request_for_query_block(
    query_start_locations_ptr,
    query_block_index,
    num_requests,
    BLOCK_Q: tl.constexpr,
):
    """Map one packed query-block program to exactly one request."""

    left: tl.int32 = 0
    right = num_requests
    while left < right:
        middle = (left + right) // 2
        flattened_query_start = tl.load(
            query_start_locations_ptr + middle
        )
        packed_query_start = (
            flattened_query_start // BLOCK_Q + middle
        )
        if packed_query_start <= query_block_index:
            left = middle + 1
        else:
            right = middle
    return left - 1


@triton.jit
def _paged_attention_kernel(
    query_ptr,
    cache_ptr,
    query_positions_ptr,
    query_context_start_positions_ptr,
    query_start_locations_ptr,
    block_table_ptr,
    output_ptr,
    query_token_stride,
    query_head_stride,
    query_dim_stride,
    cache_plane_stride,
    cache_block_stride,
    cache_offset_stride,
    cache_head_stride,
    cache_dim_stride,
    block_table_request_stride,
    block_table_block_stride,
    output_token_stride,
    output_head_stride,
    output_dim_stride,
    scale,
    num_requests,
    HEAD_GROUP_SIZE: tl.constexpr,
    VALID_QUERY_ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    query_block_index = tl.program_id(0)
    kv_head_index = tl.program_id(1)
    request_index = _find_request_for_query_block(
        query_start_locations_ptr,
        query_block_index,
        num_requests,
        BLOCK_Q,
    )
    request_query_start = tl.load(
        query_start_locations_ptr + request_index
    )
    request_query_end = tl.load(
        query_start_locations_ptr + request_index + 1
    )
    request_query_length = request_query_end - request_query_start
    request_query_block_start = (
        request_query_start // BLOCK_Q + request_index
    )
    request_query_block_index = (
        query_block_index - request_query_block_start
    )
    if request_query_block_index * BLOCK_Q >= request_query_length:
        return

    query_row_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offsets < HEAD_DIM
    valid_query_row = query_row_offsets < VALID_QUERY_ROWS
    request_query_offsets = (
        request_query_block_index * BLOCK_Q
        + query_row_offsets // HEAD_GROUP_SIZE
    )
    query_indices = request_query_start + request_query_offsets
    query_head_indices = (
        kv_head_index * HEAD_GROUP_SIZE
        + query_row_offsets % HEAD_GROUP_SIZE
    )
    query_mask = (
        valid_query_row
        & (request_query_offsets < request_query_length)
    )
    query_positions = tl.load(
        query_positions_ptr + query_indices,
        mask=query_mask,
        other=-1,
    )
    context_starts = tl.load(
        query_context_start_positions_ptr + query_indices,
        mask=query_mask,
        other=0,
    )
    query_offsets = (
        query_indices[:, None] * query_token_stride
        + query_head_indices[:, None] * query_head_stride
        + dim_offsets[None, :] * query_dim_stride
    )
    query = tl.load(
        query_ptr + query_offsets,
        mask=query_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    running_max = tl.full(
        (BLOCK_M,),
        -float("inf"),
        dtype=tl.float32,
    )
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_DIM),
        dtype=tl.float32,
    )
    context_offsets = tl.arange(0, BLOCK_N)
    first_context_start = tl.min(
        tl.where(query_mask, context_starts, 2**31 - 1),
        axis=0,
    )
    last_query_position = tl.max(query_positions, axis=0)

    for block_start in tl.range(
        first_context_start,
        last_query_position + 1,
        BLOCK_N,
    ):
        logical_positions = block_start + context_offsets
        context_token_mask = logical_positions <= last_query_position
        attention_mask = (
            query_mask[:, None]
            & (
                logical_positions[None, :]
                >= context_starts[:, None]
            )
            & (
                logical_positions[None, :]
                <= query_positions[:, None]
            )
        )
        logical_blocks = logical_positions // BLOCK_SIZE
        physical_blocks = tl.load(
            block_table_ptr
            + request_index * block_table_request_stride
            + logical_blocks * block_table_block_stride,
            mask=context_token_mask,
            other=0,
        )
        block_offsets = logical_positions % BLOCK_SIZE
        cache_offsets = (
            physical_blocks[:, None] * cache_block_stride
            + block_offsets[:, None] * cache_offset_stride
            + kv_head_index * cache_head_stride
            + dim_offsets[None, :] * cache_dim_stride
        )
        element_mask = (
            context_token_mask[:, None] & dim_mask[None, :]
        )
        key = tl.load(
            cache_ptr + cache_offsets,
            mask=element_mask,
            other=0.0,
        )
        scores = tl.dot(query, tl.trans(key)) * scale
        scores = tl.where(
            attention_mask,
            scores,
            -float("inf"),
        )

        block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, block_max)
        previous_scale = tl.exp(running_max - next_max)
        probabilities = tl.exp(scores - next_max[:, None])
        next_sum = (
            running_sum * previous_scale
            + tl.sum(probabilities, axis=1)
        )

        value = tl.load(
            cache_ptr + cache_plane_stride + cache_offsets,
            mask=element_mask,
            other=0.0,
        )
        accumulator = (
            accumulator * previous_scale[:, None]
            + tl.dot(
                probabilities.to(value.dtype),
                value,
            )
        )
        running_max = next_max
        running_sum = next_sum

    output_offsets = (
        query_indices[:, None] * output_token_stride
        + query_head_indices[:, None] * output_head_stride
        + dim_offsets[None, :] * output_dim_stride
    )
    tl.store(
        output_ptr + output_offsets,
        accumulator / running_sum[:, None],
        mask=query_mask[:, None] & dim_mask[None, :],
    )


def triton_write_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_kv_cache: torch.Tensor,
    metadata: AttentionMetadata,
) -> None:
    """Write every flattened token and KV head in one Triton launch."""

    head_dim = key.shape[-1]
    block_dim = triton.next_power_of_2(head_dim)
    num_warps = 4 if block_dim >= 64 else 1
    grid = (key.shape[0], key.shape[1])
    _write_kv_cache_kernel[grid](
        key,
        value,
        layer_kv_cache,
        metadata.slot_mapping,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        layer_kv_cache.stride(0),
        layer_kv_cache.stride(1),
        layer_kv_cache.stride(2),
        layer_kv_cache.stride(3),
        layer_kv_cache.stride(4),
        HEAD_DIM=head_dim,
        BLOCK_SIZE=metadata.block_size,
        BLOCK_DIM=block_dim,
        num_warps=num_warps,
    )


def triton_paged_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_kv_cache: torch.Tensor,
    metadata: AttentionMetadata,
) -> torch.Tensor:
    """Write current K/V, then attend directly through the block table."""

    triton_write_kv_cache(
        key,
        value,
        layer_kv_cache,
        metadata,
    )
    output = torch.zeros_like(query)
    num_query_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    head_dim = query.shape[2]
    head_group_size = num_query_heads // num_kv_heads
    block_q = max(1, 16 // head_group_size)
    valid_query_rows = block_q * head_group_size
    block_m = triton.next_power_of_2(valid_query_rows)
    block_dim = max(16, triton.next_power_of_2(head_dim))
    num_warps = 4 if block_dim >= 64 else 1
    num_requests = metadata.num_requests
    total_query_blocks = query.shape[0] // block_q + num_requests
    grid = (total_query_blocks, num_kv_heads)
    _paged_attention_kernel[grid](
        query,
        layer_kv_cache,
        metadata.query_positions,
        metadata.query_context_start_positions,
        metadata.query_start_locations,
        metadata.block_table,
        output,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        layer_kv_cache.stride(0),
        layer_kv_cache.stride(1),
        layer_kv_cache.stride(2),
        layer_kv_cache.stride(3),
        layer_kv_cache.stride(4),
        metadata.block_table.stride(0),
        metadata.block_table.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        head_dim**-0.5,
        num_requests,
        HEAD_GROUP_SIZE=head_group_size,
        VALID_QUERY_ROWS=valid_query_rows,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=metadata.block_size,
        BLOCK_Q=block_q,
        BLOCK_M=block_m,
        BLOCK_N=16,
        BLOCK_DIM=block_dim,
        num_warps=num_warps,
    )
    return output


class TritonPagedAttentionOperator(AttentionOperator):
    """CUDA FP16 paged attention without request-wise CPU slicing."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_kv_cache: torch.Tensor,
        metadata: AttentionMetadata,
        /,
    ) -> torch.Tensor:
        if query.device.type != "cuda":
            raise RuntimeError(
                "triton_paged attention requires CUDA tensors"
            )
        if query.dtype != torch.float16:
            raise ValueError(
                "triton_paged attention currently supports only torch.float16"
            )
        return triton_paged_attention(
            query,
            key,
            value,
            layer_kv_cache,
            metadata,
        )
