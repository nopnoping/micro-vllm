"""Readable reference attention operator."""

import math

import torch

from micro_vllm.attention.attention import AttentionMetadata, map_cache_slots
from micro_vllm.attention.operators.base import AttentionOperator


def write_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_kv_cache: torch.Tensor,
    metadata: AttentionMetadata,
) -> None:
    """Write current query keys and values into their physical cache slots."""

    flattened_cache = layer_kv_cache.view(
        2,
        -1,
        *layer_kv_cache.shape[3:],
    )
    flattened_cache[0].index_copy_(0, metadata.slot_mapping, key)
    flattened_cache[1].index_copy_(0, metadata.slot_mapping, value)


def gather_kv_cache(
    layer_kv_cache: torch.Tensor,
    metadata: AttentionMetadata,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Gather each request's visible K/V context."""

    flattened_cache = layer_kv_cache.view(
        2,
        -1,
        *layer_kv_cache.shape[3:],
    )
    histories: list[tuple[torch.Tensor, torch.Tensor]] = []
    for request_index in range(metadata.num_requests):
        query_start = metadata.query_start_locations[request_index].item()
        context_start = metadata.query_context_start_positions[
            query_start
        ].item()
        context_length = metadata.context_lengths[request_index].item()
        block_table = metadata.block_table[request_index]
        token_positions = torch.arange(
            context_start,
            context_length,
            dtype=torch.int64,
            device=metadata.query_positions.device,
        )
        slots = map_cache_slots(
            token_positions,
            block_table,
            metadata.block_size,
        )
        histories.append(
            (
                flattened_cache[0].index_select(0, slots),
                flattened_cache[1].index_select(0, slots),
            )
        )

    return tuple(histories)


class ReferenceAttentionOperator(AttentionOperator):
    """Readable causal attention over a paged physical K/V cache."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_kv_cache: torch.Tensor,
        metadata: AttentionMetadata,
        /,
    ) -> torch.Tensor:
        num_attention_heads = query.shape[1]
        num_key_value_heads = key.shape[1]
        write_kv_cache(key, value, layer_kv_cache, metadata)
        histories = gather_kv_cache(layer_kv_cache, metadata)
        head_group_size = num_attention_heads // num_key_value_heads
        scale = 1.0 / math.sqrt(query.shape[-1])
        output = torch.empty_like(query)

        for request_index, (cached_key, cached_value) in enumerate(histories):
            query_start = metadata.query_start_locations[request_index].item()
            query_end = metadata.query_start_locations[request_index + 1].item()
            if head_group_size > 1:
                cached_key = cached_key.repeat_interleave(
                    head_group_size,
                    dim=1,
                )
                cached_value = cached_value.repeat_interleave(
                    head_group_size,
                    dim=1,
                )

            scores = torch.einsum(
                "qhd,khd->hqk",
                query[query_start:query_end].to(torch.float32),
                cached_key.to(torch.float32),
            )
            context_start = metadata.query_context_start_positions[
                query_start
            ].item()
            context_positions = torch.arange(
                context_start,
                context_start + cached_key.shape[0],
                dtype=torch.int64,
                device=metadata.query_positions.device,
            )
            query_positions = metadata.query_positions[query_start:query_end]
            query_context_starts = metadata.query_context_start_positions[
                query_start:query_end
            ]
            causal_mask = (
                context_positions.unsqueeze(0)
                < query_context_starts.unsqueeze(1)
            ) | (
                context_positions.unsqueeze(0) > query_positions.unsqueeze(1)
            )
            scores.masked_fill_(causal_mask.unsqueeze(0), -torch.inf)
            weights = torch.softmax(
                scores * scale,
                dim=-1,
                dtype=torch.float32,
            )
            attention = torch.einsum(
                "hqk,khd->qhd",
                weights,
                cached_value.to(torch.float32),
            )
            output[query_start:query_end].copy_(attention)

        return output
