"""Reference and optimized attention backend contracts."""

import math
from dataclasses import dataclass
from typing import Final, Protocol

import torch


PAD_BLOCK_ID: Final = -1


@dataclass(frozen=True, slots=True, eq=False)
class AttentionMetadata:
    """Flattened query and physical-cache indexing metadata."""

    query_positions: torch.Tensor
    query_start_locations: torch.Tensor
    context_lengths: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int

    def __post_init__(self) -> None:
        tensors = (
            ("query_positions", self.query_positions),
            ("query_start_locations", self.query_start_locations),
            ("context_lengths", self.context_lengths),
            ("block_table", self.block_table),
            ("slot_mapping", self.slot_mapping),
        )
        for field_name, tensor in tensors:
            if tensor.dtype is not torch.int64:
                raise ValueError(f"{field_name} must use torch.int64")
            if tensor.device != self.query_positions.device:
                raise ValueError("attention metadata tensors must share one device")

        for field_name, tensor in tensors[:3] + tensors[4:]:
            if tensor.ndim != 1:
                raise ValueError(f"{field_name} must be one-dimensional")
        if self.block_table.ndim != 2:
            raise ValueError("block_table must be two-dimensional")
        if self.block_size <= 0:
            raise ValueError("block_size must be greater than zero")

        if self.query_start_locations.numel() != self.num_requests + 1:
            raise ValueError(
                "query_start_locations length must equal num_requests + 1"
            )
        if self.block_table.shape[0] != self.num_requests:
            raise ValueError("block_table rows must equal num_requests")
        if self.slot_mapping.numel() != self.num_query_tokens:
            raise ValueError(
                "slot_mapping length must equal num_query_tokens"
            )

    @property
    def num_requests(self) -> int:
        """Number of requests represented by the metadata."""

        return self.context_lengths.numel()

    @property
    def num_query_tokens(self) -> int:
        """Number of flattened query tokens."""

        return self.query_positions.numel()


class AttentionBackend(Protocol):
    """Callable boundary for attention over one layer's physical KV cache."""

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_kv_cache: torch.Tensor,
        metadata: AttentionMetadata,
        /,
    ) -> torch.Tensor:
        """Return attention output for every flattened query row."""

        ...


def map_cache_slots(
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Map logical token positions to flattened physical cache slots."""

    logical_blocks = positions // block_size
    block_offsets = positions % block_size
    physical_blocks = block_table.index_select(0, logical_blocks)
    return physical_blocks * block_size + block_offsets


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
    """Gather each flattened query's complete causal K/V history."""

    flattened_cache = layer_kv_cache.view(
        2,
        -1,
        *layer_kv_cache.shape[3:],
    )
    histories: list[tuple[torch.Tensor, torch.Tensor]] = []
    for request_index in range(metadata.num_requests):
        query_start = metadata.query_start_locations[request_index].item()
        query_end = metadata.query_start_locations[request_index + 1].item()
        block_table = metadata.block_table[request_index]

        for position in metadata.query_positions[query_start:query_end].tolist():
            token_positions = torch.arange(
                position + 1,
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


class ReferenceAttentionBackend:
    """Readable full causal attention over paged physical K/V cache."""

    def __call__(
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
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

        write_kv_cache(key, value, layer_kv_cache, metadata)
        histories = gather_kv_cache(layer_kv_cache, metadata)
        head_group_size = num_attention_heads // num_key_value_heads
        scale = 1.0 / math.sqrt(query.shape[-1])
        output = torch.empty_like(query)

        for query_index, (cached_key, cached_value) in enumerate(histories):
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
                "hd,thd->ht",
                query[query_index].to(torch.float32),
                cached_key.to(torch.float32),
            )
            weights = torch.softmax(
                scores * scale,
                dim=-1,
                dtype=torch.float32,
            )
            attention = torch.einsum(
                "ht,thd->hd",
                weights,
                cached_value.to(torch.float32),
            )
            output[query_index].copy_(attention)

        return output
