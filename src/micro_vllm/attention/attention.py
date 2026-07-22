"""Attention semantics and runtime metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

PAD_BLOCK_ID: Final = -1


@dataclass(frozen=True, slots=True, eq=False)
class AttentionMetadata:
    """Flattened queries, visibility bounds, and paged-cache inputs."""

    query_positions: torch.Tensor
    query_context_start_positions: torch.Tensor
    query_start_locations: torch.Tensor
    context_lengths: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int

    @property
    def num_requests(self) -> int:
        """Number of requests represented by this batch."""

        return self.context_lengths.numel()


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


class Attention:
    """Own an operator and the metadata semantics it consumes."""

    def __init__(self, operator_name: str) -> None:
        from micro_vllm.attention.operators.registry import (
            get_attention_operator_class,
        )

        operator_class = get_attention_operator_class(operator_name)
        self.operator = operator_class()

    def build_metadata(
        self,
        *,
        query_positions: torch.Tensor,
        query_start_locations: torch.Tensor,
        context_lengths: torch.Tensor,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ) -> AttentionMetadata:
        """Build metadata for full causal attention."""

        return AttentionMetadata(
            query_positions=query_positions,
            query_context_start_positions=torch.zeros_like(query_positions),
            query_start_locations=query_start_locations,
            context_lengths=context_lengths,
            block_table=block_table,
            slot_mapping=slot_mapping,
            block_size=block_size,
        )

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_kv_cache: torch.Tensor,
        metadata: AttentionMetadata,
        /,
    ) -> torch.Tensor:
        return self.operator(
            query,
            key,
            value,
            layer_kv_cache,
            metadata,
        )
