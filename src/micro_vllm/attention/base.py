"""Attention semantic contracts shared by models and execution."""

from abc import ABC, abstractmethod
from typing import Final

import torch


PAD_BLOCK_ID: Final = -1


class AttentionMetadata:
    """Base type for metadata constructed by an attention implementation."""

    __slots__ = ()

    query_positions: torch.Tensor
    query_start_locations: torch.Tensor
    context_lengths: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int


class Attention(ABC):
    """Attention semantics consumed by model-specific projection layers."""

    @abstractmethod
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
        """Build implementation-specific metadata for one flattened batch."""

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_kv_cache: torch.Tensor,
        metadata: AttentionMetadata,
        /,
    ) -> torch.Tensor:
        return self.forward(
            query,
            key,
            value,
            layer_kv_cache,
            metadata,
        )

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_kv_cache: torch.Tensor,
        metadata: AttentionMetadata,
        /,
    ) -> torch.Tensor:
        """Apply this attention semantic to one layer's Q/K/V tensors."""


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
