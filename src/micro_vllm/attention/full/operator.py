"""Operator contract for full causal attention."""

from abc import ABC, abstractmethod

import torch

from micro_vllm.attention.full.metadata import FullAttentionMetadata


class FullAttentionOperator(ABC):
    """Tensor operator used by full causal attention."""

    def build_metadata(
        self,
        *,
        query_positions: torch.Tensor,
        query_start_locations: torch.Tensor,
        context_lengths: torch.Tensor,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ) -> FullAttentionMetadata:
        """Build metadata consumed by this full-attention operator."""

        return FullAttentionMetadata(
            query_positions=query_positions,
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
        metadata: FullAttentionMetadata,
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
        metadata: FullAttentionMetadata,
        /,
    ) -> torch.Tensor:
        """Compute full causal attention for one layer."""
