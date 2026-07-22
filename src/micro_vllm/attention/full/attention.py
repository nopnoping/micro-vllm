"""Full causal attention semantics."""

from typing import cast

import torch

from micro_vllm.attention.base import Attention, AttentionMetadata
from micro_vllm.attention.full.metadata import FullAttentionMetadata
from micro_vllm.attention.full.operator import FullAttentionOperator


class FullAttention(Attention):
    """Full causal attention delegated to a selected tensor operator."""

    def __init__(self, operator: FullAttentionOperator) -> None:
        self.operator = operator

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
        return self.operator.build_metadata(
            query_positions=query_positions,
            query_start_locations=query_start_locations,
            context_lengths=context_lengths,
            block_table=block_table,
            slot_mapping=slot_mapping,
            block_size=block_size,
        )

    def forward(
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
            cast(FullAttentionMetadata, metadata),
        )
