"""Tests for full-attention semantics and operator composition."""

import torch

from micro_vllm.attention.full.attention import FullAttention
from micro_vllm.attention.full.metadata import FullAttentionMetadata
from micro_vllm.attention.full.operator import FullAttentionOperator


class RecordingOperator(FullAttentionOperator):
    def __init__(self) -> None:
        self.built_metadata = False
        self.metadata: FullAttentionMetadata | None = None

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
        self.built_metadata = True
        return super().build_metadata(
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
        metadata: FullAttentionMetadata,
        /,
    ) -> torch.Tensor:
        self.metadata = metadata
        return query


def test_full_attention_builds_metadata_and_delegates_to_operator() -> None:
    operator = RecordingOperator()
    attention = FullAttention(operator)
    query_positions = torch.tensor([0, 1], dtype=torch.int64)
    query_start_locations = torch.tensor([0, 2], dtype=torch.int64)
    context_lengths = torch.tensor([2], dtype=torch.int64)
    block_table = torch.tensor([[0]], dtype=torch.int64)
    slot_mapping = torch.tensor([0, 1], dtype=torch.int64)
    metadata = attention.build_metadata(
        query_positions=query_positions,
        query_start_locations=query_start_locations,
        context_lengths=context_lengths,
        block_table=block_table,
        slot_mapping=slot_mapping,
        block_size=4,
    )
    query = torch.randn(2, 2, 4)
    key = torch.randn(2, 1, 4)
    value = torch.randn(2, 1, 4)
    layer_kv_cache = torch.empty(2, 1, 4, 1, 4)

    output = attention(
        query,
        key,
        value,
        layer_kv_cache,
        metadata,
    )

    assert isinstance(metadata, FullAttentionMetadata)
    assert operator.built_metadata is True
    assert metadata.query_positions is query_positions
    assert metadata.query_start_locations is query_start_locations
    assert metadata.context_lengths is context_lengths
    assert metadata.block_table is block_table
    assert metadata.slot_mapping is slot_mapping
    assert metadata.block_size == 4
    assert operator.metadata is metadata
    assert output is query
