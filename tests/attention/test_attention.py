"""Tests for attention metadata construction and operator delegation."""

import torch

from micro_vllm.attention.attention import Attention, AttentionMetadata
from micro_vllm.attention.operators.base import AttentionOperator
from micro_vllm.attention.spec import (
    ChunkedLocalSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
)


class RecordingOperator(AttentionOperator):
    def __init__(self) -> None:
        self.metadata: AttentionMetadata | None = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_kv_cache: torch.Tensor,
        metadata: AttentionMetadata,
        /,
    ) -> torch.Tensor:
        self.metadata = metadata
        return query


def test_attention_builds_metadata_and_delegates_to_operator() -> None:
    operator = RecordingOperator()
    attention = Attention(FullAttentionSpec(), "reference")
    attention.operator = operator
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

    assert metadata.query_positions is query_positions
    assert torch.equal(
        metadata.query_context_start_positions,
        torch.zeros_like(query_positions),
    )
    assert metadata.query_start_locations is query_start_locations
    assert metadata.context_lengths is context_lengths
    assert metadata.block_table is block_table
    assert metadata.slot_mapping is slot_mapping
    assert metadata.block_size == 4
    assert operator.metadata is metadata
    assert output is query


def test_attention_builds_per_query_visibility_starts() -> None:
    query_positions = torch.tensor([2, 3, 4, 5], dtype=torch.int64)
    common_arguments = {
        "query_positions": query_positions,
        "query_start_locations": torch.tensor([0, 4], dtype=torch.int64),
        "context_lengths": torch.tensor([6], dtype=torch.int64),
        "block_table": torch.tensor([[0, 1]], dtype=torch.int64),
        "slot_mapping": torch.tensor([2, 3, 4, 5], dtype=torch.int64),
        "block_size": 4,
    }

    full_metadata = Attention(FullAttentionSpec(), "reference").build_metadata(
        **common_arguments
    )
    sliding_metadata = Attention(
        SlidingWindowSpec(3),
        "reference",
    ).build_metadata(**common_arguments)
    chunked_metadata = Attention(
        ChunkedLocalSpec(4),
        "reference",
    ).build_metadata(**common_arguments)

    assert full_metadata.query_context_start_positions.tolist() == [0, 0, 0, 0]
    assert sliding_metadata.query_context_start_positions.tolist() == [0, 1, 2, 3]
    assert chunked_metadata.query_context_start_positions.tolist() == [0, 0, 4, 4]
