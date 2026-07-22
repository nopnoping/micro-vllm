"""Tests for full-attention metadata."""

from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from micro_vllm.attention.base import AttentionMetadata, PAD_BLOCK_ID
from micro_vllm.attention.full.metadata import FullAttentionMetadata


def make_metadata(**overrides: object) -> FullAttentionMetadata:
    values = {
        "query_positions": torch.tensor([0, 1, 4], dtype=torch.int64),
        "query_start_locations": torch.tensor([0, 2, 3], dtype=torch.int64),
        "context_lengths": torch.tensor([2, 5], dtype=torch.int64),
        "block_table": torch.tensor(
            [[0, PAD_BLOCK_ID], [2, 3]],
            dtype=torch.int64,
        ),
        "slot_mapping": torch.tensor([0, 1, 12], dtype=torch.int64),
        "block_size": 4,
    }
    values.update(overrides)
    return FullAttentionMetadata(**values)


def test_metadata_stores_only_flattened_attention_inputs() -> None:
    metadata = make_metadata()

    assert [field.name for field in fields(metadata)] == [
        "query_positions",
        "query_start_locations",
        "context_lengths",
        "block_table",
        "slot_mapping",
        "block_size",
    ]
    assert metadata.num_requests == 2
    assert metadata.block_table.tolist() == [[0, PAD_BLOCK_ID], [2, 3]]
    assert metadata.block_size == 4


def test_full_metadata_implements_attention_metadata_contract() -> None:
    assert isinstance(make_metadata(), AttentionMetadata)


def test_metadata_is_frozen() -> None:
    metadata = make_metadata()

    with pytest.raises(FrozenInstanceError):
        metadata.block_size = 8
