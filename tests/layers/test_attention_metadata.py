"""Tests for the low-level attention metadata contract."""

from dataclasses import FrozenInstanceError, fields

import pytest
import torch

import micro_vllm.layers as layers
from micro_vllm.layers.attention_backend import AttentionMetadata, PAD_BLOCK_ID


def make_metadata(**overrides: object) -> AttentionMetadata:
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
    return AttentionMetadata(**values)


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


def test_layers_package_exports_attention_metadata_contract() -> None:
    assert layers.__all__ == (
        "AttentionBackend",
        "AttentionMetadata",
        "LMHead",
        "Linear",
        "PAD_BLOCK_ID",
        "RMSNorm",
        "ReferenceAttentionBackend",
        "RotaryEmbedding",
        "SwiGLUMLP",
        "TokenEmbedding",
        "gather_kv_cache",
        "write_kv_cache",
    )
    assert layers.AttentionMetadata is AttentionMetadata
    assert layers.PAD_BLOCK_ID == PAD_BLOCK_ID


def test_metadata_is_frozen() -> None:
    metadata = make_metadata()

    with pytest.raises(FrozenInstanceError):
        metadata.block_size = 8
