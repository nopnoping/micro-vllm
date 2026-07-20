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
    assert metadata.num_query_tokens == 3
    assert metadata.block_table.tolist() == [[0, PAD_BLOCK_ID], [2, 3]]
    assert metadata.block_size == 4


def test_layers_package_exports_attention_metadata_contract() -> None:
    assert layers.__all__ == (
        "AttentionBackend",
        "AttentionMetadata",
        "PAD_BLOCK_ID",
        "ReferenceAttentionBackend",
        "gather_kv_cache",
        "write_kv_cache",
    )
    assert layers.AttentionMetadata is AttentionMetadata
    assert layers.PAD_BLOCK_ID == PAD_BLOCK_ID


def test_metadata_is_frozen() -> None:
    metadata = make_metadata()

    with pytest.raises(FrozenInstanceError):
        metadata.block_size = 8


@pytest.mark.parametrize(
    "field_name",
    [
        "query_positions",
        "query_start_locations",
        "context_lengths",
        "slot_mapping",
    ],
)
def test_flattened_tensors_must_be_one_dimensional(field_name: str) -> None:
    value = getattr(make_metadata(), field_name).reshape(1, -1)

    with pytest.raises(ValueError, match=rf"^{field_name} must be one-dimensional$"):
        make_metadata(**{field_name: value})


def test_block_table_must_be_two_dimensional() -> None:
    with pytest.raises(ValueError, match="block_table must be two-dimensional"):
        make_metadata(block_table=torch.tensor([0, 1], dtype=torch.int64))


@pytest.mark.parametrize(
    "field_name",
    [
        "query_positions",
        "query_start_locations",
        "context_lengths",
        "block_table",
        "slot_mapping",
    ],
)
def test_metadata_tensors_use_int64(field_name: str) -> None:
    value = getattr(make_metadata(), field_name).to(torch.int32)

    with pytest.raises(ValueError, match=rf"^{field_name} must use torch.int64$"):
        make_metadata(**{field_name: value})


def test_metadata_tensors_share_one_device() -> None:
    with pytest.raises(
        ValueError,
        match="attention metadata tensors must share one device",
    ):
        make_metadata(
            slot_mapping=torch.empty(3, dtype=torch.int64, device="meta")
        )


def test_query_start_count_matches_request_count() -> None:
    with pytest.raises(
        ValueError,
        match="query_start_locations length must equal num_requests",
    ):
        make_metadata(
            query_start_locations=torch.tensor([0, 3], dtype=torch.int64)
        )


def test_block_table_rows_match_request_count() -> None:
    with pytest.raises(ValueError, match="block_table rows must equal num_requests"):
        make_metadata(block_table=torch.tensor([[0, 1]], dtype=torch.int64))


def test_slot_mapping_count_matches_query_count() -> None:
    with pytest.raises(
        ValueError,
        match="slot_mapping length must equal num_query_tokens",
    ):
        make_metadata(slot_mapping=torch.tensor([0, 1], dtype=torch.int64))


@pytest.mark.parametrize("block_size", [0, -1])
def test_block_size_must_be_positive(block_size: int) -> None:
    with pytest.raises(ValueError, match="block_size must be greater than zero"):
        make_metadata(block_size=block_size)
