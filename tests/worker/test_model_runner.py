"""Tests for the internal flattened ModelRunner input contract."""

from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from micro_vllm.layers.attention_backend import AttentionMetadata, PAD_BLOCK_ID
from micro_vllm.core.scheduler_output import ScheduledRequest, SchedulerOutput
from micro_vllm.worker.model_runner import ModelInput, flatten_model_input


def make_attention_metadata() -> AttentionMetadata:
    return AttentionMetadata(
        query_positions=torch.tensor([0, 1, 4], dtype=torch.int64),
        query_start_locations=torch.tensor([0, 2, 3], dtype=torch.int64),
        context_lengths=torch.tensor([2, 5], dtype=torch.int64),
        block_table=torch.tensor(
            [[0, PAD_BLOCK_ID], [2, 3]],
            dtype=torch.int64,
        ),
        slot_mapping=torch.tensor([0, 1, 12], dtype=torch.int64),
        block_size=4,
    )


def make_model_input(**overrides: object) -> ModelInput:
    values = {
        "input_ids": torch.tensor([10, 11, 12], dtype=torch.int64),
        "attention_metadata": make_attention_metadata(),
        "request_ids": ("request-1", "request-2"),
        "sampling_row_indices": torch.tensor([1, 2], dtype=torch.int64),
    }
    values.update(overrides)
    return ModelInput(**values)


def test_model_input_contains_only_runner_batch_data() -> None:
    model_input = make_model_input()

    assert [field.name for field in fields(model_input)] == [
        "input_ids",
        "attention_metadata",
        "request_ids",
        "sampling_row_indices",
    ]
    assert model_input.input_ids.tolist() == [10, 11, 12]
    assert model_input.positions.tolist() == [0, 1, 4]
    assert model_input.positions is model_input.attention_metadata.query_positions
    assert model_input.request_ids == ("request-1", "request-2")
    assert model_input.sampling_row_indices.tolist() == [1, 2]


def test_model_input_is_frozen_and_owns_request_id_tuple() -> None:
    request_ids = ["request-1", "request-2"]
    model_input = make_model_input(request_ids=request_ids)
    request_ids.append("request-3")

    with pytest.raises(FrozenInstanceError):
        model_input.request_ids = ()

    assert model_input.request_ids == ("request-1", "request-2")


@pytest.mark.parametrize("field_name", ["input_ids", "sampling_row_indices"])
def test_model_input_tensors_must_be_one_dimensional(field_name: str) -> None:
    value = getattr(make_model_input(), field_name).reshape(1, -1)

    with pytest.raises(ValueError, match=rf"^{field_name} must be one-dimensional$"):
        make_model_input(**{field_name: value})


@pytest.mark.parametrize("field_name", ["input_ids", "sampling_row_indices"])
def test_model_input_tensors_use_int64(field_name: str) -> None:
    value = getattr(make_model_input(), field_name).to(torch.int32)

    with pytest.raises(ValueError, match=rf"^{field_name} must use torch.int64$"):
        make_model_input(**{field_name: value})


def test_model_input_tensors_share_attention_metadata_device() -> None:
    with pytest.raises(ValueError, match="ModelInput tensors must share one device"):
        make_model_input(
            sampling_row_indices=torch.empty(
                2,
                dtype=torch.int64,
                device="meta",
            )
        )


def test_input_ids_count_matches_query_count() -> None:
    with pytest.raises(ValueError, match="input_ids length must equal num_query_tokens"):
        make_model_input(input_ids=torch.tensor([10, 11], dtype=torch.int64))


def test_request_ids_count_matches_request_count() -> None:
    with pytest.raises(ValueError, match="request_ids length must equal num_requests"):
        make_model_input(request_ids=("request-1",))


def test_model_input_accepts_batch_without_sampling_rows() -> None:
    model_input = make_model_input(
        sampling_row_indices=torch.empty(0, dtype=torch.int64)
    )

    assert model_input.sampling_row_indices.numel() == 0


def make_scheduled_request(
    request_id: str,
    input_token_ids: tuple[int, ...],
    *,
    num_computed_tokens: int,
    requires_sampling: bool,
) -> ScheduledRequest:
    return ScheduledRequest(
        request_id=request_id,
        input_token_ids=input_token_ids,
        num_computed_tokens=num_computed_tokens,
        num_scheduled_tokens=len(input_token_ids),
        requires_sampling=requires_sampling,
        block_table=(0,),
    )


def flatten_requests(*requests: ScheduledRequest) -> ModelInput:
    total_tokens = sum(request.num_scheduled_tokens for request in requests)
    return flatten_model_input(
        SchedulerOutput(requests),
        block_table=torch.zeros((len(requests), 2), dtype=torch.int64),
        slot_mapping=torch.zeros(total_tokens, dtype=torch.int64),
        block_size=4,
    )


def test_flattens_single_request_prefill() -> None:
    model_input = flatten_requests(
        make_scheduled_request(
            "request-1",
            (10, 11, 12),
            num_computed_tokens=0,
            requires_sampling=True,
        )
    )

    assert model_input.input_ids.tolist() == [10, 11, 12]
    assert model_input.positions.tolist() == [0, 1, 2]
    assert model_input.attention_metadata.query_start_locations.tolist() == [0, 3]
    assert model_input.attention_metadata.context_lengths.tolist() == [3]
    assert model_input.request_ids == ("request-1",)
    assert model_input.sampling_row_indices.tolist() == [2]


def test_flattens_single_request_decode() -> None:
    model_input = flatten_requests(
        make_scheduled_request(
            "request-1",
            (13,),
            num_computed_tokens=3,
            requires_sampling=True,
        )
    )

    assert model_input.input_ids.tolist() == [13]
    assert model_input.positions.tolist() == [3]
    assert model_input.attention_metadata.query_start_locations.tolist() == [0, 1]
    assert model_input.attention_metadata.context_lengths.tolist() == [4]
    assert model_input.sampling_row_indices.tolist() == [0]


def test_chunked_prefill_starts_from_computed_position_without_sampling() -> None:
    model_input = flatten_requests(
        make_scheduled_request(
            "request-1",
            (12, 13),
            num_computed_tokens=2,
            requires_sampling=False,
        )
    )

    assert model_input.positions.tolist() == [2, 3]
    assert model_input.attention_metadata.context_lengths.tolist() == [4]
    assert model_input.sampling_row_indices.tolist() == []


def test_mixed_batch_preserves_request_order_and_independent_positions() -> None:
    model_input = flatten_requests(
        make_scheduled_request(
            "decode",
            (20,),
            num_computed_tokens=4,
            requires_sampling=True,
        ),
        make_scheduled_request(
            "prefill",
            (30, 31, 32),
            num_computed_tokens=0,
            requires_sampling=False,
        ),
    )

    assert model_input.input_ids.tolist() == [20, 30, 31, 32]
    assert model_input.positions.tolist() == [4, 0, 1, 2]
    assert model_input.attention_metadata.query_start_locations.tolist() == [0, 1, 4]
    assert model_input.attention_metadata.context_lengths.tolist() == [5, 3]
    assert model_input.request_ids == ("decode", "prefill")
    assert model_input.sampling_row_indices.tolist() == [0]


def test_sampling_rows_follow_request_order() -> None:
    model_input = flatten_requests(
        make_scheduled_request(
            "first",
            (10,),
            num_computed_tokens=3,
            requires_sampling=True,
        ),
        make_scheduled_request(
            "second",
            (20, 21),
            num_computed_tokens=5,
            requires_sampling=True,
        ),
    )

    assert model_input.request_ids == ("first", "second")
    assert model_input.sampling_row_indices.tolist() == [0, 2]


def test_flattening_requires_one_slot_per_scheduled_token() -> None:
    request = make_scheduled_request(
        "request-1",
        (10, 11),
        num_computed_tokens=0,
        requires_sampling=True,
    )

    with pytest.raises(
        ValueError,
        match="slot_mapping length must equal num_query_tokens",
    ):
        flatten_model_input(
            SchedulerOutput((request,)),
            block_table=torch.zeros((1, 1), dtype=torch.int64),
            slot_mapping=torch.zeros(1, dtype=torch.int64),
            block_size=4,
        )
