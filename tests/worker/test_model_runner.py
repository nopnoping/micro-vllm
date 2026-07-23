"""Tests for the internal flattened ModelRunner input contract."""

from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from micro_vllm.attention.attention import Attention, AttentionMetadata
from micro_vllm.attention.spec import FullAttentionSpec, SlidingWindowSpec
from micro_vllm.config import CacheConfig, LLMConfig, ModelConfig
from micro_vllm.core.kv_cache_spec import (
    KVCacheTopology,
    build_kv_cache_topology,
)
from micro_vllm.core.scheduler_output import (
    CacheGroupBlockTable,
    ScheduledRequest,
    SchedulerOutput,
)
from micro_vllm.sampling.params import SamplingParams
from micro_vllm.worker.model_runner import (
    ModelInput,
    ModelRunner,
)

NULL_BLOCK_ID = 7


class _InputModel:
    def __init__(self, layer_attentions: tuple[Attention, ...]) -> None:
        self.layer_attentions = layer_attentions


def make_input_runner(
    layer_attentions: tuple[Attention, ...],
    block_size: int,
    device: torch.device,
) -> ModelRunner:
    runner = ModelRunner(
        LLMConfig(
            model_config=ModelConfig(model="unused", device=device),
            cache_config=CacheConfig(
                block_size=block_size,
                num_blocks=8,
            ),
        ),
    )
    runner._model = _InputModel(layer_attentions)
    runner._kv_cache_topology = build_kv_cache_topology(
        tuple(attention.spec for attention in layer_attentions)
    )
    return runner


def flatten_model_input(
    plan: SchedulerOutput,
    *,
    topology: KVCacheTopology,
    layer_attentions: tuple[Attention, ...],
    block_tables: tuple[torch.Tensor, ...],
    slot_mappings: tuple[torch.Tensor, ...],
    block_size: int,
) -> ModelInput:
    runner = make_input_runner(
        layer_attentions,
        block_size,
        block_tables[0].device,
    )
    runner._kv_cache_topology = topology
    return runner._flatten_model_input(
        plan,
        block_tables=block_tables,
        slot_mappings=slot_mappings,
    )


def build_slot_mapping(
    plan: SchedulerOutput,
    cache_group_index: int,
    block_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    attention = Attention(FullAttentionSpec(), "reference")
    return make_input_runner(
        (attention,),
        block_size,
        device,
    )._build_slot_mapping(plan, cache_group_index)


def build_block_table(
    plan: SchedulerOutput,
    cache_group_index: int,
    *,
    null_block_id: int,
    device: torch.device,
) -> torch.Tensor:
    attention = Attention(FullAttentionSpec(), "reference")
    return make_input_runner(
        (attention,),
        block_size=4,
        device=device,
    )._build_block_table(
        plan,
        cache_group_index,
        null_block_id=null_block_id,
    )


def make_attention_metadata() -> AttentionMetadata:
    return AttentionMetadata(
        query_positions=torch.tensor([0, 1, 4], dtype=torch.int64),
        query_context_start_positions=torch.zeros(3, dtype=torch.int64),
        query_start_locations=torch.tensor([0, 2, 3], dtype=torch.int64),
        context_lengths=torch.tensor([2, 5], dtype=torch.int64),
        block_table=torch.tensor(
            [[0, NULL_BLOCK_ID], [2, 3]],
            dtype=torch.int64,
        ),
        slot_mapping=torch.tensor([0, 1, 12], dtype=torch.int64),
        block_size=4,
    )


def make_model_input(**overrides: object) -> ModelInput:
    metadata = make_attention_metadata()
    values = {
        "input_ids": torch.tensor([10, 11, 12], dtype=torch.int64),
        "positions": metadata.query_positions,
        "layer_attention_metadata": (metadata,),
        "request_ids": ("request-1", "request-2"),
        "sampling_row_indices": torch.tensor([1, 2], dtype=torch.int64),
    }
    values.update(overrides)
    return ModelInput(**values)


def test_model_input_contains_only_runner_batch_data() -> None:
    model_input = make_model_input()

    assert [field.name for field in fields(model_input)] == [
        "input_ids",
        "positions",
        "layer_attention_metadata",
        "request_ids",
        "sampling_row_indices",
    ]
    assert model_input.input_ids.tolist() == [10, 11, 12]
    assert model_input.positions.tolist() == [0, 1, 4]
    assert (
        model_input.positions
        is model_input.layer_attention_metadata[0].query_positions
    )
    assert model_input.request_ids == ("request-1", "request-2")
    assert model_input.sampling_row_indices.tolist() == [1, 2]


def test_model_input_is_frozen_and_owns_request_id_tuple() -> None:
    request_ids = ["request-1", "request-2"]
    model_input = make_model_input(request_ids=request_ids)
    request_ids.append("request-3")

    with pytest.raises(FrozenInstanceError):
        model_input.request_ids = ()

    assert model_input.request_ids == ("request-1", "request-2")


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
        block_tables=(CacheGroupBlockTable((0,)),),
        sampling_params=(
            SamplingParams(max_tokens=1) if requires_sampling else None
        ),
        sample_index=0 if requires_sampling else None,
    )


def flatten_requests(*requests: ScheduledRequest) -> ModelInput:
    total_tokens = sum(request.num_scheduled_tokens for request in requests)
    return flatten_model_input(
        SchedulerOutput(requests),
        topology=build_kv_cache_topology((FullAttentionSpec(),)),
        layer_attentions=(Attention(FullAttentionSpec(), "reference"),),
        block_tables=(
            torch.zeros((len(requests), 2), dtype=torch.int64),
        ),
        slot_mappings=(torch.zeros(total_tokens, dtype=torch.int64),),
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
    assert (
        model_input.layer_attention_metadata[
            0
        ].query_context_start_positions.tolist()
        == [0, 0, 0]
    )
    metadata = model_input.layer_attention_metadata[0]
    assert metadata.query_start_locations.tolist() == [0, 3]
    assert metadata.context_lengths.tolist() == [3]
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
    metadata = model_input.layer_attention_metadata[0]
    assert metadata.query_start_locations.tolist() == [0, 1]
    assert metadata.context_lengths.tolist() == [4]
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
    assert model_input.layer_attention_metadata[0].context_lengths.tolist() == [4]
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
    metadata = model_input.layer_attention_metadata[0]
    assert metadata.query_start_locations.tolist() == [0, 1, 4]
    assert metadata.context_lengths.tolist() == [5, 3]
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


def test_builds_per_group_metadata_over_shared_batch_tensors() -> None:
    full = Attention(FullAttentionSpec(), "reference")
    sliding = Attention(SlidingWindowSpec(3), "reference")
    topology = build_kv_cache_topology(
        (full.spec, sliding.spec, full.spec)
    )
    request = ScheduledRequest(
        request_id="request-1",
        input_token_ids=(10,),
        num_computed_tokens=4,
        num_scheduled_tokens=1,
        requires_sampling=False,
        block_tables=(
            CacheGroupBlockTable((2, 3, 4)),
            CacheGroupBlockTable((NULL_BLOCK_ID, NULL_BLOCK_ID, 5)),
        ),
    )
    plan = SchedulerOutput((request,))
    block_tables = tuple(
        build_block_table(
            plan,
            group_index,
            null_block_id=NULL_BLOCK_ID,
            device=torch.device("cpu"),
        )
        for group_index in range(2)
    )
    slot_mappings = tuple(
        build_slot_mapping(
            plan,
            group_index,
            2,
            device=torch.device("cpu"),
        )
        for group_index in range(2)
    )

    model_input = flatten_model_input(
        plan,
        topology=topology,
        layer_attentions=(full, sliding, full),
        block_tables=block_tables,
        slot_mappings=slot_mappings,
        block_size=2,
    )

    full_metadata, sliding_metadata, repeated_full_metadata = (
        model_input.layer_attention_metadata
    )
    assert repeated_full_metadata is full_metadata
    assert sliding_metadata is not full_metadata
    assert full_metadata.query_positions is sliding_metadata.query_positions
    assert (
        full_metadata.query_start_locations
        is sliding_metadata.query_start_locations
    )
    assert full_metadata.context_lengths is sliding_metadata.context_lengths
    assert full_metadata.query_context_start_positions.tolist() == [0]
    assert sliding_metadata.query_context_start_positions.tolist() == [2]
    assert full_metadata.block_table.tolist() == [[2, 3, 4]]
    assert sliding_metadata.block_table.tolist() == [
        [NULL_BLOCK_ID, NULL_BLOCK_ID, 5]
    ]
    assert full_metadata.slot_mapping.tolist() == [8]
    assert sliding_metadata.slot_mapping.tolist() == [10]


def test_slot_mapping_crosses_physical_block_boundary() -> None:
    request = ScheduledRequest(
        request_id="request-1",
        input_token_ids=(10, 11),
        num_computed_tokens=3,
        num_scheduled_tokens=2,
        requires_sampling=False,
        block_tables=(CacheGroupBlockTable((2, 5)),),
    )

    slot_mapping = build_slot_mapping(
        SchedulerOutput((request,)),
        0,
        4,
        device=torch.device("cpu"),
    )

    assert slot_mapping.tolist() == [11, 20]
    assert slot_mapping.dtype is torch.int64
    assert slot_mapping.device == torch.device("cpu")


def test_slot_mapping_preserves_mixed_request_order() -> None:
    first = ScheduledRequest(
        request_id="first",
        input_token_ids=(10,),
        num_computed_tokens=4,
        num_scheduled_tokens=1,
        requires_sampling=False,
        block_tables=(CacheGroupBlockTable((0, 3)),),
    )
    second = ScheduledRequest(
        request_id="second",
        input_token_ids=(20, 21),
        num_computed_tokens=1,
        num_scheduled_tokens=2,
        requires_sampling=False,
        block_tables=(CacheGroupBlockTable((6,)),),
    )

    slot_mapping = build_slot_mapping(
        SchedulerOutput((first, second)),
        0,
        4,
        device=torch.device("cpu"),
    )

    assert slot_mapping.tolist() == [12, 25, 26]


def test_slot_mapping_stays_within_flattened_cache_capacity() -> None:
    request = ScheduledRequest(
        request_id="request-1",
        input_token_ids=(10,),
        num_computed_tokens=7,
        num_scheduled_tokens=1,
        requires_sampling=False,
        block_tables=(CacheGroupBlockTable((0, 6)),),
    )
    cache_config = CacheConfig(block_size=4, num_blocks=8)

    slot_mapping = build_slot_mapping(
        SchedulerOutput((request,)),
        0,
        cache_config.block_size,
        device=torch.device("cpu"),
    )

    assert slot_mapping.tolist() == [27]
    assert slot_mapping.item() < cache_config.num_blocks * cache_config.block_size


def test_block_table_pads_different_request_lengths() -> None:
    first = ScheduledRequest(
        request_id="first",
        input_token_ids=(10,),
        num_computed_tokens=0,
        num_scheduled_tokens=1,
        requires_sampling=False,
        block_tables=(CacheGroupBlockTable((2,)),),
    )
    second = ScheduledRequest(
        request_id="second",
        input_token_ids=(20,),
        num_computed_tokens=4,
        num_scheduled_tokens=1,
        requires_sampling=False,
        block_tables=(CacheGroupBlockTable((3, 5, 6)),),
    )

    block_table = build_block_table(
        SchedulerOutput((first, second)),
        0,
        null_block_id=NULL_BLOCK_ID,
        device=torch.device("cpu"),
    )

    assert block_table.tolist() == [
        [2, NULL_BLOCK_ID, NULL_BLOCK_ID],
        [3, 5, 6],
    ]
    assert block_table.dtype is torch.int64
    assert block_table.device == torch.device("cpu")


def test_block_table_preserves_explicit_null_mapping() -> None:
    request = ScheduledRequest(
        request_id="request-1",
        input_token_ids=(10,),
        num_computed_tokens=0,
        num_scheduled_tokens=1,
        requires_sampling=False,
        block_tables=(CacheGroupBlockTable((2, NULL_BLOCK_ID)),),
    )

    block_table = build_block_table(
        SchedulerOutput((request,)),
        0,
        null_block_id=NULL_BLOCK_ID,
        device=torch.device("cpu"),
    )

    assert block_table.tolist() == [[2, NULL_BLOCK_ID]]


def test_empty_plan_builds_two_dimensional_empty_block_table() -> None:
    block_table = build_block_table(
        SchedulerOutput(()),
        0,
        null_block_id=NULL_BLOCK_ID,
        device=torch.device("cpu"),
    )

    assert block_table.shape == torch.Size((0, 0))
    assert block_table.dtype is torch.int64


def test_padding_is_not_referenced_by_slot_mapping() -> None:
    first = ScheduledRequest(
        request_id="first",
        input_token_ids=(10,),
        num_computed_tokens=0,
        num_scheduled_tokens=1,
        requires_sampling=False,
        block_tables=(CacheGroupBlockTable((2,)),),
    )
    second = ScheduledRequest(
        request_id="second",
        input_token_ids=(20,),
        num_computed_tokens=4,
        num_scheduled_tokens=1,
        requires_sampling=False,
        block_tables=(CacheGroupBlockTable((3, 5)),),
    )
    plan = SchedulerOutput((first, second))
    cache_config = CacheConfig(block_size=4, num_blocks=8)

    block_table = build_block_table(
        plan,
        0,
        null_block_id=NULL_BLOCK_ID,
        device=torch.device("cpu"),
    )
    slot_mapping = build_slot_mapping(
        plan,
        0,
        cache_config.block_size,
        device=torch.device("cpu"),
    )
    model_input = flatten_model_input(
        plan,
        topology=build_kv_cache_topology((FullAttentionSpec(),)),
        layer_attentions=(Attention(FullAttentionSpec(), "reference"),),
        block_tables=(block_table,),
        slot_mappings=(slot_mapping,),
        block_size=cache_config.block_size,
    )

    assert block_table.tolist() == [[2, NULL_BLOCK_ID], [3, 5]]
    assert slot_mapping.tolist() == [8, 20]
    assert model_input.input_ids.device == block_table.device
    assert model_input.positions.device == block_table.device
    assert model_input.sampling_row_indices.device == block_table.device
