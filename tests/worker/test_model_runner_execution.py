"""Tests for ModelRunner lifecycle and immutable plan execution."""

from collections.abc import Callable

import pytest
import torch
from torch.nn import functional as F

from micro_vllm.attention.attention import Attention, AttentionMetadata
from micro_vllm.attention.spec import ChunkedLocalSpec, FullAttentionSpec
from micro_vllm.config import CacheConfig, LLMConfig, ModelConfig, SchedulerConfig
from micro_vllm.core.scheduler_output import (
    CacheGroupBlockTable,
    ScheduledRequest,
    SchedulerOutput,
)
from micro_vllm.sampling.params import SamplingParams
from micro_vllm.worker.model_runner import ModelRunner

NULL_BLOCK_ID = 7


class FakeModel(torch.nn.Module):
    """Expose flattened rows directly so sampling selection is observable."""

    def __init__(self) -> None:
        super().__init__()
        attention = Attention(FullAttentionSpec(), "reference")
        self.layer_attentions = (attention, attention)
        self.config = make_model_config()
        self.forward_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.logit_inputs: list[torch.Tensor] = []
        self.layer_kv_caches: tuple[torch.Tensor, ...] = ()
        self.layer_attention_metadata: tuple[AttentionMetadata, ...] = ()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        layer_kv_caches: tuple[torch.Tensor, ...],
        layer_attention_metadata: tuple[AttentionMetadata, ...],
    ) -> torch.Tensor:
        assert torch.is_inference_mode_enabled()
        self.forward_inputs.append((input_ids.clone(), positions.clone()))
        self.layer_kv_caches = layer_kv_caches
        self.layer_attention_metadata = layer_attention_metadata
        return torch.stack((input_ids, positions), dim=-1).to(torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert torch.is_inference_mode_enabled()
        self.logit_inputs.append(hidden_states.clone())
        return F.one_hot(
            hidden_states[:, 0].to(torch.int64),
            num_classes=64,
        ).to(torch.float32)


def make_model_config() -> ModelConfig:
    return ModelConfig(
        model="unused",
        metadata={
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 64,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 32,
        },
    )


def make_runner() -> tuple[ModelRunner, FakeModel, tuple[torch.Tensor, ...]]:
    model = FakeModel()
    cache_config = CacheConfig(block_size=2, num_blocks=8)
    runner = ModelRunner(
        lambda: model,
        LLMConfig(
            model_config=ModelConfig(
                model="unused",
                device="cpu",
                dtype=torch.float32,
            ),
            cache_config=cache_config,
            scheduler_config=SchedulerConfig(
                max_num_batched_tokens=8,
                max_num_seqs=2,
                max_model_len=32,
            ),
        ),
    )
    runner.initialize_model()
    assert runner._config.model_config is model.config
    runner.initialize_kv_cache()
    assert cache_config.num_blocks == 8
    cache = runner._kv_cache_buffers
    assert cache is not None
    model.forward_inputs.clear()
    model.logit_inputs.clear()
    return runner, model, cache


def make_uninitialized_runner(
    model_initializer: Callable[[], torch.nn.Module],
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    num_blocks: int | None = 8,
) -> ModelRunner:
    return ModelRunner(
        model_initializer,
        LLMConfig(
            model_config=ModelConfig(
                model="unused",
                device=device,
                dtype=dtype,
            ),
            cache_config=CacheConfig(
                block_size=2,
                num_blocks=num_blocks,
            ),
            scheduler_config=SchedulerConfig(
                max_num_batched_tokens=8,
                max_num_seqs=2,
                max_model_len=32,
            ),
        ),
    )


def test_model_runner_owns_model_warmup_and_cache_initialization() -> None:
    model = FakeModel()
    model.train()
    initialized: list[FakeModel] = []
    runner = make_uninitialized_runner(
        lambda: initialized.append(model) or model
    )
    cache_config = runner._config.cache_config

    assert initialized == []
    runner.initialize_model()
    assert runner._config.model_config is model.config
    assert initialized == [model]
    assert runner._model is model
    assert model.training is False
    topology = runner._kv_cache_topology
    assert topology is not None
    assert len(topology.groups) == 1
    assert topology.groups[0].attention_spec == FullAttentionSpec()
    assert topology.groups[0].layer_indices == (0, 1)
    assert topology.page_layer_count == 2
    with pytest.raises(RuntimeError, match="already initialized"):
        runner.initialize_model()

    warmup_plan, _ = runner._make_warmup_plan()
    assert warmup_plan.scheduled_requests
    assert all(
        not request.requires_sampling
        and request.sampling_params is None
        and request.sample_index is None
        for request in warmup_plan.scheduled_requests
    )

    runner.initialize_kv_cache()
    cache = runner._kv_cache_buffers
    assert cache_config.num_blocks == 8
    assert cache is not None
    layout = runner.kv_cache_layout
    assert layout.block_size == 2
    assert layout.num_blocks == 8
    assert layout.topology is topology
    assert len(cache) == layout.page_layer_count == 2
    assert len(model.forward_inputs) == 1
    assert model.forward_inputs[0][0].numel() == 8
    with pytest.raises(RuntimeError, match="already initialized"):
        runner.initialize_kv_cache()

    runner.shutdown()
    assert runner._model is None
    assert runner._model_initializer is None
    assert runner._kv_cache_layout is None
    assert runner._kv_cache_topology is None
    assert runner._kv_cache_buffers is None


def test_model_runner_derives_topology_from_layer_attentions() -> None:
    model = FakeModel()
    model.layer_attentions = (
        Attention(FullAttentionSpec(), "reference"),
        Attention(ChunkedLocalSpec(4), "reference"),
    )
    runner = make_uninitialized_runner(lambda: model)

    runner.initialize_model()

    topology = runner._kv_cache_topology
    assert topology is not None
    assert tuple(group.attention_spec for group in topology.groups) == (
        FullAttentionSpec(),
        ChunkedLocalSpec(4),
    )
    assert tuple(group.layer_indices for group in topology.groups) == (
        (0,),
        (1,),
    )
    assert topology.page_layer_count == 1


def test_model_runner_derives_gpu_block_count_from_available_bytes() -> None:
    model = FakeModel()
    runner = make_uninitialized_runner(
        lambda: model,
        num_blocks=None,
    )
    runner.initialize_model()
    bytes_per_block = 2 * 2 * 2 * 1 * 4 * 4

    assert runner._determine_num_blocks(
        available_memory_bytes=bytes_per_block * 20,
    ) == 20


def test_capacity_uses_max_group_layers_instead_of_total_layers() -> None:
    model = FakeModel()
    model.layer_attentions = (
        Attention(FullAttentionSpec(), "reference"),
        Attention(ChunkedLocalSpec(4), "reference"),
    )
    runner = make_uninitialized_runner(
        lambda: model,
        num_blocks=None,
    )
    runner.initialize_model()
    bytes_per_block = 1 * 2 * 2 * 1 * 4 * 4

    assert runner._determine_num_blocks(
        available_memory_bytes=bytes_per_block * 20,
    ) == 20


def test_layer_binding_reuses_group_layer_index_buffer_addresses() -> None:
    model = FakeModel()
    full = Attention(FullAttentionSpec(), "reference")
    chunked = Attention(ChunkedLocalSpec(4), "reference")
    model.layer_attentions = (full, chunked, full, chunked)
    model.config.set_metadata(
        {
            **model.config.values,
            "num_hidden_layers": 4,
        }
    )
    runner = make_uninitialized_runner(lambda: model)

    runner.initialize_model()
    runner.initialize_kv_cache()

    runner.execute(
        SchedulerOutput(
            (
                ScheduledRequest(
                    request_id="request-1",
                    input_token_ids=(1,),
                    num_computed_tokens=0,
                    num_scheduled_tokens=1,
                    requires_sampling=False,
                    block_tables=(
                        CacheGroupBlockTable((0,)),
                        CacheGroupBlockTable((1,)),
                    ),
                ),
            )
        )
    )

    cache = runner._kv_cache_buffers
    assert cache is not None
    assert runner.kv_cache_layout.page_layer_count == 2
    assert len(cache) == 2
    assert len(model.layer_kv_caches) == 4
    assert len(model.layer_attention_metadata) == 4
    assert model.layer_kv_caches[0] is cache[0]
    assert model.layer_kv_caches[1] is cache[0]
    assert model.layer_kv_caches[2] is cache[1]
    assert model.layer_kv_caches[3] is cache[1]
    full_metadata = model.layer_attention_metadata[0]
    chunked_metadata = model.layer_attention_metadata[1]
    assert model.layer_attention_metadata[2] is full_metadata
    assert model.layer_attention_metadata[3] is chunked_metadata
    assert chunked_metadata is not full_metadata
    assert full_metadata.query_positions is chunked_metadata.query_positions
    assert full_metadata.block_table.tolist() == [[0]]
    assert chunked_metadata.block_table.tolist() == [[1]]
    assert full_metadata.slot_mapping.tolist() == [0]
    assert chunked_metadata.slot_mapping.tolist() == [2]


def test_gpu_capacity_must_fit_data_and_null_pages() -> None:
    model = FakeModel()
    runner = make_uninitialized_runner(lambda: model, num_blocks=None)
    runner.initialize_model()
    bytes_per_block = 2 * 2 * 2 * 1 * 4 * 4

    with pytest.raises(MemoryError, match="data and null KV cache blocks"):
        runner._determine_num_blocks(
            available_memory_bytes=bytes_per_block,
        )


def test_cpu_kv_cache_initialization_requires_deterministic_capacity() -> None:
    model = FakeModel()
    runner = make_uninitialized_runner(
        lambda: model,
        num_blocks=None,
    )
    runner.initialize_model()

    with pytest.raises(ValueError, match="CPU execution requires num_blocks"):
        runner.initialize_kv_cache()


def test_model_runner_does_not_fallback_when_placement_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel()

    def fail_placement(*, device: torch.device, dtype: torch.dtype) -> None:
        raise RuntimeError(f"cannot move model to {device} with {dtype}")

    monkeypatch.setattr(model, "to", fail_placement)
    runner = make_uninitialized_runner(
        lambda: model,
        device="cuda",
        dtype=torch.float16,
    )

    with pytest.raises(RuntimeError, match="cannot move model to cuda"):
        runner.initialize_model()
    assert runner._model is None


def make_request(
    request_id: str,
    input_token_ids: tuple[int, ...],
    *,
    num_computed_tokens: int,
    requires_sampling: bool,
    block_ids: tuple[int, ...],
) -> ScheduledRequest:
    return ScheduledRequest(
        request_id=request_id,
        input_token_ids=input_token_ids,
        num_computed_tokens=num_computed_tokens,
        num_scheduled_tokens=len(input_token_ids),
        requires_sampling=requires_sampling,
        block_tables=(CacheGroupBlockTable(block_ids),),
        sampling_params=(
            SamplingParams(max_tokens=4) if requires_sampling else None
        ),
        sample_index=0 if requires_sampling else None,
    )


def test_mixed_batch_selects_sampling_rows_and_preserves_plan() -> None:
    runner, model, cache = make_runner()
    plan = SchedulerOutput(
        (
            make_request(
                "first-sampled",
                (7,),
                num_computed_tokens=2,
                requires_sampling=True,
                block_ids=(6, 1),
            ),
            make_request(
                "middle-prefill",
                (20, 21),
                num_computed_tokens=0,
                requires_sampling=False,
                block_ids=(4,),
            ),
            make_request(
                "last-sampled",
                (31, 32),
                num_computed_tokens=3,
                requires_sampling=True,
                block_ids=(3, 5, 2),
            ),
        )
    )
    scheduled_requests = plan.scheduled_requests
    input_token_ids = tuple(
        request.input_token_ids for request in plan.scheduled_requests
    )
    block_tables = tuple(
        request.block_tables for request in plan.scheduled_requests
    )

    first_output = runner.execute(plan)
    second_output = runner.execute(plan)

    assert first_output == second_output
    assert [
        (sample.request_id, sample.token_id)
        for sample in first_output.sampled_tokens
    ] == [("first-sampled", 7), ("last-sampled", 32)]
    assert model.logit_inputs[-1].tolist() == [[7.0, 2.0], [32.0, 4.0]]
    assert len(model.layer_kv_caches) == len(cache) == 2
    assert all(
        actual is cache[layer_index]
        for layer_index, actual in enumerate(model.layer_kv_caches)
    )

    metadata = model.layer_attention_metadata[0]
    assert metadata.query_positions.tolist() == [2, 0, 1, 3, 4]
    assert metadata.query_start_locations.tolist() == [0, 1, 3, 5]
    assert metadata.context_lengths.tolist() == [3, 2, 5]
    assert metadata.block_table.tolist() == [
        [6, 1, NULL_BLOCK_ID],
        [4, NULL_BLOCK_ID, NULL_BLOCK_ID],
        [3, 5, 2],
    ]
    assert metadata.slot_mapping.tolist() == [2, 8, 9, 11, 4]
    assert metadata.query_positions.device == torch.device("cpu")

    assert plan.scheduled_requests is scheduled_requests
    assert tuple(
        request.input_token_ids for request in plan.scheduled_requests
    ) == input_token_ids
    assert tuple(
        request.block_tables for request in plan.scheduled_requests
    ) == block_tables


def test_batch_without_sampling_still_executes_model() -> None:
    runner, model, _ = make_runner()
    plan = SchedulerOutput(
        (
            make_request(
                "prefill",
                (10, 11),
                num_computed_tokens=0,
                requires_sampling=False,
                block_ids=(0,),
            ),
        )
    )

    output = runner.execute(plan)

    assert output.sampled_tokens == ()
    assert len(model.forward_inputs) == 1
    assert model.logit_inputs[0].shape == torch.Size((0, 2))


def test_batch_with_all_requests_sampling_preserves_request_order() -> None:
    runner, _, _ = make_runner()
    plan = SchedulerOutput(
        (
            make_request(
                "second-id",
                (12,),
                num_computed_tokens=0,
                requires_sampling=True,
                block_ids=(2,),
            ),
            make_request(
                "first-id",
                (22, 23),
                num_computed_tokens=0,
                requires_sampling=True,
                block_ids=(7,),
            ),
        )
    )

    output = runner.execute(plan)

    assert [sample.request_id for sample in output.sampled_tokens] == [
        "second-id",
        "first-id",
    ]
    assert [sample.token_id for sample in output.sampled_tokens] == [12, 23]
