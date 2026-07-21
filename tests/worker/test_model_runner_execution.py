"""Tests for executing immutable scheduler plans through ModelRunner."""

import torch
from torch.nn import functional as F

from micro_vllm.config import CacheConfig
from micro_vllm.core.scheduler_output import ScheduledRequest, SchedulerOutput
from micro_vllm.layers.attention_backend import AttentionMetadata, PAD_BLOCK_ID
from micro_vllm.config import ModelConfig
from micro_vllm.worker.config import ModelExecutionConfig
from micro_vllm.worker.kv_cache import PhysicalKVCache
from micro_vllm.worker.model_runner import ModelRunner


class FakeModel:
    """Expose flattened rows directly so sampling selection is observable."""

    def __init__(self) -> None:
        self.forward_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.logit_inputs: list[torch.Tensor] = []
        self.layer_kv_caches: tuple[torch.Tensor, ...] = ()
        self.metadata: AttentionMetadata | None = None

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        layer_kv_caches: tuple[torch.Tensor, ...],
        attention_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        assert torch.is_inference_mode_enabled()
        self.forward_inputs.append((input_ids.clone(), positions.clone()))
        self.layer_kv_caches = layer_kv_caches
        self.metadata = attention_metadata
        return torch.stack((input_ids, positions), dim=-1).to(torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert torch.is_inference_mode_enabled()
        self.logit_inputs.append(hidden_states.clone())
        return F.one_hot(
            hidden_states[:, 0].to(torch.int64),
            num_classes=64,
        ).to(torch.float32)


def make_runner() -> tuple[ModelRunner, FakeModel, PhysicalKVCache]:
    device = torch.device("cpu")
    dtype = torch.float32
    cache = PhysicalKVCache(
        CacheConfig(block_size=2, num_blocks=8),
        ModelConfig(
            {
                "architectures": ["LlamaForCausalLM"],
                "vocab_size": 64,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "max_position_embeddings": 32,
            }
        ),
        ModelExecutionConfig(device=device, dtype=dtype),
    )
    model = FakeModel()
    return (
        ModelRunner(
            model,
            cache,
            device=device,
            dtype=dtype,
            block_size=2,
        ),
        model,
        cache,
    )


def make_request(
    request_id: str,
    input_token_ids: tuple[int, ...],
    *,
    num_computed_tokens: int,
    requires_sampling: bool,
    block_table: tuple[int | None, ...],
) -> ScheduledRequest:
    return ScheduledRequest(
        request_id=request_id,
        input_token_ids=input_token_ids,
        num_computed_tokens=num_computed_tokens,
        num_scheduled_tokens=len(input_token_ids),
        requires_sampling=requires_sampling,
        block_table=block_table,
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
                block_table=(6, 1),
            ),
            make_request(
                "middle-prefill",
                (20, 21),
                num_computed_tokens=0,
                requires_sampling=False,
                block_table=(4,),
            ),
            make_request(
                "last-sampled",
                (31, 32),
                num_computed_tokens=3,
                requires_sampling=True,
                block_table=(3, 5, 2),
            ),
        )
    )
    scheduled_requests = plan.scheduled_requests
    input_token_ids = tuple(
        request.input_token_ids for request in plan.scheduled_requests
    )
    block_tables = tuple(request.block_table for request in plan.scheduled_requests)

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

    assert model.metadata is not None
    assert model.metadata.query_positions.tolist() == [2, 0, 1, 3, 4]
    assert model.metadata.query_start_locations.tolist() == [0, 1, 3, 5]
    assert model.metadata.context_lengths.tolist() == [3, 2, 5]
    assert model.metadata.block_table.tolist() == [
        [6, 1, PAD_BLOCK_ID],
        [4, PAD_BLOCK_ID, PAD_BLOCK_ID],
        [3, 5, 2],
    ]
    assert model.metadata.slot_mapping.tolist() == [2, 8, 9, 11, 4]
    assert model.metadata.query_positions.device == torch.device("cpu")

    assert plan.scheduled_requests is scheduled_requests
    assert tuple(
        request.input_token_ids for request in plan.scheduled_requests
    ) == input_token_ids
    assert tuple(
        request.block_table for request in plan.scheduled_requests
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
                block_table=(0,),
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
                block_table=(2,),
            ),
            make_request(
                "first-id",
                (22, 23),
                num_computed_tokens=0,
                requires_sampling=True,
                block_table=(7,),
            ),
        )
    )

    output = runner.execute(plan)

    assert [sample.request_id for sample in output.sampled_tokens] == [
        "second-id",
        "first-id",
    ]
    assert [sample.token_id for sample in output.sampled_tokens] == [12, 23]
