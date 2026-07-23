"""End-to-end correctness tests for hybrid attention cache groups."""

from dataclasses import dataclass

import pytest
import torch

import micro_vllm.worker.model_runner as model_runner_module
from micro_vllm.attention.attention import Attention, AttentionMetadata
from micro_vllm.attention.spec import (
    AttentionSpec,
    ChunkedLocalSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
)
from micro_vllm.config import (
    CacheConfig,
    LLMConfig,
    ModelConfig,
    SchedulerConfig,
)
from micro_vllm.core.kv_cache_manager import KVCacheManager
from micro_vllm.core.scheduler import Scheduler
from micro_vllm.core.scheduler_output import SchedulerOutput
from micro_vllm.engine.engine import Engine
from micro_vllm.executor.interface import ModelRunnerOutput
from micro_vllm.executor.single_process import SingleProcessExecutor
from micro_vllm.models.llama import LlamaModel
from micro_vllm.sampling.params import SamplingParams
from micro_vllm.worker.worker import Worker

_MODELS_TO_LOAD: list["HybridLlamaModel"] = []


@pytest.fixture(autouse=True)
def stub_model_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    _MODELS_TO_LOAD.clear()
    monkeypatch.setattr(
        model_runner_module,
        "load_model",
        lambda _: _MODELS_TO_LOAD.pop(0),
    )


class HybridLlamaModel(LlamaModel):
    """Two-layer Llama whose layers may use different cache semantics."""

    def __init__(
        self,
        config: ModelConfig,
        attention_specs: tuple[AttentionSpec, ...],
    ) -> None:
        super().__init__(config)
        attentions = tuple(
            Attention(spec, config.attention_operator)
            for spec in attention_specs
        )
        self.layer_attentions = attentions
        for layer, attention in zip(
            self.layers,
            attentions,
            strict=True,
        ):
            layer.self_attn.attention = attention
        self.hidden_batches: list[torch.Tensor] = []
        self.logit_batches: list[torch.Tensor] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        layer_kv_caches: tuple[torch.Tensor, ...],
        layer_attention_metadata: tuple[AttentionMetadata, ...],
    ) -> torch.Tensor:
        hidden_states = super().forward(
            input_ids,
            positions,
            layer_kv_caches,
            layer_attention_metadata,
        )
        self.hidden_batches.append(hidden_states.detach().clone())
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = super().compute_logits(hidden_states)
        self.logit_batches.append(logits.detach().clone())
        return logits


class RecordingExecutor(SingleProcessExecutor):
    """Retain immutable plans around the real worker execution path."""

    __slots__ = ("plans",)

    def __init__(self, worker: Worker) -> None:
        super().__init__(worker)
        self.plans: list[SchedulerOutput] = []

    def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput:
        result = super().execute(plan)
        self.plans.append(plan)
        return result


@dataclass(slots=True)
class HybridHarness:
    engine: Engine
    scheduler: Scheduler
    cache_manager: KVCacheManager
    executor: RecordingExecutor
    model: HybridLlamaModel


def make_model_config() -> ModelConfig:
    return ModelConfig(
        model="unused",
        device="cpu",
        dtype=torch.float32,
        metadata={
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 32,
        },
    )


def make_hybrid_engine(
    bounded_spec: AttentionSpec,
    *,
    token_budget: int = 8,
    sequence_budget: int = 2,
    block_size: int = 2,
    num_blocks: int = 17,
) -> HybridHarness:
    attention_specs = (FullAttentionSpec(), bounded_spec)
    torch.manual_seed(0)
    model = HybridLlamaModel(make_model_config(), attention_specs)
    config = LLMConfig(
        model_config=ModelConfig(
            model="unused",
            device="cpu",
            dtype=torch.float32,
        ),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=token_budget,
            max_num_seqs=sequence_budget,
            max_model_len=16,
        ),
        cache_config=CacheConfig(
            block_size=block_size,
            num_blocks=num_blocks,
        ),
    )
    _MODELS_TO_LOAD.append(model)
    worker = Worker(config)
    cache_manager = KVCacheManager(config, worker.kv_cache_layout)
    scheduler = Scheduler(config, cache_manager)
    executor = RecordingExecutor(worker)
    model.hidden_batches.clear()
    model.logit_batches.clear()
    return HybridHarness(
        engine=Engine(scheduler, executor),
        scheduler=scheduler,
        cache_manager=cache_manager,
        executor=executor,
        model=model,
    )


def run_to_completion(harness: HybridHarness) -> None:
    for _ in range(32):
        if not harness.engine.has_unfinished_requests():
            return
        harness.engine.step()
    raise AssertionError("engine did not finish within 32 steps")


def output_tokens(
    harness: HybridHarness,
    request_id: str,
) -> tuple[int, ...]:
    request = harness.scheduler.get_request(request_id)
    assert request is not None
    return tuple(request.output_token_ids)


def sampled_hidden_rows(harness: HybridHarness) -> tuple[torch.Tensor, ...]:
    rows: list[torch.Tensor] = []
    for plan, hidden_states in zip(
        harness.executor.plans,
        harness.model.hidden_batches,
        strict=True,
    ):
        row_start = 0
        for request in plan.scheduled_requests:
            row_end = row_start + request.num_scheduled_tokens
            if request.requires_sampling:
                rows.append(hidden_states[row_end - 1])
            row_start = row_end
    return tuple(rows)


def sampled_logits(harness: HybridHarness) -> tuple[torch.Tensor, ...]:
    return tuple(
        logits
        for logits in harness.model.logit_batches
        if logits.shape[0] > 0
    )


def dense_greedy_reference(
    bounded_spec: AttentionSpec,
    prompt_token_ids: tuple[int, ...],
    num_output_tokens: int,
) -> tuple[
    tuple[int, ...],
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
]:
    """Run the same weights over contiguous per-layer cache tensors."""

    attention_specs = (FullAttentionSpec(), bounded_spec)
    torch.manual_seed(0)
    model = HybridLlamaModel(make_model_config(), attention_specs)
    block_size = 2
    num_positions = len(prompt_token_ids) + num_output_tokens
    num_blocks = (num_positions + block_size - 1) // block_size
    num_key_value_heads = 2
    head_dim = 4
    layer_kv_caches = tuple(
        torch.empty(
            2,
            num_blocks,
            block_size,
            num_key_value_heads,
            head_dim,
            dtype=torch.float32,
        )
        for _ in attention_specs
    )
    block_table = torch.arange(
        num_blocks,
        dtype=torch.int64,
    ).unsqueeze(0)
    generated_token_ids: list[int] = []
    sampled_hidden_states: list[torch.Tensor] = []
    sampled_logit_rows: list[torch.Tensor] = []

    with torch.inference_mode():
        for output_index in range(num_output_tokens):
            if output_index == 0:
                input_token_ids = prompt_token_ids
                first_position = 0
            else:
                input_token_ids = (generated_token_ids[-1],)
                first_position = len(prompt_token_ids) + output_index - 1
            input_ids = torch.tensor(input_token_ids, dtype=torch.int64)
            positions = torch.arange(
                first_position,
                first_position + len(input_token_ids),
                dtype=torch.int64,
            )
            context_length = first_position + len(input_token_ids)
            layer_metadata = tuple(
                attention.build_metadata(
                    query_positions=positions,
                    query_start_locations=torch.tensor(
                        [0, len(input_token_ids)],
                        dtype=torch.int64,
                    ),
                    context_lengths=torch.tensor(
                        [context_length],
                        dtype=torch.int64,
                    ),
                    block_table=block_table,
                    slot_mapping=positions,
                    block_size=block_size,
                )
                for attention in model.layer_attentions
            )
            hidden_states = model(
                input_ids,
                positions,
                layer_kv_caches,
                layer_metadata,
            )
            logits = model.compute_logits(hidden_states[-1:])
            generated_token_ids.append(int(logits.argmax(dim=-1).item()))
            sampled_hidden_states.append(hidden_states[-1].detach().clone())
            sampled_logit_rows.append(logits.detach().clone())

    return (
        tuple(generated_token_ids),
        tuple(sampled_hidden_states),
        tuple(sampled_logit_rows),
    )


BOUNDED_SPECS = (
    pytest.param(ChunkedLocalSpec(4), id="chunked-local"),
    pytest.param(SlidingWindowSpec(3), id="sliding-window"),
)


@pytest.mark.parametrize("bounded_spec", BOUNDED_SPECS)
def test_hybrid_chunked_prefill_and_decode_match_dense_reference(
    bounded_spec: AttentionSpec,
) -> None:
    prompt = (1, 3, 5, 7, 9)
    full_prefill = make_hybrid_engine(
        bounded_spec,
        token_budget=8,
        sequence_budget=1,
    )
    chunked_prefill = make_hybrid_engine(
        bounded_spec,
        token_budget=2,
        sequence_budget=1,
    )
    for harness in (full_prefill, chunked_prefill):
        harness.engine.add_request(
            "target",
            list(prompt),
            SamplingParams(max_tokens=2),
        )
        run_to_completion(harness)

    reference_tokens, reference_hidden, reference_logits = (
        dense_greedy_reference(bounded_spec, prompt, 2)
    )
    assert output_tokens(full_prefill, "target") == reference_tokens
    assert output_tokens(chunked_prefill, "target") == reference_tokens
    assert [
        plan.scheduled_requests[0].num_scheduled_tokens
        for plan in full_prefill.executor.plans
    ] == [5, 1]
    assert [
        plan.scheduled_requests[0].num_scheduled_tokens
        for plan in chunked_prefill.executor.plans
    ] == [2, 2, 1, 1]

    for actual, expected in zip(
        sampled_hidden_rows(full_prefill),
        reference_hidden,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(
        sampled_hidden_rows(chunked_prefill),
        reference_hidden,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(
        sampled_logits(full_prefill),
        reference_logits,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(
        sampled_logits(chunked_prefill),
        reference_logits,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("bounded_spec", BOUNDED_SPECS)
def test_hybrid_mixed_batch_matches_independent_execution(
    bounded_spec: AttentionSpec,
) -> None:
    mixed = make_hybrid_engine(bounded_spec, token_budget=4)
    mixed.engine.add_request("decode", [1], SamplingParams(max_tokens=2))
    mixed.engine.step()
    mixed.engine.add_request(
        "prefill",
        [3, 4, 5],
        SamplingParams(max_tokens=1),
    )
    run_to_completion(mixed)

    decode_only = make_hybrid_engine(
        bounded_spec,
        token_budget=4,
        sequence_budget=1,
    )
    decode_only.engine.add_request(
        "decode",
        [1],
        SamplingParams(max_tokens=2),
    )
    run_to_completion(decode_only)
    prefill_only = make_hybrid_engine(
        bounded_spec,
        token_budget=4,
        sequence_budget=1,
    )
    prefill_only.engine.add_request(
        "prefill",
        [3, 4, 5],
        SamplingParams(max_tokens=1),
    )
    run_to_completion(prefill_only)

    assert output_tokens(mixed, "decode") == output_tokens(
        decode_only,
        "decode",
    )
    assert output_tokens(mixed, "prefill") == output_tokens(
        prefill_only,
        "prefill",
    )
    assert [
        request.request_id
        for request in mixed.executor.plans[1].scheduled_requests
    ] == ["decode", "prefill"]


@pytest.mark.parametrize("bounded_spec", BOUNDED_SPECS)
def test_bounded_group_releases_history_while_full_group_retains_it(
    bounded_spec: AttentionSpec,
) -> None:
    harness = make_hybrid_engine(
        bounded_spec,
        token_budget=8,
        sequence_budget=1,
        block_size=2,
        num_blocks=9,
    )
    harness.engine.add_request(
        "request-1",
        [1, 2, 3, 4, 5, 6],
        SamplingParams(max_tokens=2),
    )

    harness.engine.step()

    null_block_id = harness.cache_manager._block_pool.null_block_id
    full_table, bounded_table = harness.cache_manager.get_block_tables(
        "request-1"
    )
    scheduled_tables = harness.executor.plans[0].scheduled_requests[0].block_tables
    released_block_ids = scheduled_tables[1].block_ids[:2]
    assert len(full_table) == len(bounded_table) == 3
    assert null_block_id not in full_table
    assert bounded_table[:2] == (null_block_id, null_block_id)
    assert bounded_table[2] != null_block_id
    assert set(full_table).isdisjoint({bounded_table[2]})
    assert tuple(
        harness.cache_manager._block_pool.ref_count(block_id)
        for block_id in full_table
    ) == (1, 1, 1)
    assert tuple(
        harness.cache_manager._block_pool.ref_count(block_id)
        for block_id in released_block_ids
    ) == (0, 0)

    assert harness.engine.abort_request("request-1") is True
    assert harness.cache_manager.num_free_blocks == 8
    with pytest.raises(KeyError):
        harness.cache_manager.get_block_tables("request-1")


@pytest.mark.parametrize("bounded_spec", BOUNDED_SPECS)
def test_hybrid_preemption_releases_every_group_before_recompute(
    bounded_spec: AttentionSpec,
) -> None:
    harness = make_hybrid_engine(
        bounded_spec,
        token_budget=4,
        sequence_budget=2,
        block_size=2,
        num_blocks=5,
    )
    for request_id, prompt in (("first", [1, 2]), ("second", [3, 4])):
        harness.engine.add_request(
            request_id,
            prompt,
            SamplingParams(max_tokens=2),
        )
    harness.engine.step()

    harness.engine.step()

    preemption_plan = harness.executor.plans[1]
    assert preemption_plan.preempted_request_ids == ("second",)
    assert [
        request.request_id for request in preemption_plan.scheduled_requests
    ] == ["first"]
    assert harness.cache_manager.get_block_tables("second") == ((), ())
    assert harness.cache_manager.num_free_blocks == 4
    assert tuple(
        harness.cache_manager._block_pool.ref_count(block_id)
        for block_id in range(4)
    ) == (0, 0, 0, 0)

    run_to_completion(harness)

    resumed = harness.executor.plans[2].scheduled_requests[0]
    assert resumed.request_id == "second"
    assert resumed.num_computed_tokens == 0
    assert resumed.num_scheduled_tokens == 3
    assert harness.cache_manager.num_free_blocks == 4
    assert tuple(
        harness.cache_manager._block_pool.ref_count(block_id)
        for block_id in range(4)
    ) == (0, 0, 0, 0)


@pytest.mark.parametrize("bounded_spec", BOUNDED_SPECS)
def test_reused_pages_do_not_leak_stale_hybrid_kv(
    bounded_spec: AttentionSpec,
) -> None:
    reused = make_hybrid_engine(
        bounded_spec,
        token_budget=4,
        sequence_budget=1,
        block_size=4,
        num_blocks=3,
    )
    reused.engine.add_request(
        "old",
        [1, 2, 3],
        SamplingParams(max_tokens=1),
    )
    run_to_completion(reused)
    reused.engine.add_request("new", [7], SamplingParams(max_tokens=1))
    run_to_completion(reused)

    fresh = make_hybrid_engine(
        bounded_spec,
        token_budget=4,
        sequence_budget=1,
        block_size=4,
        num_blocks=3,
    )
    fresh.engine.add_request("new", [7], SamplingParams(max_tokens=1))
    run_to_completion(fresh)

    old_tables = reused.executor.plans[0].scheduled_requests[0].block_tables
    new_tables = reused.executor.plans[1].scheduled_requests[0].block_tables
    assert tuple(table.block_ids for table in old_tables) == ((0,), (1,))
    assert tuple(table.block_ids for table in new_tables) == ((0,), (1,))
    assert output_tokens(reused, "new") == output_tokens(fresh, "new")
    assert reused.cache_manager.num_free_blocks == 2
