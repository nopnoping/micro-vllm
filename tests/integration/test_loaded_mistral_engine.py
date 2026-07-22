"""Loader and Engine integration tests for tiny HF Mistral checkpoints."""

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from transformers import MistralForCausalLM

from micro_vllm.attention.spec import SlidingWindowSpec
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
from micro_vllm.model_loader import load_model
from micro_vllm.models.mistral import MistralModel
from micro_vllm.sampling.params import SamplingParams
from micro_vllm.worker.worker import Worker

PUBLIC_MISTRAL_REPO = (
    "sanchit-gandhi/tiny-random-MistralForCausalLM-1-layer"
)
PUBLIC_MISTRAL_REVISION = "53f378eafc3840b0a33e33ea5137e5978ffe4434"


class RecordingExecutor(SingleProcessExecutor):
    __slots__ = ("plans",)

    def __init__(self, worker: Worker) -> None:
        super().__init__(worker)
        self.plans: list[SchedulerOutput] = []

    def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput:
        result = super().execute(plan)
        self.plans.append(plan)
        return result


@dataclass(slots=True)
class MistralHarness:
    engine: Engine
    scheduler: Scheduler
    cache_manager: KVCacheManager
    worker: Worker
    executor: RecordingExecutor


def make_harness(
    model_source: str | Path,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    token_budget: int = 8,
    block_size: int = 2,
    num_blocks: int = 9,
) -> MistralHarness:
    config = LLMConfig(
        model_config=ModelConfig(
            model=model_source,
            revision=revision,
            local_files_only=local_files_only,
            device=device,
            dtype=dtype,
        ),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=token_budget,
            max_num_seqs=1,
            max_model_len=16,
        ),
        cache_config=CacheConfig(
            block_size=block_size,
            num_blocks=num_blocks,
        ),
    )
    worker = Worker(lambda: load_model(config), config)
    cache_manager = KVCacheManager(config, worker.kv_cache_layout)
    scheduler = Scheduler(config, cache_manager)
    executor = RecordingExecutor(worker)
    return MistralHarness(
        engine=Engine(scheduler, executor),
        scheduler=scheduler,
        cache_manager=cache_manager,
        worker=worker,
        executor=executor,
    )


def run_to_completion(harness: MistralHarness) -> None:
    for _ in range(32):
        if not harness.engine.has_unfinished_requests():
            return
        harness.engine.step()
    raise AssertionError("engine did not finish within 32 steps")


def output_tokens(
    harness: MistralHarness,
    request_id: str,
) -> tuple[int, ...]:
    request = harness.scheduler.get_request(request_id)
    assert request is not None
    return tuple(request.output_token_ids)


@torch.no_grad()
def reference_greedy_tokens(
    reference: MistralForCausalLM,
    prompt: tuple[int, ...],
    num_output_tokens: int,
) -> tuple[int, ...]:
    token_ids = list(prompt)
    output: list[int] = []
    for _ in range(num_output_tokens):
        logits = reference(
            torch.tensor([token_ids], dtype=torch.int64),
            use_cache=False,
        ).logits[:, -1]
        token_id = int(logits.argmax(dim=-1).item())
        output.append(token_id)
        token_ids.append(token_id)
    return tuple(output)


def test_load_model_restores_tiny_mistral_checkpoint(
    tiny_hf_mistral_checkpoint: tuple[MistralForCausalLM, Path],
) -> None:
    reference, checkpoint_path = tiny_hf_mistral_checkpoint

    model = load_model(
        LLMConfig(model_config=ModelConfig(model=checkpoint_path))
    )

    assert isinstance(model, MistralModel)
    assert model.layer_attentions[0].spec == SlidingWindowSpec(4)
    expected_state = {
        (
            parameter_name
            if parameter_name == "lm_head.weight"
            else parameter_name.removeprefix("model.")
        ): parameter
        for parameter_name, parameter in reference.state_dict().items()
    }
    assert set(model.state_dict()) == set(expected_state)
    for parameter_name, parameter in model.state_dict().items():
        torch.testing.assert_close(
            parameter,
            expected_state[parameter_name],
            rtol=0.0,
            atol=0.0,
        )


def test_loaded_mistral_full_chunked_and_decode_match_transformers(
    tiny_hf_mistral_checkpoint: tuple[MistralForCausalLM, Path],
) -> None:
    reference, checkpoint_path = tiny_hf_mistral_checkpoint
    prompt = (1, 3, 5, 7, 9, 11)
    full = make_harness(checkpoint_path, token_budget=8)
    chunked = make_harness(checkpoint_path, token_budget=2)
    for harness in (full, chunked):
        harness.engine.add_request(
            "request-1",
            list(prompt),
            SamplingParams(max_tokens=2),
        )
        run_to_completion(harness)

    expected = reference_greedy_tokens(reference, prompt, 2)
    assert output_tokens(full, "request-1") == expected
    assert output_tokens(chunked, "request-1") == expected
    assert [
        plan.scheduled_requests[0].num_scheduled_tokens
        for plan in full.executor.plans
    ] == [6, 1]
    assert [
        plan.scheduled_requests[0].num_scheduled_tokens
        for plan in chunked.executor.plans
    ] == [2, 2, 2, 1]


def test_loaded_mistral_releases_window_blocks_and_reuses_pages(
    tiny_hf_mistral_checkpoint: tuple[MistralForCausalLM, Path],
) -> None:
    reference, checkpoint_path = tiny_hf_mistral_checkpoint
    harness = make_harness(
        checkpoint_path,
        token_budget=8,
        block_size=2,
        num_blocks=5,
    )
    harness.engine.add_request(
        "old",
        [1, 2, 3, 4, 5, 6],
        SamplingParams(max_tokens=2),
    )

    harness.engine.step()

    null_block_id = harness.cache_manager._block_pool.null_block_id
    (table,) = harness.cache_manager.get_block_tables("old")
    initial_table = harness.executor.plans[0].scheduled_requests[0].block_tables[
        0
    ].block_ids
    assert table == (null_block_id, initial_table[1], initial_table[2])
    assert harness.cache_manager._block_pool.ref_count(initial_table[0]) == 0
    assert tuple(
        harness.cache_manager._block_pool.ref_count(block_id)
        for block_id in initial_table[1:]
    ) == (1, 1)

    run_to_completion(harness)
    assert harness.cache_manager.num_free_blocks == 4
    harness.engine.add_request("new", [7], SamplingParams(max_tokens=1))
    run_to_completion(harness)

    expected = reference_greedy_tokens(reference, (7,), 1)
    assert output_tokens(harness, "new") == expected
    assert harness.cache_manager.num_free_blocks == 4


@pytest.mark.model
def test_public_mistral_checkpoint_reloads_from_standard_hub_cache() -> None:
    reference = MistralForCausalLM.from_pretrained(
        PUBLIC_MISTRAL_REPO,
        revision=PUBLIC_MISTRAL_REVISION,
        local_files_only=True,
    ).eval()
    harness = make_harness(
        PUBLIC_MISTRAL_REPO,
        revision=PUBLIC_MISTRAL_REVISION,
        local_files_only=True,
        token_budget=8,
    )
    prompt = (1, 3, 5, 7)
    harness.engine.add_request(
        "request-1",
        list(prompt),
        SamplingParams(max_tokens=2),
    )

    run_to_completion(harness)

    assert output_tokens(harness, "request-1") == reference_greedy_tokens(
        reference,
        prompt,
        2,
    )
    reloaded = load_model(
        LLMConfig(
            model_config=ModelConfig(
                model=PUBLIC_MISTRAL_REPO,
                revision=PUBLIC_MISTRAL_REVISION,
                local_files_only=True,
            )
        )
    )
    assert isinstance(reloaded, MistralModel)
    assert reloaded.layer_attentions[0].spec == SlidingWindowSpec(4096)


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_public_mistral_checkpoint_runs_with_fixed_t4_budget() -> None:
    harness = make_harness(
        PUBLIC_MISTRAL_REPO,
        revision=PUBLIC_MISTRAL_REVISION,
        local_files_only=True,
        device="cuda",
        dtype=torch.float16,
        token_budget=8,
        block_size=4,
        num_blocks=64,
    )
    try:
        harness.engine.add_request(
            "request-1",
            [1, 3, 5, 7],
            SamplingParams(max_tokens=1),
        )

        run_to_completion(harness)

        assert len(output_tokens(harness, "request-1")) == 1
        assert harness.worker.kv_cache_layout.num_blocks == 64
        assert harness.worker.kv_cache_layout.page_layer_count == 1
    finally:
        harness.executor.shutdown()
