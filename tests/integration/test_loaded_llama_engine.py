"""End-to-end Engine semantics for a loaded HF-format Llama checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from transformers import LlamaForCausalLM

from micro_vllm.config import CacheConfig, LLMConfig, ModelConfig, SchedulerConfig
from micro_vllm.core.kv_cache_manager import KVCacheManager
from micro_vllm.core.request import FinishReason
from micro_vllm.core.scheduler import Scheduler
from micro_vllm.core.scheduler_output import SchedulerOutput
from micro_vllm.engine.engine import Engine
from micro_vllm.engine.output import RequestOutput
from micro_vllm.executor.interface import ModelRunnerOutput
from micro_vllm.executor.single_process import SingleProcessExecutor
from micro_vllm.model_loader import load_model
from micro_vllm.models.llama import LlamaModel
from micro_vllm.sampling.params import SamplingParams
from micro_vllm.worker.worker import Worker


class RecordingSingleProcessExecutor(SingleProcessExecutor):
    """Record plans while retaining the real loaded-model execution path."""

    __slots__ = ("plans",)

    def __init__(self, worker: Worker) -> None:
        super().__init__(worker)
        self.plans: list[SchedulerOutput] = []

    def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput:
        output = super().execute(plan)
        self.plans.append(plan)
        return output


@dataclass(slots=True)
class LoadedEngineHarness:
    engine: Engine
    scheduler: Scheduler
    cache_manager: KVCacheManager
    worker: Worker
    executor: RecordingSingleProcessExecutor
    load_calls: list[Path]


def make_loaded_engine(
    checkpoint_path: Path,
    *,
    token_budget: int = 8,
    sequence_budget: int = 2,
    max_model_len: int = 16,
    block_size: int = 2,
    num_blocks: int = 16,
    enable_prefix_cache: bool = False,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> LoadedEngineHarness:
    config = LLMConfig(
        model_config=ModelConfig(
            model=checkpoint_path,
            device=device,
            dtype=dtype,
        ),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=token_budget,
            max_num_seqs=sequence_budget,
            max_model_len=max_model_len,
        ),
        cache_config=CacheConfig(
            block_size=block_size,
            num_blocks=num_blocks + 1,
            enable_prefix_cache=enable_prefix_cache,
        ),
    )
    load_calls: list[Path] = []

    def initialize_model() -> LlamaModel:
        load_calls.append(checkpoint_path)
        model = load_model(config)
        assert isinstance(model, LlamaModel)
        return model

    worker = Worker(initialize_model, config)
    cache_manager = KVCacheManager(config)
    scheduler = Scheduler(config, cache_manager)
    executor = RecordingSingleProcessExecutor(worker)
    return LoadedEngineHarness(
        engine=Engine(scheduler, executor),
        scheduler=scheduler,
        cache_manager=cache_manager,
        worker=worker,
        executor=executor,
        load_calls=load_calls,
    )


def run_to_completion(
    harness: LoadedEngineHarness,
) -> dict[str, RequestOutput]:
    latest_outputs: dict[str, RequestOutput] = {}
    for _ in range(64):
        if not harness.engine.has_unfinished_requests():
            return latest_outputs
        for output in harness.engine.step():
            latest_outputs[output.request_id] = output
    raise AssertionError("loaded-model engine did not finish within 64 steps")


def request_output_tokens(
    harness: LoadedEngineHarness,
    request_id: str,
) -> tuple[int, ...]:
    request = harness.scheduler.get_request(request_id)
    assert request is not None
    return tuple(request.output_token_ids)


@torch.no_grad()
def reference_greedy_tokens(
    reference_model: LlamaForCausalLM,
    prompt_token_ids: list[int],
    *,
    max_tokens: int,
    eos_token_id: int | None = None,
) -> tuple[int, ...]:
    sequence = list(prompt_token_ids)
    generated: list[int] = []
    device = next(reference_model.parameters()).device
    for _ in range(max_tokens):
        input_ids = torch.tensor(
            [sequence],
            dtype=torch.int64,
            device=device,
        )
        logits = reference_model(input_ids, use_cache=False).logits[0, -1]
        token_id = int(logits.argmax(dim=-1).item())
        generated.append(token_id)
        sequence.append(token_id)
        if eos_token_id is not None and token_id == eos_token_id:
            break
    return tuple(generated)


def test_loaded_checkpoint_runs_through_model_runner_and_engine(
    tiny_hf_llama_checkpoint: tuple[LlamaForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_llama_checkpoint
    harness = make_loaded_engine(checkpoint_path)
    model_runner = harness.worker._model_runner
    assert model_runner is not None
    assert isinstance(model_runner._model, LlamaModel)
    assert harness.load_calls == [checkpoint_path]

    prompt = [1, 2, 3]
    harness.engine.add_request(
        "loaded-request",
        prompt,
        SamplingParams(max_tokens=1),
    )
    output = run_to_completion(harness)["loaded-request"]

    assert output.request_id == "loaded-request"
    assert output.output_token_ids == reference_greedy_tokens(
        reference_model,
        prompt,
        max_tokens=1,
    )
    assert output.finish_reason is FinishReason.LENGTH
    assert len(harness.executor.plans) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_loaded_checkpoint_runs_through_cuda_engine_fp16(
    tiny_hf_llama_checkpoint: tuple[LlamaForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_llama_checkpoint
    reference_model.to(device="cuda", dtype=torch.float16)
    prompt = [1, 2, 3]
    harness = make_loaded_engine(
        checkpoint_path,
        sequence_budget=1,
        device="cuda",
        dtype=torch.float16,
    )
    harness.engine.add_request(
        "cuda-loaded",
        prompt,
        SamplingParams(max_tokens=2),
    )

    output = run_to_completion(harness)["cuda-loaded"]
    model_runner = harness.worker._model_runner
    assert model_runner is not None
    kv_cache_buffers = model_runner._kv_cache_buffers
    assert kv_cache_buffers is not None

    assert output.output_token_ids == reference_greedy_tokens(
        reference_model,
        prompt,
        max_tokens=2,
    )
    assert model_runner._available_kv_cache_memory_bytes is not None
    assert model_runner._available_kv_cache_memory_bytes > 0
    assert kv_cache_buffers[0].device.type == "cuda"
    assert kv_cache_buffers[0].dtype is torch.float16


def test_loaded_full_and_chunked_prefill_match_transformers(
    tiny_hf_llama_checkpoint: tuple[LlamaForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_llama_checkpoint
    prompt = [1, 3, 5, 7, 9]
    expected = reference_greedy_tokens(reference_model, prompt, max_tokens=1)
    full = make_loaded_engine(
        checkpoint_path,
        token_budget=8,
        sequence_budget=1,
    )
    chunked = make_loaded_engine(
        checkpoint_path,
        token_budget=2,
        sequence_budget=1,
    )
    for harness in (full, chunked):
        harness.engine.add_request(
            "request-1",
            prompt,
            SamplingParams(max_tokens=1),
        )
        run_to_completion(harness)

    assert request_output_tokens(full, "request-1") == expected
    assert request_output_tokens(chunked, "request-1") == expected
    assert [
        plan.scheduled_requests[0].num_scheduled_tokens
        for plan in full.executor.plans
    ] == [5]
    assert [
        plan.scheduled_requests[0].num_scheduled_tokens
        for plan in chunked.executor.plans
    ] == [2, 2, 1]


def test_loaded_multi_token_decode_matches_transformers_step_by_step(
    tiny_hf_llama_checkpoint: tuple[LlamaForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_llama_checkpoint
    prompt = [1, 4, 6]
    harness = make_loaded_engine(checkpoint_path, sequence_budget=1)
    harness.engine.add_request(
        "decode",
        prompt,
        SamplingParams(max_tokens=4),
    )

    output = run_to_completion(harness)["decode"]

    assert output.output_token_ids == reference_greedy_tokens(
        reference_model,
        prompt,
        max_tokens=4,
    )
    assert len(harness.executor.plans) == 4
    assert [
        plan.scheduled_requests[0].num_scheduled_tokens
        for plan in harness.executor.plans
    ] == [3, 1, 1, 1]


def test_loaded_mixed_decode_prefill_matches_independent_and_transformers(
    tiny_hf_llama_checkpoint: tuple[LlamaForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_llama_checkpoint
    decode_prompt = [1]
    prefill_prompt = [3, 4, 5]
    mixed = make_loaded_engine(
        checkpoint_path,
        token_budget=4,
        sequence_budget=2,
    )
    mixed.engine.add_request(
        "decode",
        decode_prompt,
        SamplingParams(max_tokens=2),
    )
    mixed.engine.step()
    mixed.engine.add_request(
        "prefill",
        prefill_prompt,
        SamplingParams(max_tokens=1),
    )
    run_to_completion(mixed)

    decode_only = make_loaded_engine(checkpoint_path, sequence_budget=1)
    decode_only.engine.add_request(
        "decode",
        decode_prompt,
        SamplingParams(max_tokens=2),
    )
    run_to_completion(decode_only)
    prefill_only = make_loaded_engine(checkpoint_path, sequence_budget=1)
    prefill_only.engine.add_request(
        "prefill",
        prefill_prompt,
        SamplingParams(max_tokens=1),
    )
    run_to_completion(prefill_only)

    expected_decode = reference_greedy_tokens(
        reference_model,
        decode_prompt,
        max_tokens=2,
    )
    expected_prefill = reference_greedy_tokens(
        reference_model,
        prefill_prompt,
        max_tokens=1,
    )
    assert request_output_tokens(mixed, "decode") == expected_decode
    assert request_output_tokens(mixed, "prefill") == expected_prefill
    assert request_output_tokens(decode_only, "decode") == expected_decode
    assert request_output_tokens(prefill_only, "prefill") == expected_prefill
    assert [
        request.request_id
        for request in mixed.executor.plans[1].scheduled_requests
    ] == ["decode", "prefill"]


def test_loaded_prefix_cache_reuses_blocks_and_preserves_tokens(
    tiny_hf_llama_checkpoint: tuple[LlamaForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_llama_checkpoint
    prompt = [1, 2, 3, 4, 5]
    expected = reference_greedy_tokens(reference_model, prompt, max_tokens=1)
    prefix_on = make_loaded_engine(checkpoint_path, enable_prefix_cache=True)
    for request_id in ("first", "second"):
        prefix_on.engine.add_request(
            request_id,
            prompt,
            SamplingParams(max_tokens=1),
        )
        run_to_completion(prefix_on)

    prefix_off = make_loaded_engine(checkpoint_path, enable_prefix_cache=False)
    prefix_off.engine.add_request(
        "uncached",
        prompt,
        SamplingParams(max_tokens=1),
    )
    run_to_completion(prefix_off)

    first_plan, second_plan = prefix_on.executor.plans
    first = first_plan.scheduled_requests[0]
    second = second_plan.scheduled_requests[0]
    assert second.num_computed_tokens == 4
    assert second.input_token_ids == (5,)
    assert (
        second.block_tables[0].block_ids[:2]
        == first.block_tables[0].block_ids[:2]
    )
    assert request_output_tokens(prefix_on, "first") == expected
    assert request_output_tokens(prefix_on, "second") == expected
    assert request_output_tokens(prefix_off, "uncached") == expected


def test_loaded_preemption_recompute_matches_unconstrained_execution(
    tiny_hf_llama_checkpoint: tuple[LlamaForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_llama_checkpoint
    prompts = {"first": [1, 2], "second": [3, 4]}
    preempted = make_loaded_engine(
        checkpoint_path,
        token_budget=4,
        sequence_budget=2,
        num_blocks=2,
    )
    baseline = make_loaded_engine(
        checkpoint_path,
        token_budget=4,
        sequence_budget=2,
        num_blocks=4,
    )
    for harness in (preempted, baseline):
        for request_id, prompt in prompts.items():
            harness.engine.add_request(
                request_id,
                prompt,
                SamplingParams(max_tokens=2),
            )
        run_to_completion(harness)

    assert preempted.executor.plans[1].preempted_request_ids == ("second",)
    resumed = preempted.executor.plans[2].scheduled_requests[0]
    assert resumed.request_id == "second"
    assert resumed.num_computed_tokens == 0
    for request_id, prompt in prompts.items():
        expected = reference_greedy_tokens(
            reference_model,
            prompt,
            max_tokens=2,
        )
        assert request_output_tokens(preempted, request_id) == expected
        assert request_output_tokens(baseline, request_id) == expected


def test_loaded_reused_block_does_not_leak_stale_kv(
    tiny_hf_llama_checkpoint: tuple[LlamaForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_llama_checkpoint
    reused = make_loaded_engine(
        checkpoint_path,
        token_budget=4,
        sequence_budget=1,
        block_size=4,
        num_blocks=1,
    )
    reused.engine.add_request("old", [1, 2, 3], SamplingParams(max_tokens=1))
    run_to_completion(reused)
    reused.engine.add_request("new", [7], SamplingParams(max_tokens=1))
    run_to_completion(reused)

    fresh = make_loaded_engine(
        checkpoint_path,
        token_budget=4,
        sequence_budget=1,
        block_size=4,
        num_blocks=1,
    )
    fresh.engine.add_request("new", [7], SamplingParams(max_tokens=1))
    run_to_completion(fresh)

    expected = reference_greedy_tokens(reference_model, [7], max_tokens=1)
    assert (
        reused.executor.plans[0]
        .scheduled_requests[0]
        .block_tables[0]
        .block_ids
        == (0,)
    )
    assert (
        reused.executor.plans[1]
        .scheduled_requests[0]
        .block_tables[0]
        .block_ids
        == (0,)
    )
    assert request_output_tokens(reused, "new") == expected
    assert request_output_tokens(fresh, "new") == expected


def test_loaded_eos_max_tokens_and_model_length_termination(
    tiny_hf_llama_checkpoint: tuple[LlamaForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_llama_checkpoint
    prompt = [5]
    first_token = reference_greedy_tokens(
        reference_model,
        prompt,
        max_tokens=1,
    )[0]

    eos = make_loaded_engine(checkpoint_path)
    eos.engine.add_request(
        "eos",
        prompt,
        SamplingParams(max_tokens=4, eos_token_id=first_token),
    )
    eos_output = run_to_completion(eos)["eos"]

    length = make_loaded_engine(checkpoint_path)
    length.engine.add_request(
        "length",
        prompt,
        SamplingParams(max_tokens=1),
    )
    length_output = run_to_completion(length)["length"]

    model_length = make_loaded_engine(checkpoint_path, max_model_len=2)
    model_length.engine.add_request(
        "model-length",
        prompt,
        SamplingParams(max_tokens=4),
    )
    model_length_output = run_to_completion(model_length)["model-length"]

    assert eos_output.output_token_ids == (first_token,)
    assert eos_output.finish_reason is FinishReason.EOS
    assert length_output.finish_reason is FinishReason.LENGTH
    assert model_length_output.finish_reason is FinishReason.MODEL_LENGTH


def test_loaded_finished_and_aborted_requests_release_logical_blocks(
    tiny_hf_llama_checkpoint: tuple[LlamaForCausalLM, Path],
) -> None:
    _, checkpoint_path = tiny_hf_llama_checkpoint
    harness = make_loaded_engine(
        checkpoint_path,
        token_budget=2,
        sequence_budget=1,
        block_size=2,
        num_blocks=2,
    )
    harness.engine.add_request(
        "finished",
        [7],
        SamplingParams(max_tokens=1),
    )
    run_to_completion(harness)
    assert harness.cache_manager.num_free_blocks == 2
    with pytest.raises(KeyError):
        harness.cache_manager.get_block_table("finished")

    harness.engine.add_request(
        "aborted",
        [1, 2, 3],
        SamplingParams(max_tokens=1),
    )
    assert harness.engine.step() == ()
    assert harness.cache_manager.num_free_blocks == 1
    assert harness.engine.abort_request("aborted") is True
    assert harness.cache_manager.num_free_blocks == 2
    with pytest.raises(KeyError):
        harness.cache_manager.get_block_table("aborted")
