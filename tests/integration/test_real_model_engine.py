"""End-to-end Engine tests using the real tiny Llama execution path."""

from dataclasses import dataclass

import pytest
import torch

from micro_vllm.config import (
    CacheConfig,
    LLMConfig,
    ModelConfig,
    SchedulerConfig,
)
from micro_vllm.core.kv_cache_manager import KVCacheManager
from micro_vllm.core.request import FinishReason, RequestStatus
from micro_vllm.core.scheduler import Scheduler
from micro_vllm.core.scheduler_output import SchedulerOutput
from micro_vllm.engine.engine import Engine
from micro_vllm.engine.output import RequestOutput
from micro_vllm.executor.interface import ModelRunnerOutput
from micro_vllm.executor.single_process import SingleProcessExecutor
from micro_vllm.layers.attention_backend import ReferenceAttentionBackend
from micro_vllm.models.llama import LlamaModel
from micro_vllm.sampling.params import SamplingParams
from micro_vllm.worker.worker import Worker


class RecordingSingleProcessExecutor(SingleProcessExecutor):
    """Record immutable plans while retaining the real executor path."""

    __slots__ = ("plans", "results")

    def __init__(self, worker: Worker) -> None:
        super().__init__(worker)
        self.plans: list[SchedulerOutput] = []
        self.results: list[ModelRunnerOutput] = []

    def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput:
        result = super().execute(plan)
        self.plans.append(plan)
        self.results.append(result)
        return result


@dataclass(slots=True)
class RealEngineHarness:
    engine: Engine
    scheduler: Scheduler
    cache_manager: KVCacheManager
    worker: Worker
    executor: RecordingSingleProcessExecutor


def make_model_config() -> ModelConfig:
    return ModelConfig(
        {
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 32,
        }
    )


def make_real_engine(
    *,
    token_budget: int = 8,
    sequence_budget: int = 2,
    max_model_len: int = 16,
    block_size: int = 2,
    num_blocks: int = 16,
    enable_prefix_cache: bool = False,
) -> RealEngineHarness:
    torch.manual_seed(0)
    config = LLMConfig(
        model="unused",
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=token_budget,
            max_num_seqs=sequence_budget,
            max_model_len=max_model_len,
        ),
        cache_config=CacheConfig(
            block_size=block_size,
            num_blocks=num_blocks,
            enable_prefix_cache=enable_prefix_cache,
        ),
        device="cpu",
        dtype=torch.float32,
    )
    worker = Worker(
        lambda: LlamaModel(make_model_config(), ReferenceAttentionBackend()),
        config,
    )
    cache_manager = KVCacheManager(config)
    scheduler = Scheduler(config, cache_manager)
    executor = RecordingSingleProcessExecutor(worker)
    return RealEngineHarness(
        engine=Engine(scheduler, executor),
        scheduler=scheduler,
        cache_manager=cache_manager,
        worker=worker,
        executor=executor,
    )


def run_to_completion(
    harness: RealEngineHarness,
) -> dict[str, RequestOutput]:
    latest_outputs: dict[str, RequestOutput] = {}
    for _ in range(32):
        if not harness.engine.has_unfinished_requests():
            return latest_outputs
        for output in harness.engine.step():
            latest_outputs[output.request_id] = output
    raise AssertionError("engine did not finish within 32 steps")


def request_output_tokens(
    harness: RealEngineHarness,
    request_id: str,
) -> tuple[int, ...]:
    request = harness.scheduler.get_request(request_id)
    assert request is not None
    return tuple(request.output_token_ids)


def test_control_plane_uses_model_runner_cache_capacity() -> None:
    harness = make_real_engine(num_blocks=7)
    model_runner = harness.worker._model_runner
    assert model_runner is not None
    physical_kv_cache = model_runner._physical_kv_cache
    assert physical_kv_cache is not None

    assert harness.worker.config.cache_config.num_blocks == 7
    assert physical_kv_cache[0].shape[1] == 7
    assert harness.cache_manager._block_pool.num_blocks == 7


def test_real_model_token_moves_request_from_waiting_to_finished() -> None:
    harness = make_real_engine()
    logits: list[torch.Tensor] = []
    model_runner = harness.worker._model_runner
    assert model_runner is not None
    model = model_runner._model
    assert isinstance(model, LlamaModel)
    hook = model.lm_head.register_forward_hook(
        lambda _module, _inputs, output: logits.append(output.detach())
    )
    harness.engine.add_request(
        "request-1",
        [1, 2, 3],
        SamplingParams(max_tokens=1),
    )
    request = harness.scheduler.get_request("request-1")
    assert request is not None and request.status is RequestStatus.WAITING

    outputs = harness.engine.step()
    hook.remove()

    assert isinstance(harness.executor, SingleProcessExecutor)
    assert len(outputs) == 1
    assert outputs[0].output_token_ids == (logits[0].argmax(dim=-1).item(),)
    assert outputs[0].finished is True
    assert outputs[0].finish_reason is FinishReason.LENGTH
    assert request.status is RequestStatus.FINISHED


def test_chunked_prefill_matches_full_prefill_first_token() -> None:
    prompt = [1, 3, 5, 7, 9]
    full = make_real_engine(token_budget=8, sequence_budget=1)
    chunked = make_real_engine(token_budget=2, sequence_budget=1)
    for harness in (full, chunked):
        harness.engine.add_request(
            "request-1",
            prompt,
            SamplingParams(max_tokens=1),
        )
        run_to_completion(harness)

    assert request_output_tokens(full, "request-1") == request_output_tokens(
        chunked,
        "request-1",
    )
    assert [
        plan.scheduled_requests[0].num_scheduled_tokens
        for plan in full.executor.plans
    ] == [5]
    assert [
        plan.scheduled_requests[0].num_scheduled_tokens
        for plan in chunked.executor.plans
    ] == [2, 2, 1]


def test_mixed_decode_and_prefill_match_independent_execution() -> None:
    mixed = make_real_engine(token_budget=4, sequence_budget=2)
    mixed.engine.add_request("decode", [1], SamplingParams(max_tokens=2))
    mixed.engine.step()
    mixed.engine.add_request(
        "prefill",
        [3, 4, 5],
        SamplingParams(max_tokens=1),
    )
    run_to_completion(mixed)

    decode_only = make_real_engine(token_budget=4, sequence_budget=1)
    decode_only.engine.add_request("decode", [1], SamplingParams(max_tokens=2))
    run_to_completion(decode_only)
    prefill_only = make_real_engine(token_budget=4, sequence_budget=1)
    prefill_only.engine.add_request(
        "prefill",
        [3, 4, 5],
        SamplingParams(max_tokens=1),
    )
    run_to_completion(prefill_only)

    assert request_output_tokens(mixed, "decode") == request_output_tokens(
        decode_only,
        "decode",
    )
    assert request_output_tokens(mixed, "prefill") == request_output_tokens(
        prefill_only,
        "prefill",
    )
    assert [
        request.request_id
        for request in mixed.executor.plans[1].scheduled_requests
    ] == ["decode", "prefill"]


def test_prefix_hit_reuses_physical_blocks_and_matches_cache_off() -> None:
    prompt = [1, 2, 3, 4, 5]
    prefix_on = make_real_engine(enable_prefix_cache=True)
    for request_id in ("first", "second"):
        prefix_on.engine.add_request(
            request_id,
            prompt,
            SamplingParams(max_tokens=1),
        )
        run_to_completion(prefix_on)

    prefix_off = make_real_engine(enable_prefix_cache=False)
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
    assert second.block_table[:2] == first.block_table[:2]
    assert request_output_tokens(prefix_on, "first") == request_output_tokens(
        prefix_on,
        "second",
    )
    assert request_output_tokens(prefix_on, "second") == request_output_tokens(
        prefix_off,
        "uncached",
    )


def test_preemption_recompute_matches_execution_without_preemption() -> None:
    preempted = make_real_engine(
        token_budget=4,
        sequence_budget=2,
        block_size=2,
        num_blocks=2,
    )
    baseline = make_real_engine(
        token_budget=4,
        sequence_budget=2,
        block_size=2,
        num_blocks=4,
    )
    for harness in (preempted, baseline):
        harness.engine.add_request("first", [1, 2], SamplingParams(max_tokens=2))
        harness.engine.add_request("second", [3, 4], SamplingParams(max_tokens=2))
        run_to_completion(harness)

    assert preempted.executor.plans[1].preempted_request_ids == ("second",)
    resumed = preempted.executor.plans[2].scheduled_requests[0]
    assert resumed.request_id == "second"
    assert resumed.num_computed_tokens == 0
    for request_id in ("first", "second"):
        assert request_output_tokens(preempted, request_id) == request_output_tokens(
            baseline,
            request_id,
        )


def test_reused_physical_block_does_not_leak_stale_kv() -> None:
    reused = make_real_engine(
        token_budget=4,
        sequence_budget=1,
        block_size=4,
        num_blocks=1,
    )
    reused.engine.add_request(
        "old",
        [1, 2, 3],
        SamplingParams(max_tokens=1),
    )
    run_to_completion(reused)
    reused.engine.add_request("new", [7], SamplingParams(max_tokens=1))
    run_to_completion(reused)

    fresh = make_real_engine(
        token_budget=4,
        sequence_budget=1,
        block_size=4,
        num_blocks=1,
    )
    fresh.engine.add_request("new", [7], SamplingParams(max_tokens=1))
    run_to_completion(fresh)

    assert reused.executor.plans[0].scheduled_requests[0].block_table == (0,)
    assert reused.executor.plans[1].scheduled_requests[0].block_table == (0,)
    assert request_output_tokens(reused, "new") == request_output_tokens(
        fresh,
        "new",
    )


def test_real_model_eos_max_tokens_and_model_length_termination() -> None:
    token_source = make_real_engine()
    token_source.engine.add_request("source", [5], SamplingParams(max_tokens=1))
    first_token = run_to_completion(token_source)["source"].output_token_ids[0]

    eos = make_real_engine()
    eos.engine.add_request(
        "eos",
        [5],
        SamplingParams(max_tokens=4, eos_token_id=first_token),
    )
    eos_output = run_to_completion(eos)["eos"]

    length = make_real_engine()
    length.engine.add_request("length", [5], SamplingParams(max_tokens=1))
    length_output = run_to_completion(length)["length"]

    model_length = make_real_engine(max_model_len=2)
    model_length.engine.add_request(
        "model-length",
        [5],
        SamplingParams(max_tokens=4),
    )
    model_length_output = run_to_completion(model_length)["model-length"]

    assert eos_output.finish_reason is FinishReason.EOS
    assert length_output.finish_reason is FinishReason.LENGTH
    assert model_length_output.finish_reason is FinishReason.MODEL_LENGTH


def test_abort_releases_blocks_and_removes_request_from_future_plans() -> None:
    harness = make_real_engine(
        token_budget=2,
        sequence_budget=1,
        block_size=2,
        num_blocks=2,
    )
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

    harness.engine.add_request("survivor", [7], SamplingParams(max_tokens=1))
    outputs = harness.engine.step()

    aborted = harness.scheduler.get_request("aborted")
    assert aborted is not None and aborted.finish_reason is FinishReason.ABORTED
    assert outputs[0].request_id == "survivor"
    assert [
        request.request_id
        for request in harness.executor.plans[-1].scheduled_requests
    ] == ["survivor"]


def test_seeded_sampling_matches_across_full_chunked_and_mixed_batches() -> None:
    params = SamplingParams(
        max_tokens=4,
        temperature=0.8,
        top_k=16,
        top_p=0.9,
        seed=17,
    )
    prompt = [1, 3, 5, 7, 9]
    full = make_real_engine(token_budget=8, sequence_budget=1)
    chunked = make_real_engine(token_budget=2, sequence_budget=1)
    for harness in (full, chunked):
        harness.engine.add_request("target", prompt, params)
        run_to_completion(harness)

    mixed = make_real_engine(token_budget=8, sequence_budget=2)
    mixed.engine.add_request("other", [2, 4], SamplingParams(max_tokens=2))
    mixed.engine.add_request("target", prompt, params)
    run_to_completion(mixed)

    assert request_output_tokens(full, "target") == request_output_tokens(
        chunked,
        "target",
    )
    assert request_output_tokens(full, "target") == request_output_tokens(
        mixed,
        "target",
    )
    assert [
        plan.scheduled_requests[0].requires_sampling
        for plan in chunked.executor.plans[:3]
    ] == [False, False, True]
    assert [
        plan.scheduled_requests[0].sample_index
        for plan in chunked.executor.plans[:3]
    ] == [None, None, 0]


def test_two_seeded_requests_match_independent_execution_when_interleaved() -> None:
    first_params = SamplingParams(
        max_tokens=3,
        temperature=1.0,
        top_p=0.8,
        seed=11,
    )
    second_params = SamplingParams(
        max_tokens=3,
        temperature=0.7,
        top_k=8,
        seed=29,
    )
    mixed = make_real_engine(token_budget=4, sequence_budget=2)
    mixed.engine.add_request("first", [1], first_params)
    mixed.engine.step()
    mixed.engine.add_request("second", [3, 4, 5], second_params)
    run_to_completion(mixed)

    first_only = make_real_engine(token_budget=2, sequence_budget=1)
    first_only.engine.add_request("first", [1], first_params)
    run_to_completion(first_only)
    second_only = make_real_engine(token_budget=8, sequence_budget=1)
    second_only.engine.add_request("second", [3, 4, 5], second_params)
    run_to_completion(second_only)

    assert request_output_tokens(mixed, "first") == request_output_tokens(
        first_only,
        "first",
    )
    assert request_output_tokens(mixed, "second") == request_output_tokens(
        second_only,
        "second",
    )


def test_seeded_sampling_survives_preemption_without_skipping_index() -> None:
    params = SamplingParams(
        max_tokens=2,
        temperature=1.0,
        top_k=12,
        seed=23,
    )
    preempted = make_real_engine(
        token_budget=4,
        sequence_budget=2,
        block_size=2,
        num_blocks=2,
    )
    baseline = make_real_engine(
        token_budget=4,
        sequence_budget=2,
        block_size=2,
        num_blocks=4,
    )
    for harness in (preempted, baseline):
        harness.engine.add_request("first", [1, 2], params)
        harness.engine.add_request("second", [3, 4], params)
        run_to_completion(harness)

    assert preempted.executor.plans[1].preempted_request_ids == ("second",)
    resumed = preempted.executor.plans[2].scheduled_requests[0]
    assert resumed.request_id == "second"
    assert resumed.sample_index == 1
    for request_id in ("first", "second"):
        assert request_output_tokens(
            preempted,
            request_id,
        ) == request_output_tokens(baseline, request_id)


def test_pending_sampling_plan_retry_returns_identical_token() -> None:
    harness = make_real_engine(token_budget=4, sequence_budget=1)
    params = SamplingParams(
        max_tokens=2,
        temperature=1.0,
        top_p=0.9,
        seed=31,
    )
    harness.engine.add_request("request", [1, 2], params)

    plan = harness.scheduler.schedule()
    first_result = harness.executor.execute(plan)
    second_result = harness.executor.execute(plan)

    assert harness.scheduler.schedule() is plan
    assert plan.scheduled_requests[0].sample_index == 0
    assert first_result == second_result
    harness.scheduler.update(plan, second_result)
    next_plan = harness.scheduler.schedule()
    assert next_plan.scheduled_requests[0].sample_index == 1


def test_unseeded_request_retry_uses_admission_seed() -> None:
    harness = make_real_engine(token_budget=4, sequence_budget=1)
    harness.engine.add_request(
        "request",
        [1, 2],
        SamplingParams(max_tokens=2, temperature=1.0),
    )
    request = harness.scheduler.get_request("request")
    assert request is not None
    assert request.sampling_params.seed is not None

    plan = harness.scheduler.schedule()
    first_result = harness.executor.execute(plan)
    second_result = harness.executor.execute(plan)

    assert plan.scheduled_requests[0].sampling_params is request.sampling_params
    assert first_result == second_result


def test_stochastic_eos_ignore_eos_and_model_length_termination() -> None:
    params = SamplingParams(max_tokens=1, temperature=1.0, seed=41)
    source = make_real_engine()
    source.engine.add_request("source", [5], params)
    first_token = run_to_completion(source)["source"].output_token_ids[0]

    eos = make_real_engine()
    eos.engine.add_request(
        "eos",
        [5],
        SamplingParams(
            max_tokens=4,
            eos_token_id=first_token,
            temperature=1.0,
            seed=41,
        ),
    )
    eos_output = run_to_completion(eos)["eos"]

    ignored = make_real_engine()
    ignored.engine.add_request(
        "ignored",
        [5],
        SamplingParams(
            max_tokens=2,
            eos_token_id=first_token,
            ignore_eos=True,
            temperature=1.0,
            seed=41,
        ),
    )
    ignored_output = run_to_completion(ignored)["ignored"]

    model_length = make_real_engine(max_model_len=2)
    model_length.engine.add_request(
        "model-length",
        [5],
        SamplingParams(max_tokens=4, temperature=1.0, seed=41),
    )
    model_length_output = run_to_completion(model_length)["model-length"]

    assert eos_output.output_token_ids == (first_token,)
    assert eos_output.finish_reason is FinishReason.EOS
    assert ignored_output.finish_reason is FinishReason.LENGTH
    assert len(ignored_output.output_token_ids) == 2
    assert model_length_output.finish_reason is FinishReason.MODEL_LENGTH


def test_aborted_stochastic_request_leaves_no_sampler_state() -> None:
    harness = make_real_engine(
        token_budget=2,
        sequence_budget=1,
        block_size=2,
        num_blocks=2,
    )
    harness.engine.add_request(
        "aborted",
        [1, 2, 3],
        SamplingParams(max_tokens=2, temperature=1.0, seed=47),
    )
    assert harness.engine.step() == ()
    assert harness.engine.abort_request("aborted") is True

    model_runner = harness.worker._model_runner
    assert model_runner is not None
    assert not hasattr(model_runner._sampler, "__dict__")
    assert harness.cache_manager.num_free_blocks == 2
