"""End-to-end tests for the synchronous engine control loop."""

from micro_vllm.config import CacheConfig, LLMConfig, ModelConfig, SchedulerConfig
from micro_vllm.core.kv_cache_manager import KVCacheManager
from micro_vllm.core.request import FinishReason, RequestStatus
from micro_vllm.core.scheduler import Scheduler
from micro_vllm.core.scheduler_output import SchedulerOutput
from micro_vllm.engine.engine import Engine
from micro_vllm.executor.interface import ModelRunnerOutput, SampledToken
from micro_vllm.sampling.params import SamplingParams


class FakeExecutor:
    """Deterministic test-only executor that consumes immutable plans."""

    def __init__(
        self,
        *,
        default_token_id: int = 7,
        sampled_token_ids: dict[tuple[int, str], int] | None = None,
        result_overrides: dict[int, ModelRunnerOutput] | None = None,
    ) -> None:
        self.default_token_id = default_token_id
        self.sampled_token_ids = dict(sampled_token_ids or {})
        self.result_overrides = dict(result_overrides or {})
        self.plans: list[SchedulerOutput] = []

    def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput:
        step_index = len(self.plans)
        self.plans.append(plan)
        override = self.result_overrides.get(step_index)
        if override is not None:
            return override
        return ModelRunnerOutput(
            tuple(
                SampledToken(
                    scheduled.request_id,
                    self.sampled_token_ids.get(
                        (step_index, scheduled.request_id), self.default_token_id
                    ),
                )
                for scheduled in plan.scheduled_requests
                if scheduled.requires_sampling
            )
        )


def make_engine(
    *,
    token_budget: int = 4,
    sequence_budget: int = 2,
    max_model_len: int = 128,
    executor: FakeExecutor | None = None,
) -> tuple[Engine, Scheduler, FakeExecutor]:
    config = LLMConfig(
        model_config=ModelConfig(model="unused"),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=token_budget,
            max_num_seqs=sequence_budget,
            max_model_len=max_model_len,
        ),
        cache_config=CacheConfig(
            block_size=4,
            num_blocks=128,
            enable_prefix_cache=False,
        ),
    )
    scheduler = Scheduler(
        config,
        KVCacheManager(config),
    )
    fake_executor = executor or FakeExecutor()
    return Engine(scheduler, fake_executor), scheduler, fake_executor


def test_single_request_moves_from_waiting_to_finished() -> None:
    engine, scheduler, executor = make_engine()
    engine.add_request("request-1", [1, 2], SamplingParams(max_tokens=1))
    request = scheduler.get_request("request-1")
    assert request is not None and request.status is RequestStatus.WAITING

    outputs = engine.step()

    assert len(executor.plans) == 1
    assert len(outputs) == 1
    assert outputs[0].request_id == "request-1"
    assert outputs[0].output_token_ids == (7,)
    assert outputs[0].finished is True
    assert outputs[0].finish_reason is FinishReason.LENGTH
    assert request.status is RequestStatus.FINISHED
    assert engine.has_unfinished_requests() is False


def test_long_prompt_uses_four_four_two_chunks_before_first_token() -> None:
    engine, scheduler, executor = make_engine(token_budget=4, sequence_budget=1)
    engine.add_request("request-1", list(range(10)), SamplingParams(max_tokens=1))

    first_outputs = engine.step()
    second_outputs = engine.step()
    third_outputs = engine.step()

    assert first_outputs == second_outputs == ()
    assert [
        plan.scheduled_requests[0].num_scheduled_tokens for plan in executor.plans
    ] == [4, 4, 2]
    assert [
        plan.scheduled_requests[0].requires_sampling for plan in executor.plans
    ] == [False, False, True]
    assert third_outputs[0].output_token_ids == (7,)
    assert third_outputs[0].finished is True
    request = scheduler.get_request("request-1")
    assert request is not None and request.num_computed_tokens == 10


def test_running_decode_and_waiting_prefill_share_one_plan() -> None:
    engine, _, executor = make_engine(token_budget=4, sequence_budget=2)
    engine.add_request("decode", [1], SamplingParams(max_tokens=3))
    engine.step()
    engine.add_request("prefill", list(range(10)), SamplingParams(max_tokens=2))

    outputs = engine.step()
    plan = executor.plans[1]

    assert [item.request_id for item in plan.scheduled_requests] == [
        "decode",
        "prefill",
    ]
    assert [item.num_scheduled_tokens for item in plan.scheduled_requests] == [1, 3]
    assert [output.request_id for output in outputs] == ["decode"]


def test_executor_stops_after_request_reaches_max_tokens() -> None:
    engine, _, executor = make_engine(sequence_budget=1)
    engine.add_request("request-1", [1], SamplingParams(max_tokens=2))

    first_outputs = engine.step()
    second_outputs = engine.step()
    empty_outputs = engine.step()

    assert first_outputs[0].finished is False
    assert second_outputs[0].finished is True
    assert second_outputs[0].finish_reason is FinishReason.LENGTH
    assert empty_outputs == ()
    assert len(executor.plans) == 2


def test_engine_reports_model_length_finish_reason() -> None:
    engine, _, executor = make_engine(max_model_len=3, sequence_budget=1)
    engine.add_request("request-1", [1], SamplingParams(max_tokens=10))

    first_outputs = engine.step()
    second_outputs = engine.step()

    assert first_outputs[0].finished is False
    assert second_outputs[0].output_token_ids == (7, 7)
    assert second_outputs[0].finished is True
    assert second_outputs[0].finish_reason is FinishReason.MODEL_LENGTH
    assert len(executor.plans) == 2


def test_eos_finishes_request() -> None:
    executor = FakeExecutor(sampled_token_ids={(0, "request-1"): 0})
    engine, _, _ = make_engine(executor=executor)
    engine.add_request(
        "request-1",
        [1],
        SamplingParams(max_tokens=4, eos_token_id=0),
    )

    outputs = engine.step()

    assert outputs[0].finished is True
    assert outputs[0].finish_reason is FinishReason.EOS
    assert engine.has_unfinished_requests() is False


def test_ignore_eos_continues_until_max_tokens() -> None:
    executor = FakeExecutor(
        sampled_token_ids={(0, "request-1"): 0, (1, "request-1"): 0}
    )
    engine, _, _ = make_engine(executor=executor)
    engine.add_request(
        "request-1",
        [1],
        SamplingParams(max_tokens=2, eos_token_id=0, ignore_eos=True),
    )

    first_outputs = engine.step()
    second_outputs = engine.step()

    assert first_outputs[0].finished is False
    assert second_outputs[0].output_token_ids == (0, 0)
    assert second_outputs[0].finished is True
    assert second_outputs[0].finish_reason is FinishReason.LENGTH


def test_request_can_be_added_while_engine_is_running() -> None:
    engine, _, executor = make_engine(token_budget=4, sequence_budget=2)
    engine.add_request("first", [1], SamplingParams(max_tokens=2))
    first_outputs = engine.step()
    engine.add_request("second", [2], SamplingParams(max_tokens=2))

    second_outputs = engine.step()

    assert [output.request_id for output in first_outputs] == ["first"]
    assert [output.request_id for output in second_outputs] == ["first", "second"]
    assert [item.request_id for item in executor.plans[1].scheduled_requests] == [
        "first",
        "second",
    ]


def test_waiting_and_running_requests_can_be_aborted() -> None:
    engine, scheduler, executor = make_engine(token_budget=2, sequence_budget=1)
    engine.add_request("waiting", [1], SamplingParams(max_tokens=2))
    assert engine.abort_request("waiting") is True
    waiting = scheduler.get_request("waiting")
    assert waiting is not None and waiting.finish_reason is FinishReason.ABORTED

    engine.add_request("running", [1, 2, 3], SamplingParams(max_tokens=2))
    assert engine.step() == ()
    assert len(executor.plans) == 1
    running = scheduler.get_request("running")
    assert running is not None and running.status is RequestStatus.RUNNING
    assert engine.abort_request("running") is True
    assert running.finish_reason is FinishReason.ABORTED
    assert engine.has_unfinished_requests() is False
    assert engine.abort_request("running") is False
    assert engine.abort_request("unknown") is False


def test_empty_schedule_does_not_execute() -> None:
    engine, _, executor = make_engine()

    assert engine.step() == ()
    assert executor.plans == []
