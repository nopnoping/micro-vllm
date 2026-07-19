"""Tests for unified token-budget scheduling and result updates."""

import pytest

from micro_vllm.config import SchedulerConfig
from micro_vllm.core.request import FinishReason, Request, RequestStatus
from micro_vllm.core.scheduler import Scheduler
from micro_vllm.core.scheduler_output import SchedulerOutput
from micro_vllm.executor.interface import ModelRunnerOutput, SampledToken
from micro_vllm.sampling.params import SamplingParams


def make_request(
    request_id: str,
    prompt_token_ids: list[int],
    *,
    max_tokens: int = 4,
    eos_token_id: int | None = None,
    ignore_eos: bool = False,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(
            max_tokens=max_tokens,
            eos_token_id=eos_token_id,
            ignore_eos=ignore_eos,
        ),
    )


def make_output(sampled_tokens: dict[str, int]) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        tuple(
            SampledToken(request_id, token_id)
            for request_id, token_id in sampled_tokens.items()
        )
    )


def test_single_request_full_prompt_is_scheduled_in_one_step() -> None:
    scheduler = Scheduler(SchedulerConfig(4, 2))
    request = make_request("request-1", [1, 2, 3])
    scheduler.add_request(request)

    plan = scheduler.schedule()

    assert len(plan.scheduled_requests) == 1
    scheduled = plan.scheduled_requests[0]
    assert scheduled.request_id == "request-1"
    assert scheduled.input_token_ids == (1, 2, 3)
    assert scheduled.num_computed_tokens == 0
    assert scheduled.num_scheduled_tokens == 3
    assert scheduled.requires_sampling is True
    assert request.status is RequestStatus.RUNNING
    assert request.num_computed_tokens == 0

    scheduler.update(plan, make_output({"request-1": 7}))

    assert request.num_computed_tokens == 3
    assert request.output_token_ids == [7]


def test_long_prompt_is_split_across_multiple_chunks() -> None:
    scheduler = Scheduler(SchedulerConfig(4, 1))
    request = make_request("request-1", list(range(10)))
    scheduler.add_request(request)
    chunk_sizes: list[int] = []
    sampling_boundaries: list[bool] = []

    for index in range(3):
        plan = scheduler.schedule()
        scheduled = plan.scheduled_requests[0]
        chunk_sizes.append(scheduled.num_scheduled_tokens)
        sampling_boundaries.append(scheduled.requires_sampling)
        result = (
            make_output({"request-1": 10})
            if scheduled.requires_sampling
            else ModelRunnerOutput()
        )
        scheduler.update(plan, result)
        assert request.num_computed_tokens == min((index + 1) * 4, 10)

    assert chunk_sizes == [4, 4, 2]
    assert sampling_boundaries == [False, False, True]
    assert request.output_token_ids == [10]


def test_intermediate_prompt_chunk_rejects_sampled_token() -> None:
    scheduler = Scheduler(SchedulerConfig(2, 1))
    request = make_request("request-1", [1, 2, 3])
    scheduler.add_request(request)
    plan = scheduler.schedule()

    assert plan.scheduled_requests[0].requires_sampling is False
    with pytest.raises(ValueError, match="does not require sampling"):
        scheduler.update(plan, make_output({"request-1": 4}))

    assert request.num_computed_tokens == 0
    assert scheduler.schedule() is plan
    scheduler.update(plan, ModelRunnerOutput())
    assert request.num_computed_tokens == 2


def test_running_decode_precedes_waiting_prefill_in_same_plan() -> None:
    scheduler = Scheduler(SchedulerConfig(4, 2))
    decode = make_request("decode", [1])
    scheduler.add_request(decode)
    prompt_plan = scheduler.schedule()
    scheduler.update(prompt_plan, make_output({"decode": 2}))

    prefill = make_request("prefill", list(range(10)))
    scheduler.add_request(prefill)
    plan = scheduler.schedule()

    assert [item.request_id for item in plan.scheduled_requests] == [
        "decode",
        "prefill",
    ]
    assert [item.num_scheduled_tokens for item in plan.scheduled_requests] == [1, 3]
    assert [item.input_token_ids for item in plan.scheduled_requests] == [
        (2,),
        (0, 1, 2),
    ]
    assert [item.requires_sampling for item in plan.scheduled_requests] == [
        True,
        False,
    ]


def test_token_budget_is_never_exceeded() -> None:
    scheduler = Scheduler(SchedulerConfig(5, 4))
    for index in range(4):
        scheduler.add_request(
            make_request(f"request-{index}", [index, index + 1, index + 2])
        )

    plan = scheduler.schedule()

    assert plan.total_num_scheduled_tokens == 5
    assert all(item.num_scheduled_tokens > 0 for item in plan.scheduled_requests)


def test_sequence_budget_is_never_exceeded() -> None:
    scheduler = Scheduler(SchedulerConfig(100, 2))
    requests = [make_request(f"request-{index}", [index]) for index in range(3)]
    for request in requests:
        scheduler.add_request(request)

    plan = scheduler.schedule()

    assert [item.request_id for item in plan.scheduled_requests] == [
        "request-0",
        "request-1",
    ]
    assert list(scheduler._waiting) == [requests[2]]


def test_waiting_queue_admits_requests_in_fifo_order() -> None:
    scheduler = Scheduler(SchedulerConfig(1, 1))
    requests = [
        make_request(f"request-{index}", [index], max_tokens=1)
        for index in range(3)
    ]
    for request in requests:
        scheduler.add_request(request)

    admitted_ids: list[str] = []
    for request in requests:
        plan = scheduler.schedule()
        admitted_ids.append(plan.scheduled_requests[0].request_id)
        scheduler.update(plan, make_output({request.request_id: 10}))

    assert admitted_ids == ["request-0", "request-1", "request-2"]


def test_running_order_remains_stable() -> None:
    scheduler = Scheduler(SchedulerConfig(8, 3))
    requests = [make_request(f"request-{index}", [index]) for index in range(3)]
    for request in requests:
        scheduler.add_request(request)
    first_plan = scheduler.schedule()
    scheduler.update(
        first_plan,
        make_output({request.request_id: index + 10 for index, request in enumerate(requests)}),
    )

    second_plan = scheduler.schedule()

    assert [item.request_id for item in first_plan.scheduled_requests] == [
        "request-0",
        "request-1",
        "request-2",
    ]
    assert [item.request_id for item in second_plan.scheduled_requests] == [
        "request-0",
        "request-1",
        "request-2",
    ]


def test_duplicate_request_id_is_rejected_without_queue_pollution() -> None:
    scheduler = Scheduler(SchedulerConfig(4, 2))
    first = make_request("request-1", [1])
    duplicate = make_request("request-1", [2])
    scheduler.add_request(first)

    with pytest.raises(ValueError, match="duplicate request_id"):
        scheduler.add_request(duplicate)

    assert list(scheduler._waiting) == [first]
    assert scheduler.get_request("request-1") is first


def test_invalid_executor_outputs_fail_atomically_and_can_be_retried() -> None:
    scheduler = Scheduler(SchedulerConfig(4, 2))
    first = make_request("first", [1])
    second = make_request("second", [2])
    scheduler.add_request(first)
    scheduler.add_request(second)
    plan = scheduler.schedule()

    with pytest.raises(ValueError, match="missing sampled token"):
        scheduler.update(plan, make_output({"first": 3}))
    with pytest.raises(ValueError, match="unknown sampled token request_id"):
        scheduler.update(
            plan,
            make_output({"first": 3, "second": 4, "unknown": 5}),
        )
    with pytest.raises(ValueError, match="duplicate request_id"):
        ModelRunnerOutput((SampledToken("first", 3), SampledToken("first", 4)))

    assert first.num_computed_tokens == second.num_computed_tokens == 0
    assert first.output_token_ids == second.output_token_ids == []
    assert scheduler.schedule() is plan

    scheduler.update(plan, make_output({"first": 3, "second": 4}))
    assert first.num_computed_tokens == second.num_computed_tokens == 1


def test_eos_finishes_request_while_ignore_eos_continues() -> None:
    scheduler = Scheduler(SchedulerConfig(4, 2))
    eos = make_request("eos", [1], max_tokens=2, eos_token_id=0)
    ignored = make_request(
        "ignored", [2], max_tokens=2, eos_token_id=0, ignore_eos=True
    )
    scheduler.add_request(eos)
    scheduler.add_request(ignored)

    first_plan = scheduler.schedule()
    scheduler.update(first_plan, make_output({"eos": 0, "ignored": 0}))

    assert eos.finish_reason is FinishReason.EOS
    assert eos not in scheduler._running
    assert ignored.status is RequestStatus.RUNNING
    assert ignored.output_token_ids == [0]

    second_plan = scheduler.schedule()
    scheduler.update(second_plan, make_output({"ignored": 0}))
    assert ignored.finish_reason is FinishReason.LENGTH


def test_max_tokens_finishes_request_and_removes_it_from_running() -> None:
    scheduler = Scheduler(SchedulerConfig(4, 1))
    request = make_request("request-1", [1], max_tokens=1)
    scheduler.add_request(request)
    plan = scheduler.schedule()

    scheduler.update(plan, make_output({"request-1": 2}))

    assert request.finish_reason is FinishReason.LENGTH
    assert request not in scheduler._running
    assert scheduler.has_unfinished_requests() is False


def test_plan_cannot_be_updated_twice() -> None:
    scheduler = Scheduler(SchedulerConfig(4, 1))
    request = make_request("request-1", [1])
    scheduler.add_request(request)
    plan = scheduler.schedule()
    assert scheduler.schedule() is plan
    result = make_output({"request-1": 2})
    scheduler.update(plan, result)
    state_after_update = (request.num_computed_tokens, tuple(request.output_token_ids))

    with pytest.raises(ValueError, match="already updated"):
        scheduler.update(plan, result)

    assert (request.num_computed_tokens, tuple(request.output_token_ids)) == state_after_update


def test_equal_but_foreign_plan_is_rejected() -> None:
    scheduler = Scheduler(SchedulerConfig(4, 1))
    request = make_request("request-1", [1])
    scheduler.add_request(request)
    pending_plan = scheduler.schedule()
    foreign_plan = SchedulerOutput(pending_plan.scheduled_requests)

    with pytest.raises(ValueError, match="not pending"):
        scheduler.update(foreign_plan, make_output({"request-1": 2}))

    assert request.num_computed_tokens == 0
    assert scheduler.schedule() is pending_plan


def test_abort_handles_waiting_and_running_requests() -> None:
    scheduler = Scheduler(SchedulerConfig(4, 1))
    waiting = make_request("waiting", [1])
    scheduler.add_request(waiting)
    assert scheduler.abort_request("waiting") is True
    assert waiting.finish_reason is FinishReason.ABORTED
    assert scheduler.abort_request("waiting") is False
    assert scheduler.abort_request("unknown") is False

    running = make_request("running", [2])
    scheduler.add_request(running)
    pending_plan = scheduler.schedule()
    assert scheduler.abort_request("running") is True
    with pytest.raises(ValueError, match="not pending"):
        scheduler.update(pending_plan, make_output({"running": 3}))
    assert running.finish_reason is FinishReason.ABORTED


def test_active_request_without_pending_tokens_is_rejected() -> None:
    scheduler = Scheduler(SchedulerConfig(4, 1))
    request = Request(
        request_id="request-1",
        prompt_token_ids=[1],
        sampling_params=SamplingParams(max_tokens=2),
        num_computed_tokens=1,
    )
    scheduler.add_request(request)

    with pytest.raises(RuntimeError, match="active request must have pending tokens"):
        scheduler.schedule()
