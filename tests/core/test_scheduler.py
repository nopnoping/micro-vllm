"""Tests for unified token-budget scheduling and result updates."""

import pytest

from micro_vllm.config import CacheConfig, SchedulerConfig
from micro_vllm.core.kv_cache_manager import KVCacheManager
from micro_vllm.core.prefix_cache import hash_block
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


def make_scheduler(
    token_budget: int,
    sequence_budget: int,
    *,
    block_size: int = 4,
    num_blocks: int = 128,
    enable_prefix_cache: bool = False,
    cache_namespace: str = "default",
) -> Scheduler:
    return Scheduler(
        SchedulerConfig(token_budget, sequence_budget),
        KVCacheManager(
            CacheConfig(
                block_size=block_size,
                num_blocks=num_blocks,
                enable_prefix_cache=enable_prefix_cache,
            )
        ),
        cache_namespace=cache_namespace,
    )


def test_scheduler_requires_explicit_kv_cache_manager() -> None:
    with pytest.raises(TypeError):
        Scheduler(SchedulerConfig(4, 1))  # type: ignore[call-arg]


def test_scheduler_owns_request_while_manager_only_registers_cache_state() -> None:
    manager = KVCacheManager(CacheConfig(block_size=4, num_blocks=8))
    scheduler = Scheduler(SchedulerConfig(4, 1), manager)
    request = make_request("request-1", [1, 2, 3])
    request_state_before = (
        request.status,
        request.num_computed_tokens,
        tuple(request.output_token_ids),
    )

    scheduler.add_request(request)

    assert scheduler._kv_cache_manager is manager
    assert scheduler.get_request("request-1") is request
    assert manager.get_block_table("request-1") == ()
    assert (
        request.status,
        request.num_computed_tokens,
        tuple(request.output_token_ids),
    ) == request_state_before


def test_each_scheduler_uses_an_independent_cache_manager() -> None:
    first_manager = KVCacheManager(CacheConfig(block_size=4, num_blocks=8))
    second_manager = KVCacheManager(CacheConfig(block_size=4, num_blocks=8))
    first_scheduler = Scheduler(SchedulerConfig(4, 1), first_manager)
    second_scheduler = Scheduler(SchedulerConfig(4, 1), second_manager)

    first_scheduler.add_request(make_request("request-1", [1]))
    second_scheduler.add_request(make_request("request-1", [2]))

    assert first_scheduler._kv_cache_manager is first_manager
    assert second_scheduler._kv_cache_manager is second_manager
    assert first_manager is not second_manager
    assert first_manager.get_block_table("request-1") == ()
    assert second_manager.get_block_table("request-1") == ()


def test_single_request_full_prompt_is_scheduled_in_one_step() -> None:
    scheduler = make_scheduler(4, 2)
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
    scheduler = make_scheduler(4, 1)
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


def test_prompt_chunks_append_blocks_only_when_crossing_boundary() -> None:
    scheduler = make_scheduler(3, 1, block_size=4, num_blocks=4)
    request = make_request("request-1", [1, 2, 3, 4, 5])
    scheduler.add_request(request)

    first_plan = scheduler.schedule()
    first = first_plan.scheduled_requests[0]
    assert first.num_scheduled_tokens == 3
    assert first.block_table == (0,)
    scheduler.update(first_plan, ModelRunnerOutput())

    second_plan = scheduler.schedule()
    second = second_plan.scheduled_requests[0]
    assert second.num_computed_tokens == 3
    assert second.num_scheduled_tokens == 2
    assert second.block_table == (0, 1)
    assert len(second.block_table) * 4 >= 5


def test_sampled_token_gets_a_slot_only_on_next_schedule() -> None:
    scheduler = make_scheduler(4, 1, block_size=4, num_blocks=2)
    request = make_request("request-1", [1, 2, 3, 4], max_tokens=2)
    scheduler.add_request(request)

    first_plan = scheduler.schedule()
    assert first_plan.scheduled_requests[0].block_table == (0,)
    scheduler.update(first_plan, make_output({"request-1": 7}))
    assert scheduler._kv_cache_manager.get_block_table("request-1") == (0,)

    second_plan = scheduler.schedule()
    second = second_plan.scheduled_requests[0]
    assert second.input_token_ids == (7,)
    assert second.num_computed_tokens == 4
    assert second.block_table == (0, 1)


def test_pending_plan_is_reused_without_allocating_blocks_twice() -> None:
    scheduler = make_scheduler(8, 1, block_size=4, num_blocks=4)
    scheduler.add_request(make_request("request-1", [1, 2, 3, 4, 5]))

    plan = scheduler.schedule()
    table_before = scheduler._kv_cache_manager.get_block_table("request-1")
    free_before = scheduler._kv_cache_manager._block_pool.free_block_ids

    assert scheduler.schedule() is plan
    assert scheduler._kv_cache_manager.get_block_table("request-1") == table_before
    assert scheduler._kv_cache_manager._block_pool.free_block_ids == free_before


def test_cache_capacity_failure_stops_waiting_admission_in_fifo_order() -> None:
    scheduler = make_scheduler(5, 2, block_size=4, num_blocks=1)
    first = make_request("first", [1, 2, 3, 4, 5])
    second = make_request("second", [6])
    scheduler.add_request(first)
    scheduler.add_request(second)

    plan = scheduler.schedule()

    assert plan.scheduled_requests == ()
    assert first.status is RequestStatus.WAITING
    assert second.status is RequestStatus.WAITING
    assert scheduler._kv_cache_manager.get_block_table("first") == ()
    assert scheduler._kv_cache_manager.get_block_table("second") == ()
    assert scheduler._kv_cache_manager._block_pool.free_block_ids == (0,)


def test_capacity_failure_keeps_prior_success_and_later_request_waiting() -> None:
    scheduler = make_scheduler(9, 2, block_size=4, num_blocks=2)
    first = make_request("first", [1, 2, 3, 4])
    second = make_request("second", [5, 6, 7, 8, 9])
    scheduler.add_request(first)
    scheduler.add_request(second)

    plan = scheduler.schedule()

    assert [item.request_id for item in plan.scheduled_requests] == ["first"]
    assert plan.scheduled_requests[0].block_table == (0,)
    assert first.status is RequestStatus.RUNNING
    assert second.status is RequestStatus.WAITING
    assert scheduler._kv_cache_manager.get_block_table("second") == ()
    assert scheduler._kv_cache_manager._block_pool.free_block_ids == (1,)


def test_intermediate_prompt_chunk_rejects_sampled_token() -> None:
    scheduler = make_scheduler(2, 1)
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
    scheduler = make_scheduler(4, 2)
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
    scheduler = make_scheduler(5, 4)
    for index in range(4):
        scheduler.add_request(
            make_request(f"request-{index}", [index, index + 1, index + 2])
        )

    plan = scheduler.schedule()

    assert plan.total_num_scheduled_tokens == 5
    assert all(item.num_scheduled_tokens > 0 for item in plan.scheduled_requests)


def test_sequence_budget_is_never_exceeded() -> None:
    scheduler = make_scheduler(100, 2)
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
    scheduler = make_scheduler(1, 1)
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
    scheduler = make_scheduler(8, 3)
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
    scheduler = make_scheduler(4, 2)
    first = make_request("request-1", [1])
    duplicate = make_request("request-1", [2])
    scheduler.add_request(first)

    with pytest.raises(ValueError, match="duplicate request_id"):
        scheduler.add_request(duplicate)

    assert list(scheduler._waiting) == [first]
    assert scheduler.get_request("request-1") is first


def test_invalid_executor_outputs_fail_atomically_and_can_be_retried() -> None:
    scheduler = make_scheduler(4, 2)
    first = make_request("first", [1])
    second = make_request("second", [2])
    scheduler.add_request(first)
    scheduler.add_request(second)
    plan = scheduler.schedule()
    first_table = scheduler._kv_cache_manager.get_block_table("first")
    second_table = scheduler._kv_cache_manager.get_block_table("second")
    free_blocks = scheduler._kv_cache_manager._block_pool.free_block_ids

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
    assert scheduler._kv_cache_manager.get_block_table("first") == first_table
    assert scheduler._kv_cache_manager.get_block_table("second") == second_table
    assert scheduler._kv_cache_manager._block_pool.free_block_ids == free_blocks
    assert scheduler.schedule() is plan

    scheduler.update(plan, make_output({"first": 3, "second": 4}))
    assert first.num_computed_tokens == second.num_computed_tokens == 1


def test_successful_update_commits_prefix_before_finished_request_release() -> None:
    scheduler = make_scheduler(
        2,
        1,
        block_size=2,
        num_blocks=2,
        enable_prefix_cache=True,
        cache_namespace="model-a",
    )
    request = make_request("request-1", [1, 2], max_tokens=1)
    scheduler.add_request(request)
    plan = scheduler.schedule()
    pool = scheduler._kv_cache_manager._block_pool

    with pytest.raises(ValueError, match="missing sampled token"):
        scheduler.update(plan, ModelRunnerOutput())
    assert pool.num_cached_blocks == 0

    scheduler.update(plan, make_output({"request-1": 7}))

    block_hash = hash_block(
        None,
        (1, 2),
        block_size=2,
        cache_namespace="model-a",
    )
    cached_block = pool.get_cached_block(block_hash)
    assert request.finish_reason is FinishReason.LENGTH
    assert cached_block is not None
    assert cached_block.block_id == 0
    assert cached_block.ref_count == 0
    assert pool.free_block_ids == (1, 0)


def test_successful_update_does_not_commit_partial_block() -> None:
    scheduler = make_scheduler(
        1,
        1,
        block_size=2,
        enable_prefix_cache=True,
    )
    request = make_request("request-1", [1], max_tokens=2)
    scheduler.add_request(request)
    plan = scheduler.schedule()

    scheduler.update(plan, make_output({"request-1": 7}))

    assert scheduler._kv_cache_manager._block_pool.num_cached_blocks == 0


def test_waiting_request_uses_prefix_hit_before_admission() -> None:
    scheduler = make_scheduler(
        5,
        1,
        block_size=2,
        num_blocks=4,
        enable_prefix_cache=True,
        cache_namespace="model-a",
    )
    first = make_request("first", [1, 2, 3, 4, 5], max_tokens=1)
    scheduler.add_request(first)
    first_plan = scheduler.schedule()
    scheduler.update(first_plan, make_output({"first": 9}))

    second = make_request("second", [1, 2, 3, 4, 5], max_tokens=1)
    scheduler.add_request(second)
    second_plan = scheduler.schedule()

    assert second.status is RequestStatus.RUNNING
    assert second.num_computed_tokens == 4
    assert len(second_plan.scheduled_requests) == 1
    scheduled = second_plan.scheduled_requests[0]
    assert scheduled.input_token_ids == (5,)
    assert scheduled.num_computed_tokens == 4
    assert scheduled.num_scheduled_tokens == 1
    assert scheduled.requires_sampling is True
    assert scheduled.block_table[:2] == (0, 1)


def test_waiting_prefix_hit_rolls_back_when_slot_allocation_fails() -> None:
    manager = KVCacheManager(
        CacheConfig(
            block_size=2,
            num_blocks=2,
            enable_prefix_cache=True,
        )
    )
    manager.register_request("seed")
    assert manager.allocate_slots("seed", 0, 4) is not None
    assert manager.commit_prefix_blocks(
        "seed",
        (1, 2, 3, 4),
        4,
        cache_namespace="model-a",
    ) == (0, 1)
    manager.free_request("seed")
    scheduler = Scheduler(
        SchedulerConfig(1, 1),
        manager,
        cache_namespace="model-a",
    )
    request = make_request("request-1", [1, 2, 3, 4, 5])
    scheduler.add_request(request)

    assert scheduler.schedule().scheduled_requests == ()
    assert request.status is RequestStatus.WAITING
    assert request.num_computed_tokens == 0
    assert manager.get_block_table("request-1") == ()
    assert manager._block_pool.ref_count(0) == 0
    assert manager._block_pool.ref_count(1) == 0

    # A retry performs a fresh read-only lookup and rolls back again.
    assert scheduler.schedule().scheduled_requests == ()
    assert request.status is RequestStatus.WAITING
    assert request.num_computed_tokens == 0
    assert manager.get_block_table("request-1") == ()
    assert manager._block_pool.ref_count(0) == 0
    assert manager._block_pool.ref_count(1) == 0


def test_eos_finishes_request_while_ignore_eos_continues() -> None:
    scheduler = make_scheduler(4, 2)
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
    scheduler = make_scheduler(4, 1, block_size=4, num_blocks=2)
    request = make_request("request-1", [1], max_tokens=1)
    scheduler.add_request(request)
    plan = scheduler.schedule()

    scheduler.update(plan, make_output({"request-1": 2}))

    assert request.finish_reason is FinishReason.LENGTH
    assert request not in scheduler._running
    assert scheduler.has_unfinished_requests() is False
    assert scheduler._kv_cache_manager._block_pool.num_free_blocks == 2
    assert scheduler._kv_cache_manager._block_pool.free_block_ids == (1, 0)
    with pytest.raises(KeyError, match="unknown request_id: request-1"):
        scheduler._kv_cache_manager.get_block_table("request-1")


def test_non_finished_update_keeps_request_cache_reserved() -> None:
    scheduler = make_scheduler(8, 1, block_size=4, num_blocks=3)
    request = make_request("request-1", [1, 2, 3, 4, 5], max_tokens=2)
    scheduler.add_request(request)
    plan = scheduler.schedule()
    assert plan.scheduled_requests[0].block_table == (0, 1)

    scheduler.update(plan, make_output({"request-1": 7}))

    assert request.status is RequestStatus.RUNNING
    assert scheduler._kv_cache_manager.get_block_table("request-1") == (0, 1)
    assert scheduler._kv_cache_manager._block_pool.free_block_ids == (2,)


def test_successful_update_clears_pending_plan_after_cache_commit() -> None:
    scheduler = make_scheduler(4, 1, block_size=4, num_blocks=2)
    request = make_request("request-1", [1], max_tokens=2)
    scheduler.add_request(request)
    first_plan = scheduler.schedule()

    scheduler.update(first_plan, make_output({"request-1": 2}))

    assert scheduler._pending_plan is None
    second_plan = scheduler.schedule()
    assert second_plan is not first_plan
    assert second_plan.scheduled_requests[0].input_token_ids == (2,)


def test_plan_cannot_be_updated_twice() -> None:
    scheduler = make_scheduler(4, 1)
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
    scheduler = make_scheduler(4, 1)
    request = make_request("request-1", [1])
    scheduler.add_request(request)
    pending_plan = scheduler.schedule()
    foreign_plan = SchedulerOutput(pending_plan.scheduled_requests)
    block_table = scheduler._kv_cache_manager.get_block_table("request-1")
    free_blocks = scheduler._kv_cache_manager._block_pool.free_block_ids

    with pytest.raises(ValueError, match="not pending"):
        scheduler.update(foreign_plan, make_output({"request-1": 2}))

    assert request.num_computed_tokens == 0
    assert scheduler._kv_cache_manager.get_block_table("request-1") == block_table
    assert scheduler._kv_cache_manager._block_pool.free_block_ids == free_blocks
    assert scheduler.schedule() is pending_plan


def test_abort_handles_waiting_and_running_requests() -> None:
    scheduler = make_scheduler(4, 1, block_size=4, num_blocks=2)
    waiting = make_request("waiting", [1])
    scheduler.add_request(waiting)
    free_blocks_before = scheduler._kv_cache_manager._block_pool.free_block_ids
    assert scheduler.abort_request("waiting") is True
    assert waiting.finish_reason is FinishReason.ABORTED
    assert scheduler._kv_cache_manager._block_pool.free_block_ids == free_blocks_before
    with pytest.raises(KeyError, match="unknown request_id: waiting"):
        scheduler._kv_cache_manager.get_block_table("waiting")
    assert scheduler.abort_request("waiting") is False
    assert scheduler.abort_request("unknown") is False

    running = make_request("running", [2, 3, 4, 5])
    scheduler.add_request(running)
    pending_plan = scheduler.schedule()
    assert pending_plan.scheduled_requests[0].block_table == (0,)
    assert scheduler._kv_cache_manager._block_pool.num_free_blocks == 1
    assert scheduler.abort_request("running") is True
    with pytest.raises(ValueError, match="not pending"):
        scheduler.update(pending_plan, make_output({"running": 3}))
    assert running.finish_reason is FinishReason.ABORTED
    assert scheduler._kv_cache_manager._block_pool.num_free_blocks == 2
    assert scheduler._kv_cache_manager._block_pool.free_block_ids == (1, 0)
    with pytest.raises(KeyError, match="unknown request_id: running"):
        scheduler._kv_cache_manager.get_block_table("running")


def test_abort_pending_request_preserves_other_request_reservation() -> None:
    scheduler = make_scheduler(8, 2, block_size=4, num_blocks=2)
    first = make_request("first", [1, 2, 3, 4])
    second = make_request("second", [5, 6, 7, 8])
    scheduler.add_request(first)
    scheduler.add_request(second)
    pending_plan = scheduler.schedule()
    assert [item.block_table for item in pending_plan.scheduled_requests] == [
        (0,),
        (1,),
    ]

    assert scheduler.abort_request("first") is True

    assert first.finish_reason is FinishReason.ABORTED
    assert second.status is RequestStatus.RUNNING
    assert scheduler._pending_plan is None
    assert scheduler._kv_cache_manager.get_block_table("second") == (1,)
    assert scheduler._kv_cache_manager._block_pool.ref_count(0) == 0
    assert scheduler._kv_cache_manager._block_pool.ref_count(1) == 1
    assert scheduler._kv_cache_manager._block_pool.free_block_ids == (0,)

    rebuilt_plan = scheduler.schedule()
    assert [item.request_id for item in rebuilt_plan.scheduled_requests] == ["second"]
    assert rebuilt_plan.scheduled_requests[0].block_table == (1,)
    assert scheduler._kv_cache_manager._block_pool.free_block_ids == (0,)


def test_abort_finished_request_does_not_release_cache_twice() -> None:
    scheduler = make_scheduler(4, 1, block_size=4, num_blocks=2)
    request = make_request("request-1", [1], max_tokens=1)
    scheduler.add_request(request)
    plan = scheduler.schedule()
    scheduler.update(plan, make_output({"request-1": 2}))
    free_blocks_before = scheduler._kv_cache_manager._block_pool.free_block_ids

    assert scheduler.abort_request("request-1") is False

    assert scheduler._kv_cache_manager._block_pool.free_block_ids == free_blocks_before
    assert scheduler._kv_cache_manager._block_pool.ref_count(0) == 0


def test_active_request_without_pending_tokens_is_rejected() -> None:
    scheduler = make_scheduler(4, 1)
    request = Request(
        request_id="request-1",
        prompt_token_ids=[1],
        sampling_params=SamplingParams(max_tokens=2),
        num_computed_tokens=1,
    )
    scheduler.add_request(request)

    with pytest.raises(RuntimeError, match="active request must have pending tokens"):
        scheduler.schedule()
