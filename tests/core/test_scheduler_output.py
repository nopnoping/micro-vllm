"""Tests for immutable scheduler execution-plan contracts."""

from dataclasses import FrozenInstanceError

import pytest

from micro_vllm.core.scheduler_output import ScheduledRequest, SchedulerOutput


def make_scheduled_request() -> ScheduledRequest:
    return ScheduledRequest(
        request_id="request-1",
        input_token_ids=(1, 2),
        num_computed_tokens=0,
        num_scheduled_tokens=2,
        requires_sampling=True,
        block_table=(0,),
    )


@pytest.mark.parametrize(
    ("field_name", "new_value"),
    [
        ("request_id", "request-2"),
        ("input_token_ids", (3,)),
        ("num_computed_tokens", 1),
        ("num_scheduled_tokens", 1),
        ("requires_sampling", False),
        ("block_table", (1,)),
    ],
)
def test_scheduled_request_is_frozen(field_name: str, new_value: object) -> None:
    scheduled_request = make_scheduled_request()

    with pytest.raises(FrozenInstanceError):
        setattr(scheduled_request, field_name, new_value)

    assert isinstance(scheduled_request.input_token_ids, tuple)


def test_scheduler_output_is_frozen() -> None:
    output = SchedulerOutput((make_scheduled_request(),))

    with pytest.raises(FrozenInstanceError):
        output.scheduled_requests = ()

    assert isinstance(output.scheduled_requests, tuple)


def test_original_token_list_does_not_change_plan_snapshot() -> None:
    input_token_ids = [1, 2]

    scheduled_request = ScheduledRequest(
        request_id="request-1",
        input_token_ids=input_token_ids,
        num_computed_tokens=0,
        num_scheduled_tokens=2,
        requires_sampling=True,
        block_table=(0,),
    )
    input_token_ids[0] = 9
    input_token_ids.append(3)

    assert scheduled_request.input_token_ids == (1, 2)


def test_total_num_scheduled_tokens_is_computed_from_requests() -> None:
    output = SchedulerOutput(
        (
            make_scheduled_request(),
            ScheduledRequest(
                request_id="request-2",
                input_token_ids=(3,),
                num_computed_tokens=4,
                num_scheduled_tokens=1,
                requires_sampling=False,
                block_table=(1,),
            ),
        )
    )

    assert output.total_num_scheduled_tokens == 3
    assert SchedulerOutput(()).total_num_scheduled_tokens == 0


def test_original_block_table_list_does_not_change_plan_snapshot() -> None:
    block_table = [0, None]
    scheduled_request = ScheduledRequest(
        request_id="request-1",
        input_token_ids=(1,),
        num_computed_tokens=0,
        num_scheduled_tokens=1,
        requires_sampling=True,
        block_table=block_table,
    )

    block_table[0] = 9
    block_table.append(2)

    assert scheduled_request.block_table == (0, None)
    assert isinstance(scheduled_request.block_table, tuple)
