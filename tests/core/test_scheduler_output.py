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
    )


@pytest.mark.parametrize(
    ("field_name", "new_value"),
    [
        ("request_id", "request-2"),
        ("input_token_ids", (3,)),
        ("num_computed_tokens", 1),
        ("num_scheduled_tokens", 1),
        ("requires_sampling", False),
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
            ),
        )
    )

    assert output.total_num_scheduled_tokens == 3
    assert SchedulerOutput(()).total_num_scheduled_tokens == 0


@pytest.mark.parametrize(
    ("field_name", "value", "error_type", "message"),
    [
        ("num_computed_tokens", -1, ValueError, "must be non-negative"),
        ("num_computed_tokens", True, TypeError, "must be an integer"),
        ("num_scheduled_tokens", 0, ValueError, "must be greater than zero"),
        ("num_scheduled_tokens", -1, ValueError, "must be greater than zero"),
        ("num_scheduled_tokens", True, TypeError, "must be an integer"),
        ("requires_sampling", 1, TypeError, "must be a boolean"),
    ],
)
def test_scheduled_request_rejects_invalid_fields(
    field_name: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    values = {
        "request_id": "request-1",
        "input_token_ids": (1, 2),
        "num_computed_tokens": 0,
        "num_scheduled_tokens": 2,
        "requires_sampling": True,
    }
    values[field_name] = value

    with pytest.raises(error_type, match=message):
        ScheduledRequest(**values)


@pytest.mark.parametrize("token_id", [-1, True, 1.5])
def test_scheduled_request_rejects_invalid_input_token_ids(
    token_id: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match=r"input_token_ids\[0\]"):
        ScheduledRequest("request-1", (token_id,), 0, 1, True)


def test_scheduled_request_rejects_inconsistent_token_count() -> None:
    with pytest.raises(
        ValueError,
        match="num_scheduled_tokens must equal the number of input_token_ids",
    ):
        ScheduledRequest("request-1", (1, 2), 0, 1, True)


def test_scheduler_output_rejects_duplicate_request_ids() -> None:
    first = make_scheduled_request()
    duplicate = ScheduledRequest("request-1", (3,), 2, 1, True)

    with pytest.raises(ValueError, match="duplicate request_id"):
        SchedulerOutput((first, duplicate))


def test_scheduler_output_rejects_non_scheduled_request_values() -> None:
    with pytest.raises(TypeError, match="must contain ScheduledRequest"):
        SchedulerOutput((make_scheduled_request(), object()))
