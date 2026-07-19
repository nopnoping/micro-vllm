"""Tests for request state, invariants, and lifecycle operations."""

from collections.abc import Callable

import pytest

from micro_vllm.core.request import FinishReason, Request, RequestStatus
from micro_vllm.sampling.params import SamplingParams


def make_request(
    *,
    request_id: str = "request-1",
    prompt_token_ids: list[int] | None = None,
    output_token_ids: list[int] | None = None,
    num_computed_tokens: int = 0,
    status: RequestStatus = RequestStatus.WAITING,
    finish_reason: FinishReason | None = None,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3]
        if prompt_token_ids is None
        else prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=4, eos_token_id=0),
        output_token_ids=[] if output_token_ids is None else output_token_ids,
        num_computed_tokens=num_computed_tokens,
        status=status,
        finish_reason=finish_reason,
    )


def test_initial_state_and_computed_properties() -> None:
    prompt_token_ids = [1, 2, 3]
    output_token_ids = [4]
    request = make_request(
        prompt_token_ids=prompt_token_ids,
        output_token_ids=output_token_ids,
        num_computed_tokens=2,
    )

    prompt_token_ids.append(5)
    output_token_ids.append(6)

    assert request.request_id == "request-1"
    assert request.prompt_token_ids == [1, 2, 3]
    assert request.output_token_ids == [4]
    assert request.num_prompt_tokens == 3
    assert request.num_output_tokens == 1
    assert request.num_known_tokens == 4
    assert request.num_computed_tokens == 2
    assert request.num_pending_tokens == 2
    assert request.all_token_ids == (1, 2, 3, 4)
    assert request.status is RequestStatus.WAITING
    assert request.finish_reason is None
    assert request.is_finished is False


def test_all_token_ids_is_an_immutable_snapshot() -> None:
    request = make_request(output_token_ids=[4])

    snapshot = request.all_token_ids
    request.output_token_ids.append(5)

    assert isinstance(snapshot, tuple)
    assert snapshot == (1, 2, 3, 4)
    assert request.all_token_ids == (1, 2, 3, 4, 5)


def test_append_sampled_token_updates_known_and_pending_tokens() -> None:
    request = make_request()
    request.transition_to(RequestStatus.RUNNING)
    request.advance_computed_tokens(3)

    request.append_sampled_token(4)

    assert request.output_token_ids == [4]
    assert request.num_output_tokens == 1
    assert request.num_known_tokens == 4
    assert request.num_computed_tokens == 3
    assert request.num_pending_tokens == 1
    assert request.all_token_ids == (1, 2, 3, 4)


def test_append_sampled_token_requires_running_request_with_no_pending_tokens() -> None:
    waiting = make_request()
    with pytest.raises(ValueError, match="running request"):
        waiting.append_sampled_token(4)

    running = make_request()
    running.transition_to(RequestStatus.RUNNING)
    with pytest.raises(ValueError, match="known tokens are pending"):
        running.append_sampled_token(4)

    assert waiting.output_token_ids == []
    assert running.output_token_ids == []


def test_advance_computed_tokens_updates_pending_tokens() -> None:
    request = make_request()
    request.transition_to(RequestStatus.RUNNING)

    request.advance_computed_tokens(2)

    assert request.num_computed_tokens == 2
    assert request.num_pending_tokens == 1


@pytest.mark.parametrize("num_tokens", [0, -1])
def test_advance_computed_tokens_rejects_non_positive_count(num_tokens: int) -> None:
    request = make_request()
    request.transition_to(RequestStatus.RUNNING)

    with pytest.raises(ValueError, match="greater than zero"):
        request.advance_computed_tokens(num_tokens)

    assert request.num_computed_tokens == 0


def test_advance_computed_tokens_rejects_count_beyond_pending_tokens() -> None:
    request = make_request()
    request.transition_to(RequestStatus.RUNNING)

    with pytest.raises(ValueError, match="must not exceed num_pending_tokens"):
        request.advance_computed_tokens(4)

    assert request.num_computed_tokens == 0


@pytest.mark.parametrize("num_computed_tokens", [-1, 4])
def test_initial_computed_tokens_must_be_within_known_tokens(
    num_computed_tokens: int,
) -> None:
    with pytest.raises(
        ValueError, match="between zero and num_known_tokens"
    ):
        make_request(num_computed_tokens=num_computed_tokens)


@pytest.mark.parametrize("request_id", [""])
def test_request_rejects_empty_request_id(request_id: str) -> None:
    with pytest.raises(ValueError, match="request_id must not be empty"):
        make_request(request_id=request_id)


def test_request_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="prompt_token_ids must not be empty"):
        make_request(prompt_token_ids=[])


@pytest.mark.parametrize("field_name", ["prompt_token_ids", "output_token_ids"])
def test_request_rejects_negative_token_ids(field_name: str) -> None:
    values = {field_name: [-1]}

    with pytest.raises(ValueError, match=rf"{field_name}\[0\]"):
        make_request(**values)


def test_legal_status_transitions() -> None:
    request = make_request()

    request.transition_to(RequestStatus.RUNNING)
    assert request.status is RequestStatus.RUNNING
    request.transition_to(RequestStatus.PREEMPTED)
    assert request.status is RequestStatus.PREEMPTED
    request.transition_to(RequestStatus.RUNNING)
    assert request.status is RequestStatus.RUNNING


@pytest.mark.parametrize(
    ("operation", "target_status"),
    [
        (lambda request: None, RequestStatus.PREEMPTED),
        (lambda request: request.transition_to(RequestStatus.RUNNING), RequestStatus.WAITING),
        (lambda request: request.transition_to(RequestStatus.RUNNING), RequestStatus.FINISHED),
    ],
)
def test_illegal_status_transitions(
    operation: Callable[[Request], None], target_status: RequestStatus
) -> None:
    request = make_request()
    operation(request)

    with pytest.raises(ValueError, match="cannot transition request"):
        request.transition_to(target_status)


@pytest.mark.parametrize("reason", list(FinishReason))
def test_running_request_can_finish_with_each_reason(reason: FinishReason) -> None:
    request = make_request()
    request.transition_to(RequestStatus.RUNNING)

    request.finish(reason)

    assert request.status is RequestStatus.FINISHED
    assert request.finish_reason is reason
    assert request.is_finished is True


def test_waiting_and_preempted_requests_can_be_aborted() -> None:
    waiting = make_request(request_id="waiting")
    preempted = make_request(request_id="preempted")
    preempted.transition_to(RequestStatus.RUNNING)
    preempted.transition_to(RequestStatus.PREEMPTED)

    waiting.finish(FinishReason.ABORTED)
    preempted.finish(FinishReason.ABORTED)

    assert waiting.finish_reason is FinishReason.ABORTED
    assert preempted.finish_reason is FinishReason.ABORTED


def test_finish_rejects_illegal_state_and_preserves_request() -> None:
    request = make_request()

    with pytest.raises(ValueError, match="cannot finish WAITING request with EOS"):
        request.finish(FinishReason.EOS)

    assert request.status is RequestStatus.WAITING
    assert request.finish_reason is None


@pytest.mark.parametrize(
    ("status", "finish_reason", "message"),
    [
        (RequestStatus.FINISHED, None, "finished request must have a finish_reason"),
        (RequestStatus.WAITING, FinishReason.ABORTED, "unfinished request"),
    ],
)
def test_initial_status_and_finish_reason_must_be_consistent(
    status: RequestStatus,
    finish_reason: FinishReason | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_request(status=status, finish_reason=finish_reason)
