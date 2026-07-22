"""Tests for request state, invariants, and lifecycle operations."""

from collections.abc import Callable

import pytest

from micro_vllm.core.request import FinishReason, Request, RequestStatus
from micro_vllm.sampling.params import SamplingParams


def make_request(
    *,
    request_id: str = "request-1",
    prompt_token_ids: list[int] | None = None,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3]
        if prompt_token_ids is None
        else prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=4, eos_token_id=0),
    )


def test_initial_state_and_computed_properties() -> None:
    prompt_token_ids = [1, 2, 3]
    request = make_request(prompt_token_ids=prompt_token_ids)

    prompt_token_ids.append(5)

    assert request.request_id == "request-1"
    assert request.prompt_token_ids == [1, 2, 3]
    assert request.output_token_ids == []
    assert request.num_prompt_tokens == 3
    assert request.num_output_tokens == 0
    assert request.num_known_tokens == 3
    assert request.num_computed_tokens == 0
    assert request.num_pending_tokens == 3
    assert request.all_token_ids == (1, 2, 3)
    assert request.status is RequestStatus.WAITING
    assert request.finish_reason is None
    assert request.is_finished is False


def test_all_token_ids_is_an_immutable_snapshot() -> None:
    request = make_request()
    request.append_sampled_token(4)

    snapshot = request.all_token_ids
    request.output_token_ids.append(5)

    assert isinstance(snapshot, tuple)
    assert snapshot == (1, 2, 3, 4)
    assert request.all_token_ids == (1, 2, 3, 4, 5)


def test_stochastic_request_assigns_one_effective_seed() -> None:
    params = SamplingParams(max_tokens=4, temperature=1.0)

    request = Request("request", [1], params)

    assert params.seed is None
    assert request.sampling_params is not params
    assert request.sampling_params.seed is not None
    assert 0 <= request.sampling_params.seed < 2**63


def test_explicit_seed_and_greedy_params_are_preserved() -> None:
    seeded = SamplingParams(max_tokens=4, temperature=1.0, seed=7)
    greedy = SamplingParams(max_tokens=4)

    seeded_request = Request("seeded", [1], seeded)
    greedy_request = Request("greedy", [1], greedy)

    assert seeded_request.sampling_params is seeded
    assert greedy_request.sampling_params is greedy
    assert greedy_request.sampling_params.seed is None


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


def test_advance_computed_tokens_updates_pending_tokens() -> None:
    request = make_request()
    request.transition_to(RequestStatus.RUNNING)

    request.advance_computed_tokens(2)

    assert request.num_computed_tokens == 2
    assert request.num_pending_tokens == 1


def test_preempt_and_resume_reset_only_computed_progress() -> None:
    request = make_request()
    request.append_sampled_token(4)
    request.resume(3)

    request.preempt()

    assert request.status is RequestStatus.WAITING
    assert request.num_computed_tokens == 0
    assert request.all_token_ids == (1, 2, 3, 4)

    request.resume(2)

    assert request.status is RequestStatus.RUNNING
    assert request.num_computed_tokens == 2
    assert request.all_token_ids == (1, 2, 3, 4)


@pytest.mark.parametrize("request_id", [""])
def test_request_rejects_empty_request_id(request_id: str) -> None:
    with pytest.raises(ValueError, match="request_id must not be empty"):
        make_request(request_id=request_id)


def test_request_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="prompt_token_ids must not be empty"):
        make_request(prompt_token_ids=[])


def test_request_rejects_negative_prompt_token_ids() -> None:
    with pytest.raises(ValueError, match=r"prompt_token_ids\[0\]"):
        make_request(prompt_token_ids=[-1])


def test_legal_status_transitions() -> None:
    request = make_request()

    request.transition_to(RequestStatus.RUNNING)
    assert request.status is RequestStatus.RUNNING
    request.transition_to(RequestStatus.WAITING)
    assert request.status is RequestStatus.WAITING
    request.transition_to(RequestStatus.RUNNING)
    assert request.status is RequestStatus.RUNNING


@pytest.mark.parametrize(
    ("operation", "target_status"),
    [
        (lambda request: None, RequestStatus.WAITING),
        (
            lambda request: request.transition_to(RequestStatus.RUNNING),
            RequestStatus.RUNNING,
        ),
        (
            lambda request: request.finish(FinishReason.LENGTH),
            RequestStatus.RUNNING,
        ),
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


def test_waiting_requests_can_be_aborted() -> None:
    waiting = make_request(request_id="waiting")
    victim = make_request(request_id="victim")
    victim.resume()
    victim.preempt()

    waiting.finish(FinishReason.ABORTED)
    victim.finish(FinishReason.ABORTED)

    assert waiting.finish_reason is FinishReason.ABORTED
    assert victim.finish_reason is FinishReason.ABORTED
