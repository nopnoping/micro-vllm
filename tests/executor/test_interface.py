"""Tests for the executor input and output contracts."""

from dataclasses import FrozenInstanceError

import pytest

from micro_vllm.core.scheduler_output import ScheduledRequest, SchedulerOutput
from micro_vllm.executor.interface import ModelRunnerOutput, SampledToken


def make_plan() -> SchedulerOutput:
    return SchedulerOutput(
        (
            ScheduledRequest("sampling", (1,), 0, 1, True),
            ScheduledRequest("no-sampling", (2,), 0, 1, False),
        )
    )


def test_model_runner_output_allows_no_sampled_tokens() -> None:
    assert ModelRunnerOutput().sampled_tokens == ()


def test_model_runner_output_owns_an_immutable_snapshot() -> None:
    sampled_token = SampledToken("sampling", 3)
    source = [sampled_token]

    output = ModelRunnerOutput(source)
    source.clear()

    assert output.sampled_tokens == (sampled_token,)
    with pytest.raises(FrozenInstanceError):
        output.sampled_tokens = ()


@pytest.mark.parametrize(
    ("request_id", "token_id", "error_type", "message"),
    [
        ("", 1, ValueError, "request_id must not be empty"),
        ("request-1", -1, ValueError, "token_id must be non-negative"),
        ("request-1", True, TypeError, "token_id must be an integer"),
    ],
)
def test_sampled_token_rejects_invalid_values(
    request_id: str,
    token_id: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        SampledToken(request_id, token_id)


def test_model_runner_output_rejects_duplicate_request_ids() -> None:
    with pytest.raises(ValueError, match="duplicate request_id"):
        ModelRunnerOutput(
            (SampledToken("sampling", 3), SampledToken("sampling", 4))
        )


def test_model_runner_output_accepts_known_sampling_request() -> None:
    output = ModelRunnerOutput((SampledToken("sampling", 3),))

    output.validate_for(make_plan())


def test_model_runner_output_rejects_unknown_request_id() -> None:
    output = ModelRunnerOutput((SampledToken("unknown", 3),))

    with pytest.raises(ValueError, match="unknown sampled token request_id"):
        output.validate_for(make_plan())


def test_model_runner_output_rejects_token_for_non_sampling_request() -> None:
    output = ModelRunnerOutput((SampledToken("no-sampling", 3),))

    with pytest.raises(ValueError, match="does not require sampling"):
        output.validate_for(make_plan())
