"""Tests for per-request sampling parameters."""

from dataclasses import FrozenInstanceError

import pytest

from micro_vllm.sampling.params import SamplingParams


def test_sampling_params_defaults() -> None:
    params = SamplingParams(max_tokens=4)

    assert params.max_tokens == 4
    assert params.eos_token_id is None
    assert params.ignore_eos is False


def test_sampling_params_stores_eos_configuration() -> None:
    params = SamplingParams(max_tokens=4, eos_token_id=2, ignore_eos=True)

    assert params.eos_token_id == 2
    assert params.ignore_eos is True


@pytest.mark.parametrize("value", [0, -1])
def test_sampling_params_rejects_non_positive_max_tokens(value: int) -> None:
    with pytest.raises(ValueError, match=r"^max_tokens must be greater than zero$"):
        SamplingParams(max_tokens=value)
def test_sampling_params_rejects_negative_eos_token_id() -> None:
    with pytest.raises(ValueError, match=r"^eos_token_id must be non-negative$"):
        SamplingParams(max_tokens=4, eos_token_id=-1)
def test_sampling_params_is_frozen() -> None:
    params = SamplingParams(max_tokens=4)

    with pytest.raises(FrozenInstanceError):
        params.max_tokens = 8
