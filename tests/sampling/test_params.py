"""Tests for per-request sampling parameters."""

from dataclasses import FrozenInstanceError

import pytest

from micro_vllm.sampling.params import SamplingParams


def test_sampling_params_defaults() -> None:
    params = SamplingParams(max_tokens=4)

    assert params.max_tokens == 4
    assert params.eos_token_id is None
    assert params.ignore_eos is False
    assert params.temperature == 0.0
    assert params.top_k is None
    assert params.top_p == 1.0
    assert params.seed is None
    assert params.is_greedy is True


def test_sampling_params_stores_eos_configuration() -> None:
    params = SamplingParams(max_tokens=4, eos_token_id=2, ignore_eos=True)

    assert params.eos_token_id == 2
    assert params.ignore_eos is True


def test_sampling_params_stores_stochastic_configuration() -> None:
    params = SamplingParams(
        max_tokens=8,
        temperature=0.7,
        top_k=20,
        top_p=0.8,
        seed=42,
    )

    assert params.temperature == 0.7
    assert params.top_k == 20
    assert params.top_p == 0.8
    assert params.seed == 42
    assert params.is_greedy is False


def test_sampling_params_normalizes_numeric_values_to_float() -> None:
    params = SamplingParams(max_tokens=4, temperature=1, top_p=1)

    assert params.temperature == 1.0
    assert type(params.temperature) is float
    assert params.top_p == 1.0
    assert type(params.top_p) is float


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "4"])
def test_sampling_params_rejects_invalid_max_tokens(value: object) -> None:
    with pytest.raises(ValueError, match=r"^max_tokens must be greater than zero$"):
        SamplingParams(max_tokens=value)


@pytest.mark.parametrize("value", [True, -1, 1.5, "2"])
def test_sampling_params_rejects_invalid_eos_token_id(value: object) -> None:
    with pytest.raises(ValueError, match=r"^eos_token_id must be non-negative$"):
        SamplingParams(max_tokens=4, eos_token_id=value)


def test_sampling_params_rejects_non_boolean_ignore_eos() -> None:
    with pytest.raises(ValueError, match=r"^ignore_eos must be a boolean$"):
        SamplingParams(max_tokens=4, ignore_eos=1)


@pytest.mark.parametrize(
    "value, message",
    [
        (True, "temperature must be a finite number"),
        ("0.7", "temperature must be a finite number"),
        (float("nan"), "temperature must be a finite number"),
        (float("inf"), "temperature must be a finite number"),
        (-0.1, "temperature must be non-negative"),
    ],
)
def test_sampling_params_rejects_invalid_temperature(
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=rf"^{message}$"):
        SamplingParams(max_tokens=4, temperature=value)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "20"])
def test_sampling_params_rejects_invalid_top_k(value: object) -> None:
    with pytest.raises(
        ValueError,
        match=r"^top_k must be a positive integer or None$",
    ):
        SamplingParams(max_tokens=4, temperature=1.0, top_k=value)


@pytest.mark.parametrize(
    "value, message",
    [
        (True, "top_p must be a finite number"),
        ("0.8", "top_p must be a finite number"),
        (float("nan"), "top_p must be a finite number"),
        (float("inf"), "top_p must be a finite number"),
        (0.0, "top_p must be greater than zero and at most one"),
        (-0.1, "top_p must be greater than zero and at most one"),
        (1.1, "top_p must be greater than zero and at most one"),
    ],
)
def test_sampling_params_rejects_invalid_top_p(
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=rf"^{message}$"):
        SamplingParams(max_tokens=4, temperature=1.0, top_p=value)


@pytest.mark.parametrize("value", [True, -1, 1.5, "42"])
def test_sampling_params_rejects_invalid_seed(value: object) -> None:
    with pytest.raises(
        ValueError,
        match=r"^seed must be a non-negative integer or None$",
    ):
        SamplingParams(max_tokens=4, temperature=1.0, seed=value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"top_k": 20},
        {"top_p": 0.8},
        {"top_k": 20, "top_p": 0.8},
    ],
)
def test_greedy_sampling_rejects_stochastic_filters(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(
        ValueError,
        match=r"^top_k and top_p require a positive temperature$",
    ):
        SamplingParams(max_tokens=4, **overrides)


def test_sampling_params_is_frozen() -> None:
    params = SamplingParams(max_tokens=4)

    with pytest.raises(FrozenInstanceError):
        params.max_tokens = 8
