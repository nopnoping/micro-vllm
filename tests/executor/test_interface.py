"""Tests for the executor input and output contracts."""

from dataclasses import FrozenInstanceError

import pytest

from micro_vllm.executor.interface import ModelRunnerOutput, SampledToken


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
