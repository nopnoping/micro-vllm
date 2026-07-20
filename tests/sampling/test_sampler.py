"""Tests for stateless greedy token sampling."""

import pytest
import torch

from micro_vllm.executor.interface import ModelRunnerOutput, SampledToken
from micro_vllm.sampling.sampler import GreedySampler


def test_single_row_selects_largest_logit() -> None:
    logits = torch.tensor([[0.1, 2.0, -1.0]])

    output = GreedySampler()(logits, ("request-1",))

    assert output == ModelRunnerOutput((SampledToken("request-1", 1),))


def test_multiple_rows_preserve_request_id_order() -> None:
    logits = torch.tensor(
        [
            [0.0, 1.0, 3.0],
            [4.0, 2.0, 1.0],
            [0.0, 5.0, 1.0],
        ]
    )

    output = GreedySampler()(logits, ("third", "first", "second"))

    assert output.sampled_tokens == (
        SampledToken("third", 2),
        SampledToken("first", 0),
        SampledToken("second", 1),
    )


def test_empty_sampling_batch_returns_empty_output() -> None:
    output = GreedySampler()(torch.empty((0, 4)), ())

    assert output == ModelRunnerOutput()


def test_equal_maximum_logits_follow_torch_argmax_tie_behavior() -> None:
    logits = torch.tensor([[1.0, 3.0, 3.0, 2.0]])

    output = GreedySampler()(logits, ("request-1",))

    assert output.sampled_tokens[0].token_id == logits.argmax(dim=-1).item()
    assert output.sampled_tokens[0].token_id == 1


def test_logits_rows_must_match_sampling_request_ids() -> None:
    with pytest.raises(
        ValueError,
        match="logits rows must equal sampling request IDs",
    ):
        GreedySampler()(torch.empty((2, 4)), ("request-1",))


@pytest.mark.gpu
def test_cpu_and_gpu_choose_identical_tokens() -> None:
    logits = torch.tensor(
        [
            [0.0, 2.0, 1.0],
            [3.0, 1.0, 2.0],
        ]
    )
    request_ids = ("first", "second")

    cpu_output = GreedySampler()(logits, request_ids)
    gpu_output = GreedySampler()(logits.cuda(), request_ids)

    assert gpu_output == cpu_output
