"""Tests for stateless greedy token sampling."""

import pytest
import torch

from micro_vllm.executor.interface import ModelRunnerOutput, SampledToken
from micro_vllm.sampling.params import SamplingParams
from micro_vllm.sampling.sampler import GreedySampler, Sampler


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


def test_sampler_greedy_matches_compatibility_sampler() -> None:
    logits = torch.tensor([[0.1, 2.0, -1.0], [4.0, 2.0, 1.0]])
    request_ids = ("first", "second")

    output = Sampler()(
        logits,
        request_ids,
        (SamplingParams(max_tokens=1), SamplingParams(max_tokens=1)),
        (0, 0),
    )

    assert output == GreedySampler()(logits, request_ids)


def test_temperature_sampling_matches_seeded_torch_reference() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    params = SamplingParams(max_tokens=1, temperature=0.5, seed=3)
    generator = torch.Generator().manual_seed(5)
    expected = torch.multinomial(
        torch.softmax(logits[0] / 0.5, dim=-1),
        1,
        generator=generator,
    ).item()

    output = Sampler()(logits, ("request",), (params,), (2,))

    assert output.sampled_tokens[0].token_id == expected


def test_top_k_never_samples_a_filtered_token() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
    params = SamplingParams(
        max_tokens=32,
        temperature=1.0,
        top_k=2,
        seed=0,
    )

    sampled = {
        Sampler()(logits, ("request",), (params,), (index,))
        .sampled_tokens[0]
        .token_id
        for index in range(32)
    }

    assert sampled <= {3, 4}
    assert sampled


def test_top_k_at_least_vocab_size_does_not_filter() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    unfiltered = SamplingParams(max_tokens=1, temperature=1.0, seed=9)
    large_top_k = SamplingParams(
        max_tokens=1,
        temperature=1.0,
        top_k=3,
        seed=9,
    )

    first = Sampler()(logits, ("request",), (unfiltered,), (0,))
    second = Sampler()(logits, ("request",), (large_top_k,), (0,))

    assert first == second


def test_top_p_keeps_the_boundary_token_and_removes_tail() -> None:
    logits = torch.tensor([[3.0, 2.0, 0.0, -1.0]])
    params = SamplingParams(
        max_tokens=32,
        temperature=1.0,
        top_p=0.8,
        seed=0,
    )

    sampled = {
        Sampler()(logits, ("request",), (params,), (index,))
        .sampled_tokens[0]
        .token_id
        for index in range(32)
    }

    assert sampled <= {0, 1}
    assert sampled


def test_mixed_greedy_and_stochastic_rows_use_independent_policies() -> None:
    logits = torch.tensor([[0.0, 5.0, 1.0], [1.0, 1.0, 1.0]])
    greedy = SamplingParams(max_tokens=1)
    stochastic = SamplingParams(max_tokens=1, temperature=1.0, seed=7)

    output = Sampler()(
        logits,
        ("greedy", "stochastic"),
        (greedy, stochastic),
        (0, 0),
    )

    assert output.sampled_tokens[0] == SampledToken("greedy", 1)
    assert output.sampled_tokens[1].request_id == "stochastic"


def test_seed_and_sample_index_define_result_independent_of_other_rows() -> None:
    logits = torch.zeros((2, 64))
    params = SamplingParams(max_tokens=8, temperature=1.0, seed=11)

    mixed = Sampler()(
        logits,
        ("other", "target"),
        (SamplingParams(max_tokens=1), params),
        (0, 4),
    )
    alone = Sampler()(logits[1:], ("target",), (params,), (4,))
    repeated = Sampler()(logits[1:], ("target",), (params,), (4,))

    assert mixed.sampled_tokens[1] == alone.sampled_tokens[0]
    assert repeated == alone


def test_different_seeds_can_produce_different_sequences() -> None:
    logits = torch.zeros((1, 64))
    first = SamplingParams(max_tokens=8, temperature=1.0, seed=11)
    second = SamplingParams(max_tokens=8, temperature=1.0, seed=29)

    first_tokens = tuple(
        Sampler()(logits, ("request",), (first,), (index,))
        .sampled_tokens[0]
        .token_id
        for index in range(8)
    )
    second_tokens = tuple(
        Sampler()(logits, ("request",), (second,), (index,))
        .sampled_tokens[0]
        .token_id
        for index in range(8)
    )

    assert first_tokens != second_tokens


def test_sampler_rejects_mismatched_row_metadata_lengths() -> None:
    with pytest.raises(ValueError, match="must have equal length"):
        Sampler()(
            torch.empty((2, 4)),
            ("request",),
            (SamplingParams(max_tokens=1),),
            (0,),
        )


@pytest.mark.parametrize("sample_index", [True, -1, 1.5])
def test_sampler_rejects_invalid_sample_index(sample_index: object) -> None:
    params = SamplingParams(max_tokens=1, temperature=1.0, seed=0)

    with pytest.raises(ValueError, match="non-negative integers"):
        Sampler()(
            torch.zeros((1, 4)),
            ("request",),
            (params,),
            (sample_index,),
        )
