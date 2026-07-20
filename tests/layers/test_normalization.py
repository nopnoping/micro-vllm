"""Tests for Llama RMS normalization."""

import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from micro_vllm.layers.normalization import RMSNorm


def test_rms_norm_parameter_shape_and_epsilon() -> None:
    norm = RMSNorm(hidden_size=8, eps=1e-5)

    assert norm.weight.shape == torch.Size((8,))
    assert norm.variance_epsilon == 1e-5
    assert tuple(norm.state_dict()) == ("weight",)


def test_rms_norm_accumulates_variance_in_float32() -> None:
    hidden_states = torch.tensor(
        [[[512.0, -384.0, 256.0, -128.0]]],
        dtype=torch.float16,
    )
    norm = RMSNorm(hidden_size=4).to(dtype=torch.float16)

    output = norm(hidden_states)

    float_states = hidden_states.to(torch.float32)
    expected = float_states * torch.rsqrt(
        float_states.pow(2).mean(dim=-1, keepdim=True)
        + norm.variance_epsilon
    )
    assert torch.isfinite(output).all()
    assert torch.equal(output, expected.to(torch.float16))


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
)
def test_rms_norm_preserves_model_dtype(dtype: torch.dtype) -> None:
    norm = RMSNorm(hidden_size=6).to(dtype=dtype)
    hidden_states = torch.linspace(-1.0, 1.0, steps=24).reshape(2, 2, 6).to(dtype)

    output = norm(hidden_states)

    assert output.shape == hidden_states.shape
    assert output.dtype is dtype


def test_rms_norm_matches_transformers_llama_rms_norm() -> None:
    hidden_size = 8
    eps = 1e-5
    norm = RMSNorm(hidden_size, eps=eps)
    reference = LlamaRMSNorm(hidden_size, eps=eps)
    weight = torch.linspace(0.5, 1.5, steps=hidden_size)
    with torch.no_grad():
        norm.weight.copy_(weight)
        reference.weight.copy_(weight)
    hidden_states = torch.linspace(
        -2.0,
        2.0,
        steps=2 * 3 * hidden_size,
    ).reshape(2, 3, hidden_size)

    output = norm(hidden_states)
    expected = reference(hidden_states)

    torch.testing.assert_close(output, expected, atol=0, rtol=0)
