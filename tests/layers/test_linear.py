"""Tests for the single-device dense linear layer."""

import pytest
import torch
from torch import nn

from micro_vllm.layers.linear import Linear


@pytest.mark.parametrize("bias", [False, True])
def test_linear_matches_torch_dense_linear(bias: bool) -> None:
    linear = Linear(4, 3, bias=bias)
    reference = nn.Linear(4, 3, bias=bias)
    with torch.no_grad():
        reference.weight.copy_(linear.weight)
        if bias:
            assert linear.bias is not None
            assert reference.bias is not None
            reference.bias.copy_(linear.bias)
    hidden_states = torch.linspace(-1.0, 1.0, steps=2 * 4).reshape(2, 4)

    output = linear(hidden_states)

    assert linear.weight.shape == torch.Size((3, 4))
    assert (linear.bias is not None) is bias
    torch.testing.assert_close(output, reference(hidden_states), atol=0, rtol=0)
