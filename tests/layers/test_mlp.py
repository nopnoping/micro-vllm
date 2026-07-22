"""Tests for the reusable dense SwiGLU MLP."""

import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP as HFLlamaMLP

from micro_vllm.layers.mlp import SwiGLUMLP


def test_swiglu_mlp_keeps_checkpoint_parameter_names() -> None:
    mlp = SwiGLUMLP(hidden_size=8, intermediate_size=16)

    assert tuple(mlp.state_dict()) == (
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    )


def test_swiglu_mlp_matches_transformers_llama_mlp() -> None:
    config = LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_act="silu",
        mlp_bias=False,
    )
    mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)
    reference = HFLlamaMLP(config)
    reference.load_state_dict(mlp.state_dict())
    hidden_states = torch.linspace(-1.0, 1.0, steps=2 * 3 * 8).reshape(2, 3, 8)

    output = mlp(hidden_states)
    expected = reference(hidden_states)

    torch.testing.assert_close(output, expected, atol=0, rtol=0)
