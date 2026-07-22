"""Tests for initializing Qwen3 from the generic model config."""

import pytest

from micro_vllm.config import ModelConfig
from micro_vllm.models.qwen import Qwen3Model


def make_model(**overrides: object) -> Qwen3Model:
    values = {
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 24,
    }
    values.update(overrides)
    return Qwen3Model(ModelConfig(model="unused", metadata=values))


def test_qwen3_initializes_explicit_head_dimension_and_qk_norm() -> None:
    model = make_model(
        vocab_size=320,
        hidden_size=96,
        intermediate_size=192,
        num_hidden_layers=3,
        num_attention_heads=6,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-5,
        rope_theta=500_000.0,
        tie_word_embeddings=True,
    )

    attention = model.layers[0].self_attn
    assert model.embed_tokens.weight.shape == (320, 96)
    assert len(model.layers) == 3
    assert attention.q_proj.weight.shape == (192, 96)
    assert attention.k_proj.weight.shape == (64, 96)
    assert attention.v_proj.weight.shape == (64, 96)
    assert attention.o_proj.weight.shape == (96, 192)
    assert attention.q_proj.bias is None
    assert attention.k_proj.bias is None
    assert attention.v_proj.bias is None
    assert attention.o_proj.bias is None
    assert attention.q_norm.weight.shape == (32,)
    assert attention.k_norm.weight.shape == (32,)
    assert attention.q_norm.variance_epsilon == 1e-5
    assert attention.rotary_embedding.base == 500_000.0
    assert model.embed_tokens.weight is model.lm_head.weight
    assert model.attention_groups == (attention,)
    assert model.layer_attention_group_ids == (0, 0, 0)
    assert all(
        layer.self_attn.attention is model.attention_groups[0]
        for layer in model.layers
    )


def test_qwen3_uses_simple_defaults() -> None:
    model = make_model(head_dim=16, num_key_value_heads=4)
    attention = model.layers[0].self_attn

    assert attention.head_dim == 16
    assert attention.num_key_value_heads == 4
    assert attention.rotary_embedding.base == 1_000_000.0
    assert model.norm.variance_epsilon == 1e-6
    assert model.embed_tokens.weight is not model.lm_head.weight


def test_qwen3_can_construct_projection_bias_when_config_requests_it() -> None:
    model = make_model(attention_bias=True)
    attention = model.layers[0].self_attn

    assert attention.q_proj.bias is not None
    assert attention.k_proj.bias is not None
    assert attention.v_proj.bias is not None
    assert attention.o_proj.bias is not None


def test_qwen3_rejects_incompatible_attention_head_counts() -> None:
    with pytest.raises(
        ValueError,
        match="num_attention_heads must be divisible by num_key_value_heads",
    ):
        make_model(num_attention_heads=3, num_key_value_heads=2)
