"""Tests for initializing Llama from the generic model config."""

import pytest

from micro_vllm.attention.spec import FullAttentionSpec
from micro_vllm.config import ModelConfig
from micro_vllm.models.llama import LlamaModel


def make_model(**overrides: object) -> LlamaModel:
    values = {
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
    }
    values.update(overrides)
    return LlamaModel(ModelConfig(model="unused", metadata=values))


def test_llama_initializes_from_generic_config() -> None:
    model = make_model(
        vocab_size=320,
        hidden_size=96,
        intermediate_size=192,
        num_hidden_layers=3,
        num_attention_heads=6,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        rope_theta=500_000.0,
        tie_word_embeddings=True,
    )

    assert model.embed_tokens.weight.shape == (320, 96)
    assert len(model.layers) == 3
    assert model.layers[0].self_attn.num_attention_heads == 6
    assert model.layers[0].self_attn.num_key_value_heads == 2
    assert model.layers[0].self_attn.rotary_embedding.base == 500_000.0
    assert model.norm.variance_epsilon == 1e-5
    assert model.embed_tokens.weight is model.lm_head.weight
    assert model.attention_groups == (model.layers[0].self_attn.attention,)
    assert model.attention_groups[0].spec == FullAttentionSpec()
    assert model.layer_attention_group_ids == (0, 0, 0)
    assert all(
        layer.self_attn.attention is model.attention_groups[0]
        for layer in model.layers
    )


def test_llama_uses_simple_defaults() -> None:
    model = make_model()

    assert model.layers[0].self_attn.num_key_value_heads == 4
    assert model.layers[0].self_attn.rotary_embedding.base == 10_000.0
    assert model.norm.variance_epsilon == 1e-6
    assert model.embed_tokens.weight is not model.lm_head.weight


def test_llama_rejects_incompatible_attention_head_counts() -> None:
    with pytest.raises(
        ValueError,
        match="num_attention_heads must be divisible by num_key_value_heads",
    ):
        make_model(num_attention_heads=3, num_key_value_heads=2)
