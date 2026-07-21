"""Tests for initializing Llama from the generic model config."""

from micro_vllm.config import ModelConfig
from micro_vllm.layers.attention_backend import ReferenceAttentionBackend
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
    return LlamaModel(ModelConfig(values), ReferenceAttentionBackend())


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


def test_llama_uses_simple_defaults() -> None:
    model = make_model()

    assert model.layers[0].self_attn.num_key_value_heads == 4
    assert model.layers[0].self_attn.rotary_embedding.base == 10_000.0
    assert model.norm.variance_epsilon == 1e-6
    assert model.embed_tokens.weight is not model.lm_head.weight
