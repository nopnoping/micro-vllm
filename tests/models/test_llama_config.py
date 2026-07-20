"""Tests for the minimal dense Llama architecture contract."""

from dataclasses import FrozenInstanceError

import pytest

from micro_vllm.models.llama import LlamaModelConfig


def make_config(**overrides: object) -> LlamaModelConfig:
    values = {
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
    }
    values.update(overrides)
    return LlamaModelConfig(**values)


def test_llama_config_stores_minimal_architecture() -> None:
    config = make_config(tie_word_embeddings=True)

    assert config.vocab_size == 256
    assert config.hidden_size == 64
    assert config.intermediate_size == 128
    assert config.num_hidden_layers == 2
    assert config.num_attention_heads == 4
    assert config.num_key_value_heads == 2
    assert config.max_position_embeddings == 128
    assert config.rms_norm_eps == 1e-6
    assert config.rope_theta == 10_000.0
    assert config.hidden_act == "silu"
    assert config.tie_word_embeddings is True
    assert config.head_dim == 16
    assert config.num_query_heads_per_kv_head == 2


def test_llama_config_is_frozen() -> None:
    config = make_config()

    with pytest.raises(FrozenInstanceError):
        config.hidden_size = 128


@pytest.mark.parametrize(
    "field_name",
    [
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_llama_config_rejects_non_positive_dimensions(
    field_name: str,
    value: int,
) -> None:
    with pytest.raises(ValueError, match=rf"^{field_name} must be greater than zero$"):
        make_config(**{field_name: value})


def test_llama_config_requires_integral_attention_head_dim() -> None:
    with pytest.raises(
        ValueError,
        match="hidden_size must be divisible by num_attention_heads",
    ):
        make_config(hidden_size=66)


def test_llama_config_requires_integral_gqa_group_ratio() -> None:
    with pytest.raises(
        ValueError,
        match="num_attention_heads must be divisible by num_key_value_heads",
    ):
        make_config(num_key_value_heads=3)


def test_llama_config_requires_even_head_dim_for_rope() -> None:
    with pytest.raises(
        ValueError,
        match="attention head_dim must be even for rotary embedding",
    ):
        make_config(hidden_size=60)


@pytest.mark.parametrize("field_name", ["rms_norm_eps", "rope_theta"])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_llama_config_rejects_invalid_float_parameters(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=rf"^{field_name} must be greater than zero"):
        make_config(**{field_name: value})


def test_llama_config_rejects_unsupported_activation() -> None:
    with pytest.raises(
        ValueError,
        match="hidden_act must be 'silu' for the V1 Llama model",
    ):
        make_config(hidden_act="gelu")
