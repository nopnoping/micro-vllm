"""Tests for model configuration and read-only loaded metadata."""

import pytest

from micro_vllm.config import ModelConfig


def test_model_config_accepts_metadata_after_source_configuration() -> None:
    config = ModelConfig(model="org/model", revision="main")

    assert config.metadata is None
    assert config.architectures == ()

    config.set_metadata(
        {
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 64,
        }
    )

    assert config.model == "org/model"
    assert config.revision == "main"
    assert config.architectures == ("LlamaForCausalLM",)
    assert config.require_positive_int("hidden_size") == 64


def test_model_config_preserves_generic_metadata() -> None:
    config = ModelConfig(
        model="unused",
        metadata={
            "architectures": ["Unknown", "LlamaForCausalLM"],
            "hidden_size": 64,
            "custom_metadata": {"name": "value"},
        },
    )

    assert config.architectures == ("Unknown", "LlamaForCausalLM")
    assert config.require_positive_int("hidden_size") == 64
    assert config.values.get("custom_metadata") == {"name": "value"}


def test_model_config_copies_and_exposes_read_only_values() -> None:
    values = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 64,
    }
    config = ModelConfig(model="unused", metadata=values)
    values["hidden_size"] = 128

    assert config.require_positive_int("hidden_size") == 64
    with pytest.raises(TypeError):
        config.values["hidden_size"] = 128


@pytest.mark.parametrize(
    "architectures",
    [None, [], "LlamaForCausalLM", [""], ["   "], [True]],
)
def test_model_config_rejects_invalid_architectures(
    architectures: object,
) -> None:
    with pytest.raises(ValueError, match="architectures?|architecture"):
        ModelConfig(model="unused", metadata={"architectures": architectures})


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "1"])
def test_model_config_requires_strict_positive_integers(value: object) -> None:
    config = ModelConfig(
        model="unused",
        metadata={
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": value,
        },
    )

    with pytest.raises(ValueError, match="hidden_size must be a positive integer"):
        config.require_positive_int("hidden_size")


def test_model_config_typed_readers_apply_defaults() -> None:
    config = ModelConfig(
        model="unused",
        metadata={"architectures": ["LlamaForCausalLM"]},
    )

    assert config.positive_int("num_key_value_heads", 4) == 4
    assert config.values.get("rope_theta", 10_000.0) == 10_000.0
