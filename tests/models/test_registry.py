"""Tests for static model architecture lookup."""

import pytest

from micro_vllm.config import ModelConfig
from micro_vllm.models.llama import LlamaModel
from micro_vllm.models.mistral import MistralModel
from micro_vllm.models.qwen import Qwen3Model
from micro_vllm.models.registry import get_model_class


def test_registry_uses_first_supported_requested_architecture() -> None:
    config = ModelConfig(
        model="unused",
        metadata={"architectures": ["UnknownModel", "LlamaForCausalLM"]},
    )

    assert get_model_class(config, source_label="test-model") is LlamaModel


def test_registry_resolves_qwen3_architecture() -> None:
    config = ModelConfig(
        model="unused",
        metadata={"architectures": ["Qwen3ForCausalLM"]},
    )

    assert get_model_class(config, source_label="test-model") is Qwen3Model


def test_registry_resolves_mistral_architecture() -> None:
    config = ModelConfig(
        model="unused",
        metadata={"architectures": ["MistralForCausalLM"]},
    )

    assert get_model_class(config, source_label="test-model") is MistralModel


def test_registry_reports_requested_supported_and_source() -> None:
    config = ModelConfig(
        model="unused",
        metadata={"architectures": ["Unknown", "Other"]},
    )

    with pytest.raises(ValueError) as error_info:
        get_model_class(config, source_label="test-model")

    message = str(error_info.value)
    assert "['Unknown', 'Other']" in message
    assert (
        "['LlamaForCausalLM', 'MistralForCausalLM', 'Qwen3ForCausalLM']"
        in message
    )
    assert "test-model" in message
