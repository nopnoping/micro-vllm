"""Tests for static model architecture lookup."""

import pytest

from micro_vllm.config import ModelConfig
from micro_vllm.models.llama import LlamaModel
from micro_vllm.models.registry import get_model_class


def test_registry_uses_first_supported_requested_architecture() -> None:
    config = ModelConfig(
        {"architectures": ["UnknownModel", "LlamaForCausalLM"]}
    )

    assert get_model_class(config, source_label="test-model") is LlamaModel


def test_registry_reports_requested_supported_and_source() -> None:
    config = ModelConfig({"architectures": ["Unknown", "Other"]})

    with pytest.raises(ValueError) as error_info:
        get_model_class(config, source_label="test-model")

    message = str(error_info.value)
    assert "['Unknown', 'Other']" in message
    assert "['LlamaForCausalLM']" in message
    assert "test-model" in message
