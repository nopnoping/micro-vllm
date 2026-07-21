"""Tests for model loading orchestration over a local model source."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from micro_vllm.layers.attention_backend import ReferenceAttentionBackend
from micro_vllm.model_loader import load_model
from micro_vllm.config import ModelConfig
from micro_vllm.models.llama import LlamaModel


def write_local_model(tmp_path: Path) -> ModelConfig:
    raw_config = {
        "architectures": ["UnknownModel", "LlamaForCausalLM"],
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
    }
    (tmp_path / "config.json").write_text(
        json.dumps(raw_config),
        encoding="utf-8",
    )
    config = ModelConfig(raw_config)
    reference = LlamaModel(config, ReferenceAttentionBackend())
    tensors = {
        (
            parameter_name
            if parameter_name == "lm_head.weight"
            else f"model.{parameter_name}"
        ): torch.full(parameter.shape, float(index), dtype=torch.float16)
        for index, (parameter_name, parameter) in enumerate(
            reference.named_parameters(),
            start=1,
        )
    }
    save_file(tensors, tmp_path / "model.safetensors")
    return config


def test_load_model_resolves_constructs_and_populates_local_llama(
    tmp_path: Path,
) -> None:
    config = write_local_model(tmp_path)
    backend = ReferenceAttentionBackend()

    model = load_model(tmp_path, backend)

    assert isinstance(model, LlamaModel)
    assert model.config == config
    assert model.training is False
    assert model.layers[0].self_attn.backend is backend
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_load_model_returns_independent_instances(tmp_path: Path) -> None:
    write_local_model(tmp_path)

    first = load_model(tmp_path, ReferenceAttentionBackend())
    second = load_model(tmp_path, ReferenceAttentionBackend())

    assert isinstance(first, LlamaModel)
    assert isinstance(second, LlamaModel)
    assert first is not second
    assert first.embed_tokens.weight is not second.embed_tokens.weight


def test_load_model_adds_source_context_to_model_config_error(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "vocab_size": 32,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error_info:
        load_model(tmp_path, ReferenceAttentionBackend())

    message = str(error_info.value)
    assert "missing required model config field 'hidden_size'" in message
    assert str(tmp_path) in message
