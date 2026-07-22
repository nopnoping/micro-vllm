"""Tests for model loading orchestration over a local model source."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import micro_vllm.model_loader.base as base_module
from micro_vllm.config import LLMConfig, ModelConfig
from micro_vllm.layers.attention_backend import ReferenceAttentionBackend
from micro_vllm.model_loader import load_model
from micro_vllm.model_loader.base import ModelLoader
from micro_vllm.models.llama import LlamaModel


def write_local_model(
    tmp_path: Path,
    *,
    invalid_embedding_shape: bool = False,
) -> ModelConfig:
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
    if invalid_embedding_shape:
        tensors["model.embed_tokens.weight"] = torch.zeros(
            (2, 3),
            dtype=torch.float16,
        )
    save_file(tensors, tmp_path / "model.safetensors")
    return config


def test_model_loader_template_controls_loading_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["FakeModel"]}),
        encoding="utf-8",
    )
    events: list[str] = []
    backend = ReferenceAttentionBackend()

    def create_backend(name: str) -> ReferenceAttentionBackend:
        events.append("backend")
        assert name == "recording"
        return backend

    class FakeModel(torch.nn.Module):
        def __init__(
            self,
            config: ModelConfig,
            attention_backend: ReferenceAttentionBackend,
        ) -> None:
            super().__init__()
            events.append("construct")
            assert config.architectures == ("FakeModel",)
            assert attention_backend is backend

        def eval(self) -> torch.nn.Module:
            events.append("eval")
            return super().eval()

    class RecordingLoader(ModelLoader):
        def download_weights(self, config: LLMConfig) -> Path:
            events.append("download")
            assert config.model == "org/model"
            return tmp_path

        def load_weights(self, model: torch.nn.Module, model_path: Path) -> None:
            events.append("load_weights")
            assert isinstance(model, FakeModel)
            assert model_path == tmp_path

    monkeypatch.setattr(
        base_module,
        "get_model_class",
        lambda config, *, source_label: FakeModel,
    )
    monkeypatch.setattr(
        base_module,
        "create_attention_backend",
        create_backend,
    )

    model = RecordingLoader().load_model(
        LLMConfig(model="org/model", attention_backend="recording")
    )

    assert isinstance(model, FakeModel)
    assert events == [
        "download",
        "backend",
        "construct",
        "load_weights",
        "eval",
    ]


def test_load_model_resolves_constructs_and_populates_local_llama(
    tmp_path: Path,
) -> None:
    config = write_local_model(tmp_path)
    model = load_model(LLMConfig(model=tmp_path))

    assert isinstance(model, LlamaModel)
    assert model.config == config
    assert model.training is False
    assert isinstance(
        model.layers[0].self_attn.backend,
        ReferenceAttentionBackend,
    )
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_load_model_returns_independent_instances(tmp_path: Path) -> None:
    write_local_model(tmp_path)

    config = LLMConfig(model=tmp_path)
    first = load_model(config)
    second = load_model(config)

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
        load_model(LLMConfig(model=tmp_path))

    message = str(error_info.value)
    assert "missing required model config field 'hidden_size'" in message
    assert str(tmp_path) in message


def test_load_model_preserves_shape_error_and_adds_source_context(
    tmp_path: Path,
) -> None:
    write_local_model(tmp_path, invalid_embedding_shape=True)

    with pytest.raises(RuntimeError) as error_info:
        load_model(LLMConfig(model=tmp_path))

    message = str(error_info.value)
    assert "size of tensor" in message
    assert str(tmp_path) in message
