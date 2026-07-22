"""Tests for Hugging Face snapshot loading."""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest
import torch

import micro_vllm.model_loader.loader as loader_module
from micro_vllm.layers.attention_backend import ReferenceAttentionBackend
from micro_vllm.model_loader import load_model
from micro_vllm.config import ModelConfig


def test_load_model_delegates_snapshot_resolution_to_hugging_face(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"architectures": ["FakeModel"]}),
        encoding="utf-8",
    )
    events: list[str] = []
    snapshot_call: dict[str, object] = {}

    class FakeCheckpoint:
        def __init__(self, model_path: str | Path) -> None:
            events.append("checkpoint")
            assert Path(model_path) == tmp_path

        def iter_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
            events.append("tensor")
            yield "model.weight", torch.tensor([1.0])

    class FakeModel:
        def __init__(
            self,
            config: ModelConfig,
            attention_backend: ReferenceAttentionBackend,
        ) -> None:
            events.append("build")
            assert config.architectures == ("FakeModel",)
            assert isinstance(attention_backend, ReferenceAttentionBackend)
            self.weights: list[tuple[str, torch.Tensor]] = []

        def load_weights(
            self,
            weights: Iterable[tuple[str, torch.Tensor]],
        ) -> None:
            events.append("load")
            self.weights = list(weights)

        def eval(self) -> None:
            events.append("eval")

    def fake_snapshot_download(**kwargs):
        events.append("snapshot")
        snapshot_call.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(
        loader_module,
        "snapshot_download",
        fake_snapshot_download,
    )
    monkeypatch.setattr(
        loader_module,
        "SafetensorsCheckpoint",
        FakeCheckpoint,
    )
    monkeypatch.setattr(
        loader_module,
        "get_model_class",
        lambda config, *, source_label: FakeModel,
    )

    model = load_model(
        "org/model",
        ReferenceAttentionBackend(),
        revision="branch",
        cache_dir=tmp_path / "cache",
        token="token",
        local_files_only=True,
    )

    assert isinstance(model, FakeModel)
    assert model.weights[0][0] == "model.weight"
    assert events == [
        "snapshot",
        "build",
        "checkpoint",
        "load",
        "tensor",
        "eval",
    ]
    assert snapshot_call["repo_id"] == "org/model"
    assert snapshot_call["revision"] == "branch"
    assert snapshot_call["cache_dir"] == tmp_path / "cache"
    assert snapshot_call["token"] == "token"
    assert snapshot_call["local_files_only"] is True
    assert snapshot_call["allow_patterns"] == [
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "model-*.safetensors",
    ]
