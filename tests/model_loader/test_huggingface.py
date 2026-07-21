"""Tests for direct, revision-pinned Hugging Face loading."""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import micro_vllm.model_loader.loader as loader_module
from micro_vllm.layers.attention_backend import ReferenceAttentionBackend
from micro_vllm.model_loader import load_model
from micro_vllm.config import ModelConfig


def test_load_model_pins_config_and_weights_to_one_hf_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"architectures": ["FakeModel"]}),
        encoding="utf-8",
    )
    events: list[str] = []
    calls: dict[str, dict[str, object]] = {}

    class FakeApi:
        def model_info(self, repo_id, *, revision, token):
            events.append("info")
            assert repo_id == "org/model"
            assert revision == "branch"
            assert token == "token"
            return SimpleNamespace(sha="commit-sha")

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

    def fake_hf_hub_download(**kwargs):
        events.append("config")
        calls["config"] = kwargs
        return str(config_path)

    def fake_snapshot_download(**kwargs):
        events.append("snapshot")
        calls["snapshot"] = kwargs
        return str(tmp_path)

    monkeypatch.setattr(loader_module, "HfApi", FakeApi)
    monkeypatch.setattr(
        loader_module,
        "hf_hub_download",
        fake_hf_hub_download,
    )
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
    )

    assert isinstance(model, FakeModel)
    assert model.weights[0][0] == "model.weight"
    assert events == [
        "info",
        "config",
        "build",
        "snapshot",
        "checkpoint",
        "load",
        "tensor",
        "eval",
    ]
    for kwargs in calls.values():
        assert kwargs["revision"] == "commit-sha"
        assert kwargs["cache_dir"] == tmp_path / "cache"
        assert kwargs["token"] == "token"
    assert calls["snapshot"]["allow_patterns"] == [
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "model-*.safetensors",
    ]
