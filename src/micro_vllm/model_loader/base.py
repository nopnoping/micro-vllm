"""Template for model construction and weight initialization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
import json
from pathlib import Path

import torch
from torch import nn

from micro_vllm.config import LLMConfig
from micro_vllm.layers.attention_backend_registry import (
    create_attention_backend,
)
from micro_vllm.models.registry import get_model_class


class ModelLoader(ABC):
    """Template for constructing a model and initializing its weights."""

    def load_model(self, config: LLMConfig) -> nn.Module:
        """Download, construct, initialize, and finalize one model."""
        model_path = self.download_weights(config)
        model_config = config.model_config
        source_label = str(model_config.model)
        config_path = model_path / "config.json"
        try:
            model_config.set_metadata(_read_model_metadata(config_path))
        except ValueError as error:
            raise ValueError(
                f"invalid model config {config_path}: {error}"
            ) from error
        model_class = get_model_class(model_config, source_label=source_label)

        with torch.device("cpu"):
            backend = create_attention_backend(model_config.attention_backend)
            try:
                model = model_class(model_config, backend)
            except ValueError as error:
                raise ValueError(
                    f"invalid model config from {source_label}: {error}"
                ) from error

        try:
            self.load_weights(model, model_path)
        except RuntimeError as error:
            raise RuntimeError(
                f"failed to load model weights from {source_label}: {error}"
            ) from error
        model.eval()
        return model

    @abstractmethod
    def download_weights(self, config: LLMConfig) -> Path:
        """Return a local directory containing model config and weights."""

    @abstractmethod
    def load_weights(self, model: nn.Module, model_path: Path) -> None:
        """Initialize one constructed model from its local weight files."""


def _read_model_metadata(config_path: Path) -> Mapping[str, object]:
    try:
        with config_path.open(encoding="utf-8") as config_file:
            raw_config = json.load(config_file)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid model config JSON: {config_path}") from error

    if not isinstance(raw_config, dict):
        raise ValueError(
            f"model config must be a JSON object: {config_path}"
        )
    return raw_config
