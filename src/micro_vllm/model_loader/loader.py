"""Resolve model files and orchestrate registered model initialization."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from torch import nn

from micro_vllm.config import ModelConfig
from micro_vllm.layers.attention_backend import AttentionBackend
from micro_vllm.model_loader.checkpoint import iter_safetensors_tensors
from micro_vllm.models.registry import get_model_class


_CHECKPOINT_PATTERNS = (
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
)


def load_model(
    model_name_or_path: str | Path,
    attention_backend: AttentionBackend,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> nn.Module:
    """Load one registered model from a local directory or Hugging Face Hub."""
    local_path = _resolve_local_model_path(model_name_or_path)

    if local_path is not None:
        source_label = str(local_path)
        checkpoint_path = local_path
    else:
        assert isinstance(model_name_or_path, str)
        repo_id = model_name_or_path
        source_label = repo_id if revision is None else f"{repo_id}@{revision}"
        checkpoint_path = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
                allow_patterns=list(_CHECKPOINT_PATTERNS),
            )
        )

    config = _read_model_config(checkpoint_path / "config.json")

    model_class = get_model_class(config, source_label=source_label)
    try:
        with torch.device("cpu"):
            model = model_class(config, attention_backend)
    except ValueError as error:
        raise ValueError(
            f"invalid model config from {source_label}: {error}"
        ) from error

    try:
        model.load_weights(iter_safetensors_tensors(checkpoint_path))
    except RuntimeError as error:
        raise RuntimeError(
            f"failed to load model weights from {source_label}: {error}"
        ) from error
    model.eval()
    return model


def _resolve_local_model_path(
    model_name_or_path: str | Path,
) -> Path | None:
    if isinstance(model_name_or_path, str):
        if not model_name_or_path.strip():
            raise ValueError("model name or path must be non-empty")
        model_path = Path(model_name_or_path).expanduser()
        if not model_path.exists():
            return None
    else:
        model_path = model_name_or_path.expanduser()

    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"model directory does not exist: {model_path}"
        )
    if not model_path.is_dir():
        raise ValueError(f"model path must be a directory: {model_path}")
    return model_path


def _read_model_config(config_path: Path) -> ModelConfig:
    try:
        with config_path.open(encoding="utf-8") as config_file:
            raw_config = json.load(config_file)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid model config JSON: {config_path}") from error

    if not isinstance(raw_config, dict):
        raise ValueError(
            f"model config must be a JSON object: {config_path}"
        )
    try:
        return ModelConfig(raw_config)
    except ValueError as error:
        raise ValueError(
            f"invalid model config {config_path}: {error}"
        ) from error
