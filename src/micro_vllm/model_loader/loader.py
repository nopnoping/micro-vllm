"""Resolve model files and orchestrate registered model initialization."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from torch import nn

from micro_vllm.layers.attention_backend import AttentionBackend
from micro_vllm.model_loader.checkpoint import SafetensorsCheckpoint
from micro_vllm.config import ModelConfig
from micro_vllm.models.registry import get_model_class


_CHECKPOINT_PATTERNS = (
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
)
_MODEL_ALIASES: Mapping[str, str] = MappingProxyType({})


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
    repo_id: str | None = None
    commit_hash: str | None = None

    if local_path is not None:
        source_label = str(local_path)
        config = _read_model_config(local_path / "config.json")
    else:
        assert isinstance(model_name_or_path, str)
        repo_id = _resolve_hf_repo_id(model_name_or_path)
        commit_hash = _resolve_hf_commit(repo_id, revision, token)
        source_label = f"{repo_id}@{commit_hash}"
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            revision=commit_hash,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        config = _read_model_config(config_path)

    model_class = get_model_class(config, source_label=source_label)
    try:
        with torch.device("cpu"):
            model = model_class(config, attention_backend)
    except ValueError as error:
        raise ValueError(
            f"invalid model config from {source_label}: {error}"
        ) from error

    if local_path is not None:
        checkpoint_path = local_path
    else:
        assert repo_id is not None
        assert commit_hash is not None
        checkpoint_path = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=commit_hash,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
                allow_patterns=list(_CHECKPOINT_PATTERNS),
            )
        )

    try:
        model.load_weights(SafetensorsCheckpoint(checkpoint_path).iter_tensors())
    except RuntimeError as error:
        raise RuntimeError(
            f"failed to load model weights from {source_label}: {error}"
        ) from error
    model.eval()
    return model


def _resolve_local_model_path(
    model_name_or_path: str | Path,
) -> Path | None:
    if isinstance(model_name_or_path, Path):
        return _validate_local_model_path(model_name_or_path)
    if not model_name_or_path.strip():
        raise ValueError("model name or path must be non-empty")

    local_candidate = Path(model_name_or_path).expanduser()
    if local_candidate.exists():
        return _validate_local_model_path(local_candidate)
    return None


def _validate_local_model_path(model_path: Path) -> Path:
    normalized_path = model_path.expanduser().resolve()
    if not normalized_path.exists():
        raise FileNotFoundError(
            f"model directory does not exist: {normalized_path}"
        )
    if not normalized_path.is_dir():
        raise ValueError(f"model path must be a directory: {normalized_path}")
    return normalized_path


def _resolve_hf_repo_id(model_name: str) -> str:
    return _MODEL_ALIASES.get(model_name.casefold(), model_name)


def _resolve_hf_commit(
    repo_id: str,
    revision: str | None,
    token: str | bool | None,
) -> str:
    if revision is not None and re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    model_info = HfApi().model_info(
        repo_id,
        revision=revision,
        token=token,
    )
    if not model_info.sha:
        raise ValueError(f"Hugging Face model has no resolved commit: {repo_id}")
    return model_info.sha


def _read_model_config(config_path: str | Path) -> ModelConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"model config does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"model config path is not a file: {path}")

    try:
        with path.open(encoding="utf-8") as config_file:
            raw_config = json.load(config_file)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid model config JSON: {path}") from error

    if not isinstance(raw_config, dict):
        raise ValueError(f"model config must be a JSON object: {path}")
    try:
        return ModelConfig(raw_config)
    except ValueError as error:
        raise ValueError(f"invalid model config {path}: {error}") from error
