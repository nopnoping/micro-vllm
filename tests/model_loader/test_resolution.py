"""Tests for resolving local paths and Hugging Face model identifiers."""

from pathlib import Path

import pytest

import micro_vllm.model_loader.loader as loader_module
from micro_vllm.model_loader.loader import (
    _resolve_hf_repo_id,
    _resolve_local_model_path,
)


def test_resolve_local_model_path_accepts_explicit_path(tmp_path: Path) -> None:
    assert _resolve_local_model_path(tmp_path) == tmp_path.resolve()


def test_resolve_local_model_path_rejects_missing_explicit_path(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match=str(model_path)):
        _resolve_local_model_path(model_path)


def test_nonexistent_string_is_not_a_local_path() -> None:
    assert _resolve_local_model_path("org/model") is None
    assert _resolve_hf_repo_id("org/model") == "org/model"


def test_resolve_hf_repo_id_expands_explicit_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        loader_module,
        "_MODEL_ALIASES",
        {"short-model": "org/model"},
    )

    assert _resolve_hf_repo_id("SHORT-MODEL") == "org/model"


def test_resolve_local_model_path_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        _resolve_local_model_path("   ")
