"""Tests for resolving local paths and Hugging Face model identifiers."""

from pathlib import Path

import pytest

from micro_vllm.model_loader.default import _resolve_local_model_path


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


def test_resolve_local_model_path_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        _resolve_local_model_path("   ")
