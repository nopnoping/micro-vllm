"""Model architecture registration and resolution."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_CONFIG_FILE_NAME = "config.json"


@dataclass(frozen=True, slots=True)
class _HFConfig:
    """Validated metadata read from one local Hugging Face config."""

    path: Path
    raw: dict[str, Any]
    architectures: tuple[str, ...]


def _read_hf_config(model_path: str | Path) -> _HFConfig:
    """Read and validate metadata needed before architecture resolution."""
    normalized_path = Path(model_path).expanduser().resolve()
    if not normalized_path.exists():
        raise FileNotFoundError(
            f"model directory does not exist: {normalized_path}"
        )
    if not normalized_path.is_dir():
        raise ValueError(f"model path must be a directory: {normalized_path}")

    config_path = normalized_path / _CONFIG_FILE_NAME
    if not config_path.exists():
        raise FileNotFoundError(f"model config does not exist: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"model config path is not a file: {config_path}")

    try:
        with config_path.open(encoding="utf-8") as config_file:
            raw_config = json.load(
                config_file,
                object_pairs_hook=_object_without_duplicates,
            )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid model config JSON: {config_path}") from error
    except ValueError as error:
        raise ValueError(f"invalid model config {config_path}: {error}") from error

    if not isinstance(raw_config, dict):
        raise ValueError(f"model config must be a JSON object: {config_path}")

    architectures = raw_config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise ValueError(
            "model config architectures must be a non-empty list: "
            f"{config_path}"
        )
    for index, architecture in enumerate(architectures):
        if not isinstance(architecture, str) or not architecture.strip():
            raise ValueError(
                "model config architecture at index "
                f"{index} must be a non-empty string: {config_path}"
            )

    return _HFConfig(
        path=config_path,
        raw=raw_config,
        architectures=tuple(architectures),
    )


def _select_supported_architecture(
    config: _HFConfig,
    supported_architectures: Sequence[str],
) -> str:
    """Select the first requested architecture supported by the registry."""
    supported = set(supported_architectures)
    for architecture in config.architectures:
        if architecture in supported:
            return architecture

    raise ValueError(
        f"unsupported model architectures {list(config.architectures)!r}; "
        f"supported architectures: {list(supported_architectures)!r}; "
        f"config: {config.path}"
    )


def _object_without_duplicates(
    pairs: Iterable[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result
