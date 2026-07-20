"""Discovery for local safetensors checkpoints."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import SafetensorError, safe_open


_SINGLE_FILE_NAME = "model.safetensors"
_INDEX_FILE_NAME = "model.safetensors.index.json"


@dataclass(frozen=True, slots=True)
class _CheckpointShard:
    path: Path
    weight_names: tuple[str, ...] | None


class SafetensorsSource:
    """A discovered local single-file or sharded safetensors checkpoint."""

    def __init__(self, model_path: str | Path) -> None:
        normalized_path = Path(model_path).expanduser().resolve()
        self.model_path = normalized_path
        self._shards = _discover_checkpoint(normalized_path)

    def iter_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield checkpoint tensors from one open shard at a time."""
        for shard in self._shards:
            try:
                with safe_open(
                    shard.path,
                    framework="pt",
                    device="cpu",
                ) as checkpoint:
                    actual_names = tuple(sorted(checkpoint.keys()))
                    weight_names = _validated_weight_names(shard, actual_names)
                    for weight_name in weight_names:
                        yield weight_name, checkpoint.get_tensor(weight_name)
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"checkpoint shard does not exist: {shard.path}"
                ) from error
            except SafetensorError as error:
                raise ValueError(
                    f"failed to read safetensors shard: {shard.path}"
                ) from error


def _discover_checkpoint(model_path: Path) -> tuple[_CheckpointShard, ...]:
    if not model_path.exists():
        raise FileNotFoundError(f"model directory does not exist: {model_path}")
    if not model_path.is_dir():
        raise ValueError(f"model path must be a directory: {model_path}")

    single_file = model_path / _SINGLE_FILE_NAME
    index_file = model_path / _INDEX_FILE_NAME
    has_single_file = single_file.exists()
    has_index_file = index_file.exists()

    if has_single_file and has_index_file:
        raise ValueError(
            f"ambiguous safetensors checkpoint layout in {model_path}: "
            f"both {single_file.name} and {index_file.name} exist"
        )
    if has_single_file:
        if not single_file.is_file():
            raise ValueError(f"checkpoint path is not a file: {single_file}")
        return (_CheckpointShard(path=single_file.resolve(), weight_names=None),)
    if has_index_file:
        if not index_file.is_file():
            raise ValueError(f"checkpoint index path is not a file: {index_file}")
        return _discover_sharded_checkpoint(model_path, index_file)

    raise FileNotFoundError(
        f"no safetensors checkpoint found in model directory: {model_path}"
    )


def _discover_sharded_checkpoint(
    model_path: Path,
    index_path: Path,
) -> tuple[_CheckpointShard, ...]:
    try:
        with index_path.open(encoding="utf-8") as index_file:
            index = json.load(index_file, object_pairs_hook=_object_without_duplicates)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid safetensors index JSON: {index_path}") from error
    except ValueError as error:
        raise ValueError(f"invalid safetensors index {index_path}: {error}") from error

    if not isinstance(index, dict):
        raise ValueError(f"safetensors index must be a JSON object: {index_path}")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(
            f"safetensors index weight_map must be a non-empty object: {index_path}"
        )

    weights_by_shard: dict[Path, list[str]] = {}
    for weight_name, shard_filename in weight_map.items():
        if not isinstance(weight_name, str) or not weight_name.strip():
            raise ValueError(
                f"safetensors index contains an empty weight name: {index_path}"
            )
        if not isinstance(shard_filename, str) or not shard_filename.strip():
            raise ValueError(
                f"safetensors index has an invalid shard filename for "
                f"weight {weight_name!r}: {index_path}"
            )

        shard_path = _resolve_shard_path(
            model_path,
            index_path,
            weight_name,
            shard_filename,
        )
        weights_by_shard.setdefault(shard_path, []).append(weight_name)

    return tuple(
        _CheckpointShard(path=shard_path, weight_names=tuple(sorted(weight_names)))
        for shard_path, weight_names in sorted(
            weights_by_shard.items(),
            key=lambda item: item[0].relative_to(model_path).as_posix(),
        )
    )


def _resolve_shard_path(
    model_path: Path,
    index_path: Path,
    weight_name: str,
    shard_filename: str,
) -> Path:
    relative_path = Path(shard_filename)
    if relative_path.is_absolute():
        raise ValueError(
            f"shard path for weight {weight_name!r} must be relative to "
            f"model directory {model_path}: {shard_filename!r} in {index_path}"
        )

    shard_path = (model_path / relative_path).resolve()
    try:
        shard_path.relative_to(model_path)
    except ValueError as error:
        raise ValueError(
            f"shard path for weight {weight_name!r} leaves model directory "
            f"{model_path}: {shard_filename!r} in {index_path}"
        ) from error

    if not shard_path.exists():
        raise FileNotFoundError(
            f"checkpoint shard for weight {weight_name!r} does not exist: "
            f"{shard_path} (index: {index_path})"
        )
    if not shard_path.is_file():
        raise ValueError(
            f"checkpoint shard for weight {weight_name!r} is not a file: "
            f"{shard_path} (index: {index_path})"
        )
    return shard_path


def _object_without_duplicates(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _validated_weight_names(
    shard: _CheckpointShard,
    actual_names: tuple[str, ...],
) -> tuple[str, ...]:
    if shard.weight_names is None:
        if not actual_names:
            raise ValueError(f"safetensors checkpoint is empty: {shard.path}")
        return actual_names

    expected_names = set(shard.weight_names)
    actual_name_set = set(actual_names)
    missing_names = sorted(expected_names - actual_name_set)
    unexpected_names = sorted(actual_name_set - expected_names)
    if missing_names or unexpected_names:
        details = []
        if missing_names:
            details.append(f"missing weights {missing_names!r}")
        if unexpected_names:
            details.append(f"unexpected weights {unexpected_names!r}")
        raise ValueError(
            f"safetensors shard does not match its index: {shard.path}; "
            + "; ".join(details)
        )
    return shard.weight_names
