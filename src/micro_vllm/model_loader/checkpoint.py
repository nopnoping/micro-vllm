"""Stream tensors from local safetensors checkpoints."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open


_SINGLE_FILE_NAME = "model.safetensors"
_INDEX_FILE_NAME = "model.safetensors.index.json"


@dataclass(frozen=True, slots=True)
class _CheckpointShard:
    path: Path
    weight_names: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class SafetensorsCheckpoint:
    """A local safetensors checkpoint that streams its tensors."""

    model_path: Path

    def iter_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield checkpoint tensors while opening one shard at a time."""
        for shard in _discover_checkpoint(self.model_path):
            try:
                with safe_open(
                    shard.path,
                    framework="pt",
                    device="cpu",
                ) as checkpoint:
                    weight_names = shard.weight_names
                    if weight_names is None:
                        weight_names = tuple(sorted(checkpoint.keys()))
                        if not weight_names:
                            raise ValueError(
                                f"safetensors checkpoint is empty: {shard.path}"
                            )
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
    single_file = model_path / _SINGLE_FILE_NAME
    index_file = model_path / _INDEX_FILE_NAME

    if single_file.is_file() and index_file.is_file():
        raise ValueError(
            f"ambiguous safetensors checkpoint layout in {model_path}: "
            f"both {single_file.name} and {index_file.name} exist"
        )
    if single_file.is_file():
        return (_CheckpointShard(single_file, None),)
    if index_file.is_file():
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
            index = json.load(index_file)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(
            f"invalid safetensors index JSON: {index_path}"
        ) from error

    if not isinstance(index, dict):
        raise ValueError(f"safetensors index must be a JSON object: {index_path}")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(
            f"safetensors index weight_map must be a non-empty object: "
            f"{index_path}"
        )

    weights_by_shard: dict[Path, list[str]] = {}
    for weight_name, shard_filename in weight_map.items():
        if not isinstance(weight_name, str) or not weight_name:
            raise ValueError(f"invalid weight name in safetensors index: {index_path}")
        if not isinstance(shard_filename, str) or not shard_filename:
            raise ValueError(
                f"invalid shard filename in safetensors index: {index_path}"
            )
        shard_path = _resolve_shard_path(model_path, index_path, shard_filename)
        weights_by_shard.setdefault(shard_path, []).append(weight_name)

    return tuple(
        _CheckpointShard(path, tuple(sorted(weight_names)))
        for path, weight_names in sorted(weights_by_shard.items())
    )


def _resolve_shard_path(
    model_path: Path,
    index_path: Path,
    shard_filename: str,
) -> Path:
    relative_path = Path(shard_filename)
    shard_path = (model_path / relative_path).resolve()
    if relative_path.is_absolute() or not shard_path.is_relative_to(model_path):
        raise ValueError(
            f"checkpoint shard leaves model directory {model_path}: "
            f"{shard_filename!r} in {index_path}"
        )
    if not shard_path.is_file():
        raise FileNotFoundError(
            f"checkpoint shard does not exist: {shard_path} "
            f"(index: {index_path})"
        )
    return shard_path
