"""Read Hugging Face model metadata into micro-vLLM model configs."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from micro_vllm.models.llama import LlamaModelConfig


_CONFIG_FILE_NAME = "config.json"
_MISSING = object()
_REQUIRED_LLAMA_INT_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "max_position_embeddings",
)


@dataclass(frozen=True, slots=True)
class _HFConfig:
    """Validated contents of one local Hugging Face config file."""

    path: Path
    raw: dict[str, Any]
    architectures: tuple[str, ...]


def _read_hf_config(model_path: str | Path) -> _HFConfig:
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
            f"model config architectures must be a non-empty list: {config_path}"
        )
    for index, architecture in enumerate(architectures):
        if not isinstance(architecture, str) or not architecture.strip():
            raise ValueError(
                f"model config architecture at index {index} must be a non-empty "
                f"string: {config_path}"
            )

    return _HFConfig(config_path, raw_config, tuple(architectures))


def _llama_config_from_hf(
    raw_config: Mapping[str, Any],
    config_path: str | Path,
) -> LlamaModelConfig:
    path = Path(config_path)
    values = {
        field_name: _required_positive_int(raw_config, field_name, path)
        for field_name in _REQUIRED_LLAMA_INT_FIELDS
    }
    values["num_key_value_heads"] = _positive_int(
        raw_config.get("num_key_value_heads", values["num_attention_heads"]),
        "num_key_value_heads",
        path,
    )

    rms_norm_eps = _positive_number(
        raw_config.get("rms_norm_eps", 1e-6),
        "rms_norm_eps",
        path,
    )
    rope_theta = _positive_number(
        raw_config.get("rope_theta", 10_000.0),
        "rope_theta",
        path,
    )
    hidden_act = raw_config.get("hidden_act", "silu")
    if hidden_act != "silu":
        raise ValueError(f"hidden_act must be 'silu': {path}")
    tie_word_embeddings = _boolean(
        raw_config.get("tie_word_embeddings", False),
        "tie_word_embeddings",
        path,
    )
    _reject_unsupported_llama_semantics(raw_config, values["num_hidden_layers"], path)

    try:
        config = LlamaModelConfig(
            **values,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            hidden_act=hidden_act,
            tie_word_embeddings=tie_word_embeddings,
        )
    except ValueError as error:
        raise ValueError(f"invalid Llama config {path}: {error}") from error

    head_dim = raw_config.get("head_dim", _MISSING)
    if head_dim not in (_MISSING, None):
        explicit_head_dim = _positive_int(head_dim, "head_dim", path)
        if explicit_head_dim != config.head_dim:
            raise ValueError(
                f"head_dim must equal hidden_size // num_attention_heads "
                f"({config.head_dim}), got {explicit_head_dim}: {path}"
            )
    return config


def _required_positive_int(
    raw_config: Mapping[str, Any],
    field_name: str,
    config_path: Path,
) -> int:
    if field_name not in raw_config:
        raise ValueError(
            f"missing required Llama config field {field_name!r}: {config_path}"
        )
    return _positive_int(raw_config[field_name], field_name, config_path)


def _positive_int(value: object, field_name: str, config_path: Path) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer: {config_path}")
    return value


def _positive_number(value: object, field_name: str, config_path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{field_name} must be a positive number: {config_path}")
    try:
        normalized_value = float(value)
    except OverflowError as error:
        raise ValueError(
            f"{field_name} must be a positive number: {config_path}"
        ) from error
    if not math.isfinite(normalized_value):
        raise ValueError(f"{field_name} must be a positive number: {config_path}")
    return normalized_value


def _boolean(value: object, field_name: str, config_path: Path) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a boolean: {config_path}")
    return value


def _reject_unsupported_llama_semantics(
    raw_config: Mapping[str, Any],
    num_hidden_layers: int,
    config_path: Path,
) -> None:
    for field_name in ("attention_bias", "mlp_bias"):
        if _boolean(raw_config.get(field_name, False), field_name, config_path):
            raise ValueError(f"{field_name}=true is unsupported: {config_path}")

    rope_scaling = raw_config.get("rope_scaling")
    if rope_scaling is not None:
        if not isinstance(rope_scaling, dict):
            raise ValueError(f"rope_scaling must be null or an object: {config_path}")
        if rope_scaling:
            raise ValueError(f"non-empty rope_scaling is unsupported: {config_path}")

    if raw_config.get("sliding_window") is not None:
        raise ValueError(f"sliding_window is unsupported: {config_path}")
    if _boolean(
        raw_config.get("use_sliding_window", False),
        "use_sliding_window",
        config_path,
    ):
        raise ValueError(f"use_sliding_window=true is unsupported: {config_path}")

    attention_type = raw_config.get("attention_type")
    if attention_type not in (None, "full_attention"):
        raise ValueError(f"attention_type must be 'full_attention': {config_path}")
    layer_types = raw_config.get("layer_types")
    if layer_types is not None:
        if not isinstance(layer_types, list) or len(layer_types) != num_hidden_layers:
            raise ValueError(
                f"layer_types must contain one entry per hidden layer: {config_path}"
            )
        if any(layer_type != "full_attention" for layer_type in layer_types):
            raise ValueError(
                f"layer_types must use only 'full_attention': {config_path}"
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
