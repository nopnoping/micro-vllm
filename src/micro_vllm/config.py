"""Top-level configuration objects shared across micro-vLLM components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from micro_vllm.tokenizer import Tokenizer


@dataclass(frozen=True, slots=True, init=False)
class ModelConfig:
    """Read-only model metadata shared by every registered architecture."""

    _values: Mapping[str, object]
    architectures: tuple[str, ...]

    def __init__(self, values: Mapping[str, object]) -> None:
        copied_values = dict(values)
        architectures = copied_values.get("architectures")
        if not isinstance(architectures, list) or not architectures:
            raise ValueError("architectures must be a non-empty list")
        for index, architecture in enumerate(architectures):
            if not isinstance(architecture, str) or not architecture.strip():
                raise ValueError(
                    f"architecture at index {index} must be a non-empty string"
                )

        object.__setattr__(self, "_values", MappingProxyType(copied_values))
        object.__setattr__(self, "architectures", tuple(architectures))

    @property
    def values(self) -> Mapping[str, object]:
        return self._values

    def require_positive_int(self, name: str) -> int:
        if name not in self._values:
            raise ValueError(f"missing required model config field {name!r}")
        value = self._values[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def positive_int(self, name: str, default: int) -> int:
        value = self._values.get(name, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """Device-independent logical KV-cache configuration.

    Prefix caching has no separate capacity limit. Cached zero-reference
    blocks remain reclaimable by the block pool and lose their hash mapping
    automatically when their physical storage is reused.
    """

    block_size: int
    num_blocks: int
    enable_prefix_cache: bool = False

    def __post_init__(self) -> None:
        for field_name in ("block_size", "num_blocks"):
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be greater than zero")


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Device-independent limits enforced by the scheduler."""

    max_num_batched_tokens: int
    max_num_seqs: int
    max_model_len: int

    def __post_init__(self) -> None:
        for field_name in (
            "max_num_batched_tokens",
            "max_num_seqs",
            "max_model_len",
        ):
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be greater than zero")


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """User-provided configuration for one synchronous offline LLM."""

    model: str | Path
    tokenizer: Tokenizer | None = None
    tokenizer_name_or_path: str | Path | None = None
    revision: str | None = None
    cache_dir: str | Path | None = None
    token: str | bool | None = None
    local_files_only: bool = False
    max_model_len: int = 2048
    max_num_batched_tokens: int = 512
    max_num_seqs: int = 8
    block_size: int = 16
    gpu_memory_utilization: float = 0.9
    max_num_blocks: int | None = None
    enable_prefix_cache: bool = False
    device: torch.device | str = "cuda"
    dtype: torch.dtype = torch.float16
