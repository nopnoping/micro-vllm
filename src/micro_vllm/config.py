"""Top-level configuration objects shared across micro-vLLM components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


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

    def get(self, name: str, default: object = None) -> object:
        return self._values.get(name, default)

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
