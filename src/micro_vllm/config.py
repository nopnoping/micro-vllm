"""Top-level configuration objects shared across micro-vLLM components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from micro_vllm.trace import TraceSink

if TYPE_CHECKING:
    from micro_vllm.tokenizer import Tokenizer


_EMPTY_MODEL_METADATA: Mapping[str, object] = MappingProxyType({})


@dataclass(slots=True)
class ModelConfig:
    """Model source, construction options, and loaded metadata."""

    model: str | Path
    tokenizer: Tokenizer | None = None
    tokenizer_name_or_path: str | Path | None = None
    revision: str | None = None
    cache_dir: str | Path | None = None
    token: str | bool | None = None
    local_files_only: bool = False
    attention_operator: str = "reference"
    enable_cuda_graph: bool = False
    device: torch.device | str = "cuda"
    dtype: torch.dtype = torch.float16
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        device_type = torch.device(self.device).type
        if device_type not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if self.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise ValueError("dtype must be float32, float16, or bfloat16")
        if self.enable_cuda_graph:
            if device_type != "cuda":
                raise ValueError("CUDA Graph execution requires a CUDA device")
            if self.attention_operator != "triton_paged":
                raise ValueError(
                    "CUDA Graph execution currently requires "
                    "attention_operator='triton_paged'"
                )
            if self.dtype != torch.float16:
                raise ValueError(
                    "CUDA Graph execution with triton_paged currently "
                    "requires torch.float16"
                )
        if self.metadata is not None:
            self.set_metadata(self.metadata)

    def set_metadata(self, values: Mapping[str, object]) -> None:
        """Store validated model metadata as a read-only mapping."""

        copied_values = dict(values)
        architectures = copied_values.get("architectures")
        if not isinstance(architectures, list) or not architectures:
            raise ValueError("architectures must be a non-empty list")
        for index, architecture in enumerate(architectures):
            if not isinstance(architecture, str) or not architecture.strip():
                raise ValueError(
                    f"architecture at index {index} must be a non-empty string"
                )

        self.metadata = MappingProxyType(copied_values)

    @property
    def architectures(self) -> tuple[str, ...]:
        return tuple(self.values.get("architectures", ()))

    @property
    def values(self) -> Mapping[str, object]:
        return self.metadata or _EMPTY_MODEL_METADATA

    def require_positive_int(self, name: str) -> int:
        if name not in self.values:
            raise ValueError(f"missing required model config field {name!r}")
        value = self.values[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def positive_int(self, name: str, default: int) -> int:
        value = self.values.get(name, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value


@dataclass(slots=True)
class CacheConfig:
    """KV-cache configuration shared across initialization and execution.

    Before initialization, ``num_blocks`` optionally limits total GPU pages
    and provides the deterministic CPU page count. One page is permanently
    reserved as the shared null block. The model runner replaces the value
    with the actual total page count during initialization.
    """

    block_size: int = 16
    gpu_memory_utilization: float = 0.9
    num_blocks: int | None = None
    enable_prefix_cache: bool = False

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be greater than zero")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError(
                "gpu_memory_utilization must be greater than zero and at most one"
            )
        if self.num_blocks is not None and self.num_blocks < 2:
            raise ValueError("num_blocks must include a data and null block")


@dataclass(slots=True)
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


@dataclass(slots=True)
class LLMConfig:
    """Configuration shared by one synchronous offline LLM."""

    model_config: ModelConfig
    scheduler_config: SchedulerConfig = field(
        default_factory=lambda: SchedulerConfig(
            max_num_batched_tokens=512,
            max_num_seqs=8,
            max_model_len=2048,
        )
    )
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    trace_sink: TraceSink | None = None
