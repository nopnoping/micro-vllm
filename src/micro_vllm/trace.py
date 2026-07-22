"""Immutable structured trace events emitted by the inference runtime."""

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import TypeAlias, cast

from micro_vllm.attention.spec import AttentionSpec

__all__ = (
    "AttentionSpecTrace",
    "KVCacheGroupTrace",
    "KVCacheLayoutTrace",
    "KVCacheStepTrace",
    "TraceEvent",
    "TraceSink",
)


@dataclass(frozen=True, slots=True)
class AttentionSpecTrace:
    """Serializable attention semantic name and integer parameters."""

    name: str
    parameters: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class KVCacheGroupTrace:
    """One cache group's state after a successful step commit."""

    group_index: int
    attention_spec: AttentionSpecTrace
    layer_indices: tuple[int, ...]
    block_table: tuple[int, ...]
    released_block_count: int


@dataclass(frozen=True, slots=True)
class KVCacheStepTrace:
    """Per-request cache state committed by one successful engine step."""

    request_id: str
    num_computed_tokens: int
    groups: tuple[KVCacheGroupTrace, ...]


@dataclass(frozen=True, slots=True)
class KVCacheLayoutTrace:
    """Physical cache allocation selected during runner initialization."""

    page_layer_count: int
    physical_buffer_count: int
    num_blocks: int
    block_size: int
    dtype: str
    total_bytes: int


TraceEvent: TypeAlias = KVCacheStepTrace | KVCacheLayoutTrace
TraceSink: TypeAlias = Callable[[TraceEvent], None]


def snapshot_attention_spec(spec: AttentionSpec) -> AttentionSpecTrace:
    """Convert one immutable spec into a stable trace value."""

    return AttentionSpecTrace(
        name=type(spec).__name__,
        parameters=tuple(
            (field.name, cast(int, getattr(spec, field.name)))
            for field in fields(spec)
        ),
    )
