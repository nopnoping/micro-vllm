"""Immutable, device-independent attention visibility semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AttentionSpec(ABC):
    """Define the first key visible to one absolute query position."""

    @abstractmethod
    def context_start(self, query_position: int) -> int:
        """Return the inclusive start of the query's visible key interval."""


@dataclass(frozen=True, slots=True)
class FullAttentionSpec(AttentionSpec):
    """Retain the complete causal history."""

    def context_start(self, query_position: int) -> int:
        return 0


@dataclass(frozen=True, slots=True)
class SlidingWindowSpec(AttentionSpec):
    """Retain at most ``window_size`` keys, including the query token."""

    window_size: int

    def __post_init__(self) -> None:
        if type(self.window_size) is not int or self.window_size <= 0:
            raise ValueError("window_size must be a positive integer")

    def context_start(self, query_position: int) -> int:
        return max(0, query_position - self.window_size + 1)


@dataclass(frozen=True, slots=True)
class ChunkedLocalSpec(AttentionSpec):
    """Retain the causal prefix within one aligned local chunk."""

    chunk_size: int

    def __post_init__(self) -> None:
        if type(self.chunk_size) is not int or self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

    def context_start(self, query_position: int) -> int:
        return (query_position // self.chunk_size) * self.chunk_size
