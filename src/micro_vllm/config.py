"""Top-level configuration objects shared across micro-vLLM components."""

from dataclasses import dataclass


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

    def __post_init__(self) -> None:
        for field_name in ("max_num_batched_tokens", "max_num_seqs"):
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be greater than zero")
