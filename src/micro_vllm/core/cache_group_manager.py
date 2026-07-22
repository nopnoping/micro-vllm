"""Attention-specific logical KV-cache policies over one shared block pool."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from micro_vllm.attention.spec import (
    AttentionSpec,
    ChunkedLocalSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
)
from micro_vllm.core.block import (
    BlockHash,
    BlockPool,
    BlockTable,
    KVCacheBlock,
)
from micro_vllm.core.kv_cache_spec import KVCacheGroupSpec
from micro_vllm.core.prefix_cache import hash_block


@dataclass(frozen=True, slots=True)
class CacheGroupHit:
    """Longest reusable prefix and its absolute logical block snapshot."""

    num_computed_tokens: int
    block_table: tuple[int, ...]


class CacheGroupManager(ABC):
    """Apply one cache group's policy without owning physical capacity."""

    __slots__ = (
        "_attention_spec",
        "_block_pool",
        "_block_size",
        "_cache_namespace",
        "_enable_prefix_cache",
        "_group_id",
    )

    def __init__(
        self,
        group_spec: KVCacheGroupSpec,
        *,
        block_size: int,
        block_pool: BlockPool,
        cache_namespace: str,
        enable_prefix_cache: bool,
    ) -> None:
        self._group_id = group_spec.group_id
        self._attention_spec = group_spec.attention_spec
        self._block_size = block_size
        self._block_pool = block_pool
        self._cache_namespace = cache_namespace
        self._enable_prefix_cache = enable_prefix_cache

    @property
    def attention_spec(self) -> AttentionSpec:
        return self._attention_spec

    @abstractmethod
    def find_longest_cache_hit(
        self,
        token_ids: Iterable[int],
        max_length: int,
    ) -> CacheGroupHit:
        """Return the longest reusable prefix no greater than ``max_length``."""

    def get_new_block_indices(
        self,
        block_table: BlockTable,
        num_computed_tokens: int,
        num_scheduled_tokens: int,
    ) -> tuple[int, ...]:
        """Return missing absolute logical blocks touched by scheduled tokens."""

        if num_scheduled_tokens == 0:
            return ()

        first_block = num_computed_tokens // self._block_size
        last_token = num_computed_tokens + num_scheduled_tokens - 1
        last_block = last_token // self._block_size
        return tuple(
            logical_block_index
            for logical_block_index in range(first_block, last_block + 1)
            if (
                logical_block_index >= block_table.num_logical_blocks
                or block_table.physical_block(logical_block_index)
                == self._block_pool.null_block_id
            )
        )

    def cache_computed_blocks(
        self,
        block_table: BlockTable,
        token_ids: Iterable[int],
        num_computed_tokens: int,
    ) -> tuple[int, ...]:
        """Cache retained complete blocks and return their canonical IDs."""

        if not self._enable_prefix_cache:
            return ()

        num_full_blocks = num_computed_tokens // self._block_size
        block_hashes = self._get_block_hashes(token_ids, num_full_blocks)
        canonical_block_ids: list[int] = []
        for logical_block_index, block_hash in enumerate(block_hashes):
            block_id = block_table.physical_block(logical_block_index)
            if block_id == self._block_pool.null_block_id:
                continue

            canonical_block = self._block_pool.get_cached_block(block_hash)
            if canonical_block is None:
                canonical_block = self._block_pool.cache_block(
                    block_id,
                    block_hash,
                )
            elif canonical_block.block_id != block_id:
                self._block_pool.acquire_cached_block(block_hash)
                block_table.replace(
                    logical_block_index,
                    canonical_block.block_id,
                )
                self._block_pool.release(block_id)
            canonical_block_ids.append(canonical_block.block_id)

        return tuple(canonical_block_ids)

    def remove_skipped_blocks(
        self,
        block_table: BlockTable,
        num_computed_tokens: int,
    ) -> tuple[int, ...]:
        """Release complete blocks before this policy's retained boundary."""

        first_retained_block = (
            self._get_num_skipped_tokens(num_computed_tokens)
            // self._block_size
        )
        released_block_ids = block_table.release_before(first_retained_block)
        for block_id in released_block_ids:
            self._block_pool.release(block_id)
        return released_block_ids

    def _find_bounded_cache_hit(
        self,
        token_ids: Iterable[int],
        max_length: int,
    ) -> CacheGroupHit:
        max_num_blocks = max_length // self._block_size
        block_hashes = self._get_block_hashes(token_ids, max_num_blocks)

        for num_hit_blocks in range(max_num_blocks, -1, -1):
            num_hit_tokens = num_hit_blocks * self._block_size
            first_required_token = self._attention_spec.context_start(
                num_hit_tokens
            )
            first_required_block = first_required_token // self._block_size
            cached_block_ids: list[int] = []
            for logical_block_index in range(
                first_required_block,
                num_hit_blocks,
            ):
                cached_block = self._get_cached_block(
                    block_hashes[logical_block_index]
                )
                if cached_block is None:
                    break
                cached_block_ids.append(cached_block.block_id)
            else:
                block_table = [self._block_pool.null_block_id] * num_hit_blocks
                block_table[first_required_block:] = cached_block_ids
                return CacheGroupHit(
                    num_computed_tokens=num_hit_tokens,
                    block_table=tuple(block_table),
                )

        return CacheGroupHit(0, ())

    def _get_cached_block(
        self,
        block_hash: BlockHash,
    ) -> KVCacheBlock | None:
        if not self._enable_prefix_cache:
            return None
        return self._block_pool.get_cached_block(block_hash)

    def _get_block_hashes(
        self,
        token_ids: Iterable[int],
        num_blocks: int,
    ) -> tuple[BlockHash, ...]:
        tokens = tuple(token_ids)
        parent_block_hash: BlockHash | None = None
        block_hashes: list[BlockHash] = []
        for logical_block_index in range(num_blocks):
            block_start = logical_block_index * self._block_size
            block_hash = hash_block(
                parent_block_hash,
                tokens[block_start : block_start + self._block_size],
                block_size=self._block_size,
                cache_namespace=self._cache_namespace,
                cache_group_id=self._group_id,
            )
            block_hashes.append(block_hash)
            parent_block_hash = block_hash
        return tuple(block_hashes)

    @abstractmethod
    def _get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """Return the number of leading tokens no longer retained."""


class FullCacheGroupManager(CacheGroupManager):
    """Full-attention prefix and retention policy."""

    __slots__ = ()

    def find_longest_cache_hit(
        self,
        token_ids: Iterable[int],
        max_length: int,
    ) -> CacheGroupHit:
        if not self._enable_prefix_cache:
            return CacheGroupHit(0, ())

        max_num_blocks = max_length // self._block_size
        cached_block_ids: list[int] = []
        for block_hash in self._get_block_hashes(token_ids, max_num_blocks):
            cached_block = self._block_pool.get_cached_block(block_hash)
            if cached_block is None:
                break
            cached_block_ids.append(cached_block.block_id)
        return CacheGroupHit(
            num_computed_tokens=len(cached_block_ids) * self._block_size,
            block_table=tuple(cached_block_ids),
        )

    def _get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        return 0


class SlidingWindowCacheGroupManager(CacheGroupManager):
    """Sliding-window prefix and retention policy."""

    __slots__ = ()

    def find_longest_cache_hit(
        self,
        token_ids: Iterable[int],
        max_length: int,
    ) -> CacheGroupHit:
        if not self._enable_prefix_cache:
            return CacheGroupHit(0, ())
        return self._find_bounded_cache_hit(token_ids, max_length)

    def _get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        window_size = cast(SlidingWindowSpec, self._attention_spec).window_size
        return max(0, num_computed_tokens - window_size + 1)


class ChunkedLocalCacheGroupManager(CacheGroupManager):
    """Chunk-local prefix and retention policy."""

    __slots__ = ()

    def find_longest_cache_hit(
        self,
        token_ids: Iterable[int],
        max_length: int,
    ) -> CacheGroupHit:
        return self._find_bounded_cache_hit(token_ids, max_length)

    def _get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        chunk_size = cast(ChunkedLocalSpec, self._attention_spec).chunk_size
        return (num_computed_tokens // chunk_size) * chunk_size


def create_cache_group_manager(
    group_spec: KVCacheGroupSpec,
    *,
    block_size: int,
    block_pool: BlockPool,
    cache_namespace: str,
    enable_prefix_cache: bool,
) -> CacheGroupManager:
    """Create the policy matching one topology-owned attention spec."""

    if isinstance(group_spec.attention_spec, FullAttentionSpec):
        manager_class = FullCacheGroupManager
    elif isinstance(group_spec.attention_spec, SlidingWindowSpec):
        manager_class = SlidingWindowCacheGroupManager
    elif isinstance(group_spec.attention_spec, ChunkedLocalSpec):
        manager_class = ChunkedLocalCacheGroupManager
    else:
        raise NotImplementedError(
            f"unsupported attention spec {group_spec.attention_spec!r}"
        )
    return manager_class(
        group_spec,
        block_size=block_size,
        block_pool=block_pool,
        cache_namespace=cache_namespace,
        enable_prefix_cache=enable_prefix_cache,
    )
