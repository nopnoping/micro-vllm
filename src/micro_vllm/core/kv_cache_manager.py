"""Logical KV-cache allocation, sharing, preemption, and release."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from micro_vllm.config import LLMConfig
from micro_vllm.core.block import BlockHash, BlockPool, BlockTable
from micro_vllm.core.prefix_cache import hash_block


@dataclass(frozen=True, slots=True)
class KVCacheAllocation:
    """Immutable result of reserving logical KV-cache slots for one request."""

    block_table: tuple[int, ...]
    new_block_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_table", tuple(self.block_table))
        object.__setattr__(self, "new_block_ids", tuple(self.new_block_ids))


class KVCacheManager:
    """Own logical KV-cache state for one full-attention cache group.

    The manager owns the physical BlockPool and one BlockTable per registered
    request. Request objects and physical device tensors remain outside it.
    """

    __slots__ = (
        "_block_pool",
        "_block_size",
        "_block_tables",
        "_enable_prefix_cache",
    )

    def __init__(self, llm_config: LLMConfig) -> None:
        config = llm_config.cache_config
        self._block_pool = BlockPool(cast(int, config.num_blocks))
        self._block_size = config.block_size
        self._block_tables: dict[str, BlockTable] = {}
        self._enable_prefix_cache = config.enable_prefix_cache

    def register_request(self, request_id: str) -> None:
        """Create an empty BlockTable owned by ``request_id``."""

        self._block_tables[request_id] = BlockTable(
            self._block_pool.null_block_id
        )

    def get_block_table(self, request_id: str) -> tuple[int, ...]:
        """Return an immutable snapshot for a registered request."""

        return self._require_block_table(request_id).snapshot()

    @property
    def num_free_blocks(self) -> int:
        """Number of physical blocks currently available for allocation."""

        return self._block_pool.num_free_blocks

    def find_prefix_hit(
        self,
        request_id: str,
        token_ids: Iterable[int],
        *,
        cache_namespace: str,
    ) -> tuple[tuple[int, ...], int]:
        """Find the longest contiguous sequence of cached full blocks.

        At least one known token remains for model computation. Returns
        ``(block_ids, num_hit_tokens)`` without retaining blocks or modifying
        the manager-owned BlockTable or Request state.
        """

        self._require_block_table(request_id)
        if not self._enable_prefix_cache:
            return (), 0

        tokens = tuple(token_ids)
        num_lookup_blocks = max(len(tokens) - 1, 0) // self._block_size
        hit_block_ids: list[int] = []
        parent_block_hash: BlockHash | None = None

        for logical_block_index in range(num_lookup_blocks):
            block_start = logical_block_index * self._block_size
            token_block = tokens[
                block_start : block_start + self._block_size
            ]
            block_hash = hash_block(
                parent_block_hash,
                token_block,
                block_size=self._block_size,
                cache_namespace=cache_namespace,
            )
            cached_block = self._block_pool.get_cached_block(block_hash)
            if cached_block is None:
                break

            hit_block_ids.append(cached_block.block_id)
            parent_block_hash = block_hash

        immutable_block_ids = tuple(hit_block_ids)
        return immutable_block_ids, len(immutable_block_ids) * self._block_size

    def commit_prefix_blocks(
        self,
        request_id: str,
        token_ids: Iterable[int],
        num_computed_tokens: int,
        *,
        cache_namespace: str,
    ) -> tuple[int, ...]:
        """Cache all complete computed blocks and return canonical block IDs."""

        block_table = self._require_block_table(request_id)
        if not self._enable_prefix_cache:
            return ()

        tokens = tuple(token_ids)
        num_full_blocks = num_computed_tokens // self._block_size
        parent_block_hash: BlockHash | None = None
        canonical_block_ids: list[int] = []
        for logical_block_index in range(num_full_blocks):
            block_start = logical_block_index * self._block_size
            token_block = tokens[
                block_start : block_start + self._block_size
            ]
            block_hash = hash_block(
                parent_block_hash,
                token_block,
                block_size=self._block_size,
                cache_namespace=cache_namespace,
            )
            block_id = block_table.physical_block(logical_block_index)
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
            parent_block_hash = block_hash

        return tuple(canonical_block_ids)

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_computed_tokens: int,
        num_scheduled_tokens: int,
    ) -> int:
        """Return missing blocks for computed plus currently scheduled tokens."""

        block_table = self._require_block_table(request_id)
        target_num_tokens = num_computed_tokens + num_scheduled_tokens
        required_num_blocks = (
            target_num_tokens + self._block_size - 1
        ) // self._block_size
        return max(required_num_blocks - block_table.num_logical_blocks, 0)

    def allocate_slots(
        self,
        request_id: str,
        num_computed_tokens: int,
        num_scheduled_tokens: int,
        cached_block_ids: Iterable[int] = (),
    ) -> KVCacheAllocation | None:
        """Atomically reserve slots for the currently scheduled token range."""

        block_table = self._require_block_table(request_id)
        cached_ids = tuple(cached_block_ids)
        num_new_blocks = self.get_num_blocks_to_allocate(
            request_id,
            num_computed_tokens,
            num_scheduled_tokens,
        ) - len(cached_ids)

        new_block_ids = self._block_pool.allocate(
            num_new_blocks,
            cached_ids,
        )
        if new_block_ids is None:
            return None

        block_table.extend(cached_ids)
        block_table.extend(new_block_ids)
        return KVCacheAllocation(
            block_table=block_table.snapshot(),
            new_block_ids=new_block_ids,
        )

    def free_request_blocks(self, request_id: str) -> None:
        """Release all block references while keeping an empty request table."""

        block_table = self._require_block_table(request_id)
        for block_id in block_table.clear():
            self._block_pool.release(block_id)

    def free_request(self, request_id: str) -> None:
        """Release all request block references and remove its owned table.

        Raises ``KeyError`` when the request is not registered, including when
        it has already been freed.
        """

        self.free_request_blocks(request_id)
        del self._block_tables[request_id]

    def _require_block_table(self, request_id: str) -> BlockTable:
        return self._block_tables[request_id]
