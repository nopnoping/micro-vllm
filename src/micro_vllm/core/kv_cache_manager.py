"""Coordinate logical KV-cache state across attention cache groups."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from micro_vllm.attention.spec import FullAttentionSpec
from micro_vllm.config import LLMConfig
from micro_vllm.core.block import BlockPool, BlockTable
from micro_vllm.core.cache_group_manager import (
    CacheGroupHit,
    create_cache_group_manager,
)
from micro_vllm.core.kv_cache_spec import (
    KVCacheLayout,
    build_kv_cache_topology,
)


@dataclass(frozen=True, slots=True)
class KVCacheHit:
    """Common reusable prefix with KV-internal per-group snapshots."""

    num_computed_tokens: int
    _group_hits: tuple[CacheGroupHit, ...]


@dataclass(frozen=True, slots=True)
class KVCacheAllocation:
    """Immutable per-group tables after one atomic shared-pool allocation."""

    block_tables: tuple[tuple[int, ...], ...]
    new_block_ids: tuple[int, ...]

    @property
    def block_table(self) -> tuple[int, ...]:
        """Return the first group table for the current full-only runner path."""

        return self.block_tables[0]


class KVCacheManager:
    """Own one shared block pool and coordinate all logical cache groups."""

    __slots__ = (
        "_block_pool",
        "_block_tables",
        "_group_managers",
        "_layout",
    )

    def __init__(
        self,
        llm_config: LLMConfig,
        layout: KVCacheLayout | None = None,
        *,
        cache_namespace: str = "default",
    ) -> None:
        config = llm_config.cache_config
        if layout is None:
            layout = KVCacheLayout(
                block_size=config.block_size,
                num_blocks=cast(int, config.num_blocks),
                topology=build_kv_cache_topology((FullAttentionSpec(),)),
            )

        block_pool = BlockPool(layout.num_blocks)
        self._layout = layout
        self._block_pool = block_pool
        self._group_managers = tuple(
            create_cache_group_manager(
                group_spec,
                block_size=layout.block_size,
                block_pool=block_pool,
                cache_namespace=cache_namespace,
                enable_prefix_cache=config.enable_prefix_cache,
            )
            for group_spec in layout.groups
        )
        self._block_tables: dict[str, tuple[BlockTable, ...]] = {}

    def register_request(self, request_id: str) -> None:
        """Create one empty logical block table per cache group."""

        self._block_tables[request_id] = tuple(
            BlockTable(self._block_pool.null_block_id)
            for _ in self._group_managers
        )

    def get_block_tables(
        self,
        request_id: str,
    ) -> tuple[tuple[int, ...], ...]:
        """Return immutable snapshots in stable topology group order."""

        return tuple(
            block_table.snapshot()
            for block_table in self._require_block_tables(request_id)
        )

    def get_block_table(self, request_id: str) -> tuple[int, ...]:
        """Return the first group table for the current full-only runner path."""

        return self.get_block_tables(request_id)[0]

    @property
    def num_free_blocks(self) -> int:
        return self._block_pool.num_free_blocks

    def find_longest_cache_hit(
        self,
        request_id: str,
        token_ids: Iterable[int],
    ) -> KVCacheHit:
        """Find the fixed-point prefix valid for every cache group."""

        self._require_block_tables(request_id)
        tokens = tuple(token_ids)
        candidate = max(len(tokens) - 1, 0)

        while True:
            group_hits = tuple(
                manager.find_longest_cache_hit(tokens, candidate)
                for manager in self._group_managers
            )
            next_candidate = min(
                hit.num_computed_tokens for hit in group_hits
            )
            if next_candidate == candidate:
                return KVCacheHit(candidate, group_hits)
            candidate = next_candidate

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_computed_tokens: int,
        num_scheduled_tokens: int,
    ) -> int:
        """Return the aggregate number of missing blocks across all groups."""

        return sum(
            len(
                manager.get_new_block_indices(
                    block_table,
                    num_computed_tokens,
                    num_scheduled_tokens,
                )
            )
            for manager, block_table in zip(
                self._group_managers,
                self._require_block_tables(request_id),
                strict=True,
            )
        )

    def allocate_slots(
        self,
        request_id: str,
        num_computed_tokens: int,
        num_scheduled_tokens: int,
        cache_hit: KVCacheHit | None = None,
    ) -> KVCacheAllocation | None:
        """Atomically acquire cached blocks and allocate every group slot."""

        block_tables = self._require_block_tables(request_id)
        new_indices_by_group: list[tuple[int, ...]] = []
        cached_block_ids: list[int] = []

        for group_index, (manager, block_table) in enumerate(
            zip(self._group_managers, block_tables, strict=True)
        ):
            planning_table = BlockTable(self._block_pool.null_block_id)
            if cache_hit is None:
                planning_table.extend(block_table.snapshot())
            else:
                hit_table = cache_hit._group_hits[group_index].block_table
                planning_table.extend(hit_table)
                cached_block_ids.extend(
                    block_id
                    for block_id in hit_table
                    if block_id != self._block_pool.null_block_id
                )

            new_indices_by_group.append(
                manager.get_new_block_indices(
                    planning_table,
                    num_computed_tokens,
                    num_scheduled_tokens,
                )
            )

        num_new_blocks = sum(map(len, new_indices_by_group))
        new_block_ids = self._block_pool.allocate(
            num_new_blocks,
            cached_block_ids,
        )
        if new_block_ids is None:
            return None

        next_new_block = 0
        for group_index, block_table in enumerate(block_tables):
            if cache_hit is not None:
                block_table.extend(
                    cache_hit._group_hits[group_index].block_table
                )
            new_indices = new_indices_by_group[group_index]
            group_new_block_ids = new_block_ids[
                next_new_block : next_new_block + len(new_indices)
            ]
            block_table.fill_missing(new_indices, group_new_block_ids)
            next_new_block += len(new_indices)

        return KVCacheAllocation(
            block_tables=self.get_block_tables(request_id),
            new_block_ids=new_block_ids,
        )

    def commit_step(
        self,
        request_id: str,
        token_ids: Iterable[int],
        num_computed_tokens: int,
    ) -> None:
        """Commit prefix identities, then release policy-skipped blocks."""

        tokens = tuple(token_ids)
        for manager, block_table in zip(
            self._group_managers,
            self._require_block_tables(request_id),
            strict=True,
        ):
            manager.cache_computed_blocks(
                block_table,
                tokens,
                num_computed_tokens,
            )
            manager.remove_skipped_blocks(
                block_table,
                num_computed_tokens,
            )

    def free_request_blocks(self, request_id: str) -> None:
        """Release all groups while preserving empty tables for recompute."""

        for block_table in self._require_block_tables(request_id):
            for block_id in block_table.clear():
                self._block_pool.release(block_id)

    def free_request(self, request_id: str) -> None:
        """Release all cache groups and remove the request registry entry."""

        self.free_request_blocks(request_id)
        del self._block_tables[request_id]

    def _require_block_tables(
        self,
        request_id: str,
    ) -> tuple[BlockTable, ...]:
        return self._block_tables[request_id]
