"""Tests for attention-specific KV-cache group policies."""

from micro_vllm.attention.spec import (
    AttentionSpec,
    ChunkedLocalSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
)
from micro_vllm.core.block import BlockPool, BlockTable
from micro_vllm.core.cache_group_manager import (
    CacheGroupManager,
    ChunkedLocalCacheGroupManager,
    FullCacheGroupManager,
    SlidingWindowCacheGroupManager,
    create_cache_group_manager,
)
from micro_vllm.core.kv_cache_spec import KVCacheGroupSpec


def make_manager(
    attention_spec: AttentionSpec,
    *,
    group_id: int = 0,
    block_pool: BlockPool | None = None,
    enable_prefix_cache: bool = True,
) -> tuple[CacheGroupManager, BlockPool]:
    pool = block_pool or BlockPool(16)
    manager = create_cache_group_manager(
        KVCacheGroupSpec(
            group_id=group_id,
            attention_spec=attention_spec,
            layer_indices=(group_id,),
        ),
        block_size=2,
        block_pool=pool,
        cache_namespace="model-a",
        enable_prefix_cache=enable_prefix_cache,
    )
    return manager, pool


def cache_logical_blocks(
    manager: CacheGroupManager,
    pool: BlockPool,
    token_ids: tuple[int, ...],
    logical_block_indices: tuple[int, ...],
) -> tuple[int, ...]:
    block_ids = pool.allocate(len(logical_block_indices))
    assert block_ids is not None
    block_table = BlockTable(pool.null_block_id)
    block_table.fill_missing(logical_block_indices, block_ids)
    manager.cache_computed_blocks(
        block_table,
        token_ids,
        (max(logical_block_indices) + 1) * 2,
    )
    for block_id in block_table.clear():
        pool.release(block_id)
    return block_ids


def make_allocated_table(
    pool: BlockPool,
    num_blocks: int,
) -> tuple[BlockTable, tuple[int, ...]]:
    block_ids = pool.allocate(num_blocks)
    assert block_ids is not None
    block_table = BlockTable(pool.null_block_id)
    block_table.extend(block_ids)
    return block_table, block_ids


def test_factory_selects_policy_and_managers_share_the_supplied_pool() -> None:
    pool = BlockPool(8)
    full, _ = make_manager(FullAttentionSpec(), block_pool=pool)
    sliding, _ = make_manager(SlidingWindowSpec(4), block_pool=pool)
    chunked, _ = make_manager(ChunkedLocalSpec(4), block_pool=pool)

    assert isinstance(full, FullCacheGroupManager)
    assert isinstance(sliding, SlidingWindowCacheGroupManager)
    assert isinstance(chunked, ChunkedLocalCacheGroupManager)
    assert full._block_pool is sliding._block_pool is chunked._block_pool is pool


def test_full_prefix_stops_at_the_first_missing_block() -> None:
    manager, pool = make_manager(FullAttentionSpec())
    token_ids = (1, 2, 3, 4, 5, 6, 7)
    first_block_id, _ = cache_logical_blocks(
        manager,
        pool,
        token_ids,
        (0, 2),
    )

    hit = manager.find_longest_cache_hit(token_ids, max_length=6)

    assert hit.num_computed_tokens == 2
    assert hit.block_table == (first_block_id,)


def test_sliding_window_only_requires_blocks_visible_at_candidate() -> None:
    manager, pool = make_manager(SlidingWindowSpec(3))
    token_ids = (1, 2, 3, 4, 5, 6, 7)
    (visible_block_id,) = cache_logical_blocks(
        manager,
        pool,
        token_ids,
        (2,),
    )

    hit = manager.find_longest_cache_hit(token_ids, max_length=6)

    assert hit.num_computed_tokens == 6
    assert hit.block_table == (
        pool.null_block_id,
        pool.null_block_id,
        visible_block_id,
    )


def test_sliding_window_retries_at_a_shorter_candidate() -> None:
    manager, pool = make_manager(SlidingWindowSpec(3))
    token_ids = (1, 2, 3, 4, 5, 6, 7)
    (visible_block_id,) = cache_logical_blocks(
        manager,
        pool,
        token_ids,
        (1,),
    )

    hit = manager.find_longest_cache_hit(token_ids, max_length=6)

    assert hit.num_computed_tokens == 4
    assert hit.block_table == (pool.null_block_id, visible_block_id)


def test_chunked_local_skips_completed_chunks_without_cached_blocks() -> None:
    manager, pool = make_manager(
        ChunkedLocalSpec(4),
        enable_prefix_cache=False,
    )

    hit = manager.find_longest_cache_hit(
        (1, 2, 3, 4, 5, 6, 7),
        max_length=6,
    )

    assert hit.num_computed_tokens == 4
    assert hit.block_table == (pool.null_block_id, pool.null_block_id)


def test_chunked_local_uses_cached_blocks_in_the_current_chunk() -> None:
    manager, pool = make_manager(ChunkedLocalSpec(4))
    token_ids = (1, 2, 3, 4, 5, 6, 7)
    (current_chunk_block_id,) = cache_logical_blocks(
        manager,
        pool,
        token_ids,
        (2,),
    )

    hit = manager.find_longest_cache_hit(token_ids, max_length=6)

    assert hit.num_computed_tokens == 6
    assert hit.block_table == (
        pool.null_block_id,
        pool.null_block_id,
        current_chunk_block_id,
    )


def test_group_id_is_bound_inside_each_manager_cache_identity() -> None:
    pool = BlockPool(8)
    first, _ = make_manager(FullAttentionSpec(), group_id=0, block_pool=pool)
    second, _ = make_manager(FullAttentionSpec(), group_id=1, block_pool=pool)
    token_ids = (1, 2, 3)
    cache_logical_blocks(first, pool, token_ids, (0,))

    assert first.find_longest_cache_hit(token_ids, 2).num_computed_tokens == 2
    assert second.find_longest_cache_hit(token_ids, 2).num_computed_tokens == 0


def test_scheduled_allocation_reports_absolute_missing_block_indices() -> None:
    manager, pool = make_manager(SlidingWindowSpec(3))
    block_table = BlockTable(pool.null_block_id)
    block_ids = pool.allocate(1)
    assert block_ids is not None
    (block_id,) = block_ids
    block_table.fill_missing((2,), (block_id,))

    assert manager.get_new_block_indices(block_table, 4, 4) == (3,)
    assert manager.get_new_block_indices(block_table, 8, 0) == ()


def test_cache_computed_blocks_reuses_group_canonical_block() -> None:
    manager, pool = make_manager(FullAttentionSpec())
    first_table, (first_block_id,) = make_allocated_table(pool, 1)
    second_table, (duplicate_block_id,) = make_allocated_table(pool, 1)
    token_ids = (1, 2)

    assert manager.cache_computed_blocks(first_table, token_ids, 2) == (
        first_block_id,
    )
    assert manager.cache_computed_blocks(second_table, token_ids, 2) == (
        first_block_id,
    )

    assert second_table.snapshot() == (first_block_id,)
    assert pool.ref_count(first_block_id) == 2
    assert pool.ref_count(duplicate_block_id) == 0


def test_full_policy_never_releases_history() -> None:
    manager, pool = make_manager(FullAttentionSpec())
    block_table, block_ids = make_allocated_table(pool, 3)

    assert manager.remove_skipped_blocks(block_table, 6) == ()
    assert block_table.snapshot() == block_ids


def test_sliding_window_releases_only_fully_skipped_blocks() -> None:
    manager, pool = make_manager(SlidingWindowSpec(3))
    block_table, block_ids = make_allocated_table(pool, 3)

    assert manager.remove_skipped_blocks(block_table, 6) == block_ids[:2]
    assert block_table.snapshot() == (
        pool.null_block_id,
        pool.null_block_id,
        block_ids[2],
    )
    assert tuple(pool.ref_count(block_id) for block_id in block_ids) == (0, 0, 1)


def test_chunked_local_releases_completed_chunks() -> None:
    manager, pool = make_manager(ChunkedLocalSpec(4))
    block_table, block_ids = make_allocated_table(pool, 4)

    assert manager.remove_skipped_blocks(block_table, 8) == block_ids
    assert block_table.snapshot() == (
        pool.null_block_id,
        pool.null_block_id,
        pool.null_block_id,
        pool.null_block_id,
    )
    assert tuple(pool.ref_count(block_id) for block_id in block_ids) == (0, 0, 0, 0)
