"""Tests for multi-group logical KV-cache coordination."""

from dataclasses import FrozenInstanceError

import pytest

from micro_vllm.attention.spec import (
    AttentionSpec,
    ChunkedLocalSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
)
from micro_vllm.config import CacheConfig, LLMConfig, ModelConfig
from micro_vllm.core.block import BlockTable
from micro_vllm.core.kv_cache_manager import (
    KVCacheAllocation,
    KVCacheManager,
)
from micro_vllm.core.kv_cache_spec import (
    KVCacheLayout,
    build_kv_cache_topology,
)
from micro_vllm.core.scheduler_output import CacheGroupBlockTable


def make_manager(
    block_size: int = 4,
    num_blocks: int = 9,
    *,
    enable_prefix_cache: bool = False,
    attention_specs: tuple[AttentionSpec, ...] = (FullAttentionSpec(),),
    cache_namespace: str = "model-a",
) -> KVCacheManager:
    config = LLMConfig(
        model_config=ModelConfig(model="unused"),
        cache_config=CacheConfig(
            block_size=block_size,
            num_blocks=num_blocks,
            enable_prefix_cache=enable_prefix_cache,
        ),
    )
    layout = KVCacheLayout(
        block_size=block_size,
        num_blocks=num_blocks,
        topology=build_kv_cache_topology(attention_specs),
    )
    return KVCacheManager(
        config,
        layout,
        cache_namespace=cache_namespace,
    )


def cache_group_blocks(
    manager: KVCacheManager,
    group_index: int,
    token_ids: tuple[int, ...],
    logical_block_indices: tuple[int, ...],
) -> tuple[int, ...]:
    block_ids = manager._block_pool.allocate(len(logical_block_indices))
    assert block_ids is not None
    block_table = BlockTable(manager._block_pool.null_block_id)
    block_table.fill_missing(logical_block_indices, block_ids)
    manager._group_managers[group_index].cache_computed_blocks(
        block_table,
        token_ids,
        (max(logical_block_indices) + 1) * manager._layout.block_size,
    )
    cached_block_ids = tuple(
        block_table.physical_block(index)
        for index in logical_block_indices
    )
    for block_id in block_table.clear():
        manager._block_pool.release(block_id)
    return cached_block_ids


def test_manager_builds_group_policies_over_one_shared_pool() -> None:
    manager = make_manager(
        attention_specs=(
            FullAttentionSpec(),
            SlidingWindowSpec(4),
            ChunkedLocalSpec(4),
        )
    )

    assert len(manager._group_managers) == 3
    assert all(
        group_manager._block_pool is manager._block_pool
        for group_manager in manager._group_managers
    )
    assert manager._block_pool.num_blocks == 9
    assert manager._block_pool.num_free_blocks == 8
    assert manager._block_pool.null_block_id == 8


def test_register_request_creates_one_isolated_table_per_group() -> None:
    manager = make_manager(
        attention_specs=(FullAttentionSpec(), SlidingWindowSpec(4))
    )

    manager.register_request("request-1")
    manager.register_request("request-2")

    assert manager.get_block_tables("request-1") == ((), ())
    assert manager.get_block_tables("request-2") == ((), ())
    first_tables = manager._block_tables["request-1"]
    second_tables = manager._block_tables["request-2"]
    assert all(
        first is not second
        for first, second in zip(first_tables, second_tables, strict=True)
    )


def test_default_layout_preserves_the_full_only_path() -> None:
    config = LLMConfig(
        model_config=ModelConfig(model="unused"),
        cache_config=CacheConfig(block_size=2, num_blocks=5),
    )
    manager = KVCacheManager(config)
    manager.register_request("request-1")

    allocation = manager.allocate_slots("request-1", 0, 3)

    assert allocation == KVCacheAllocation(
        (CacheGroupBlockTable((0, 1)),),
        (0, 1),
    )
    assert manager.get_block_table("request-1") == (0, 1)


def test_common_prefix_reaches_a_fixed_point_across_groups() -> None:
    manager = make_manager(
        block_size=2,
        num_blocks=12,
        enable_prefix_cache=True,
        attention_specs=(FullAttentionSpec(), SlidingWindowSpec(3)),
    )
    token_ids = (1, 2, 3, 4, 5, 6, 7)
    cache_group_blocks(manager, 0, token_ids, (0, 1))
    cache_group_blocks(manager, 1, token_ids, (2,))
    manager.register_request("request-1")

    hit = manager.find_longest_cache_hit("request-1", token_ids)

    assert hit.num_computed_tokens == 0
    assert manager.get_block_tables("request-1") == ((), ())


def test_common_hit_restores_each_group_snapshot_before_allocation() -> None:
    manager = make_manager(
        block_size=2,
        num_blocks=12,
        enable_prefix_cache=True,
        attention_specs=(FullAttentionSpec(), SlidingWindowSpec(3)),
    )
    token_ids = (1, 2, 3, 4, 5, 6, 7)
    full_block_ids = cache_group_blocks(manager, 0, token_ids, (0, 1))
    (sliding_block_id,) = cache_group_blocks(manager, 1, token_ids, (1,))
    manager.register_request("request-1")

    hit = manager.find_longest_cache_hit("request-1", token_ids)
    allocation = manager.allocate_slots(
        "request-1",
        hit.num_computed_tokens,
        1,
        cache_hit=hit,
    )

    assert hit.num_computed_tokens == 4
    assert allocation is not None
    assert allocation.block_tables[0].block_ids[:2] == full_block_ids
    assert allocation.block_tables[1].block_ids[:2] == (
        manager._block_pool.null_block_id,
        sliding_block_id,
    )
    assert len(allocation.new_block_ids) == 2
    assert (
        allocation.block_tables[0].block_ids[2]
        != allocation.block_tables[1].block_ids[2]
    )


def test_prefix_discovery_does_not_acquire_blocks() -> None:
    manager = make_manager(
        block_size=2,
        num_blocks=6,
        enable_prefix_cache=True,
    )
    token_ids = (1, 2, 3, 4, 5)
    cached_block_ids = cache_group_blocks(manager, 0, token_ids, (0, 1))
    manager.register_request("request-1")

    hit = manager.find_longest_cache_hit("request-1", token_ids)

    assert hit.num_computed_tokens == 4
    assert tuple(
        manager._block_pool.ref_count(block_id)
        for block_id in cached_block_ids
    ) == (0, 0)
    assert manager.get_block_table("request-1") == ()


def test_prefix_hit_keeps_one_known_token_for_model_computation() -> None:
    manager = make_manager(
        block_size=2,
        enable_prefix_cache=True,
    )
    token_ids = (1, 2, 3, 4)
    cache_group_blocks(manager, 0, token_ids, (0, 1))
    manager.register_request("request-1")

    hit = manager.find_longest_cache_hit("request-1", token_ids)

    assert hit.num_computed_tokens == 2


def test_chunked_group_can_skip_completed_chunks_without_prefix_cache() -> None:
    manager = make_manager(
        block_size=2,
        attention_specs=(ChunkedLocalSpec(4),),
    )
    manager.register_request("request-1")

    hit = manager.find_longest_cache_hit(
        "request-1",
        (1, 2, 3, 4, 5, 6, 7),
    )
    allocation = manager.allocate_slots(
        "request-1",
        hit.num_computed_tokens,
        2,
        cache_hit=hit,
    )

    assert hit.num_computed_tokens == 4
    assert allocation is not None
    assert allocation.block_tables[0].block_ids[:2] == (
        manager._block_pool.null_block_id,
        manager._block_pool.null_block_id,
    )
    assert len(allocation.new_block_ids) == 1


def test_multi_group_capacity_failure_is_atomic() -> None:
    manager = make_manager(
        block_size=2,
        num_blocks=2,
        attention_specs=(FullAttentionSpec(), SlidingWindowSpec(3)),
    )
    manager.register_request("request-1")
    free_blocks_before = manager._block_pool.free_block_ids

    allocation = manager.allocate_slots("request-1", 0, 1)

    assert allocation is None
    assert manager.get_block_tables("request-1") == ((), ())
    assert manager._block_pool.free_block_ids == free_blocks_before
    assert manager._block_pool.ref_count(0) == 0


def test_cached_acquire_and_new_blocks_share_one_atomic_preflight() -> None:
    manager = make_manager(
        block_size=2,
        num_blocks=4,
        enable_prefix_cache=True,
        attention_specs=(FullAttentionSpec(), SlidingWindowSpec(3)),
    )
    token_ids = (1, 2, 3)
    full_block_id = cache_group_blocks(manager, 0, token_ids, (0,))[0]
    sliding_block_id = cache_group_blocks(manager, 1, token_ids, (0,))[0]
    manager.register_request("request-1")
    hit = manager.find_longest_cache_hit("request-1", token_ids)
    free_blocks_before = manager._block_pool.free_block_ids

    allocation = manager.allocate_slots(
        "request-1",
        hit.num_computed_tokens,
        1,
        cache_hit=hit,
    )

    assert hit.num_computed_tokens == 2
    assert allocation is None
    assert manager.get_block_tables("request-1") == ((), ())
    assert manager._block_pool.free_block_ids == free_blocks_before
    assert manager._block_pool.ref_count(full_block_id) == 0
    assert manager._block_pool.ref_count(sliding_block_id) == 0


def test_new_block_count_is_aggregated_across_groups() -> None:
    manager = make_manager(
        block_size=4,
        attention_specs=(FullAttentionSpec(), SlidingWindowSpec(4)),
    )
    manager.register_request("request-1")

    assert manager.get_num_blocks_to_allocate("request-1", 0, 5) == 4
    assert manager.get_num_blocks_to_allocate("request-1", 0, 0) == 0
    assert manager.get_block_tables("request-1") == ((), ())


def test_running_allocation_fills_only_scheduled_holes() -> None:
    manager = make_manager(
        block_size=2,
        attention_specs=(SlidingWindowSpec(3),),
    )
    manager.register_request("request-1")
    first = manager.allocate_slots("request-1", 0, 6)
    assert first is not None
    manager.commit_step("request-1", (1, 2, 3, 4, 5, 6), 6)

    second = manager.allocate_slots("request-1", 6, 2)

    assert second is not None
    assert len(second.new_block_ids) == 1
    assert second.block_tables[0].block_ids[:2] == (
        manager._block_pool.null_block_id,
        manager._block_pool.null_block_id,
    )


def test_commit_step_caches_only_complete_full_blocks() -> None:
    manager = make_manager(
        block_size=2,
        num_blocks=5,
        enable_prefix_cache=True,
    )
    manager.register_request("request-1")
    assert manager.allocate_slots("request-1", 0, 5) is not None

    manager.commit_step("request-1", (1, 2, 3, 4, 5), 5)
    first_table = manager.get_block_table("request-1")
    manager.commit_step("request-1", (1, 2, 3, 4, 5), 5)

    assert manager.get_block_table("request-1") == first_table == (0, 1, 2)
    assert manager._block_pool.num_cached_blocks == 2
    assert manager._block_pool.block_hash(0) is not None
    assert manager._block_pool.block_hash(1) is not None
    assert manager._block_pool.block_hash(2) is None


def test_commit_step_reuses_a_canonical_group_block() -> None:
    manager = make_manager(
        block_size=2,
        num_blocks=3,
        enable_prefix_cache=True,
    )
    manager.register_request("request-1")
    assert manager.allocate_slots("request-1", 0, 2) is not None
    manager.commit_step("request-1", (1, 2), 2)
    manager.free_request("request-1")

    manager.register_request("request-2")
    allocation = manager.allocate_slots("request-2", 0, 2)
    assert allocation == KVCacheAllocation(
        (CacheGroupBlockTable((1,)),),
        (1,),
    )
    manager.commit_step("request-2", (1, 2), 2)

    assert manager.get_block_table("request-2") == (0,)
    assert manager._block_pool.ref_count(0) == 1
    assert manager._block_pool.ref_count(1) == 0
    assert manager._block_pool.num_cached_blocks == 1


def test_commit_step_releases_only_the_bounded_group_history() -> None:
    manager = make_manager(
        block_size=2,
        num_blocks=8,
        attention_specs=(FullAttentionSpec(), SlidingWindowSpec(3)),
    )
    manager.register_request("request-1")
    allocation = manager.allocate_slots("request-1", 0, 6)
    assert allocation is not None
    full_table, sliding_table = allocation.block_tables
    full_block_ids = full_table.block_ids
    sliding_block_ids = sliding_table.block_ids

    manager.commit_step("request-1", (1, 2, 3, 4, 5, 6), 6)

    assert manager.get_block_tables("request-1") == (
        full_block_ids,
        (
            manager._block_pool.null_block_id,
            manager._block_pool.null_block_id,
            sliding_block_ids[2],
        ),
    )
    assert tuple(
        manager._block_pool.ref_count(block_id)
        for block_id in full_block_ids
    ) == (1, 1, 1)
    assert tuple(
        manager._block_pool.ref_count(block_id)
        for block_id in sliding_block_ids
    ) == (0, 0, 1)


def test_free_request_blocks_releases_every_group_for_recompute() -> None:
    manager = make_manager(
        block_size=2,
        num_blocks=6,
        attention_specs=(FullAttentionSpec(), SlidingWindowSpec(3)),
    )
    manager.register_request("request-1")
    assert manager.allocate_slots("request-1", 0, 4) is not None

    manager.free_request_blocks("request-1")

    assert manager.get_block_tables("request-1") == ((), ())
    assert manager.num_free_blocks == 5
    assert manager.allocate_slots("request-1", 0, 1) is not None


def test_free_request_removes_all_group_tables() -> None:
    manager = make_manager(
        attention_specs=(FullAttentionSpec(), ChunkedLocalSpec(4))
    )
    manager.register_request("request-1")
    assert manager.allocate_slots("request-1", 0, 1) is not None

    manager.free_request("request-1")

    with pytest.raises(KeyError):
        manager.get_block_tables("request-1")


@pytest.mark.parametrize(
    "operation",
    ["get_block_tables", "find_longest_cache_hit", "free_request"],
)
def test_unknown_request_operations_raise_key_error(operation: str) -> None:
    manager = make_manager()

    with pytest.raises(KeyError):
        if operation == "find_longest_cache_hit":
            manager.find_longest_cache_hit("missing", (1,))
        else:
            getattr(manager, operation)("missing")


def test_allocation_snapshot_is_frozen_and_stable() -> None:
    manager = make_manager()
    manager.register_request("request-1")
    first = manager.allocate_slots("request-1", 0, 1)
    assert first is not None

    assert manager.allocate_slots("request-1", 4, 1) is not None

    assert first.block_tables == (CacheGroupBlockTable((0,)),)
    assert first.new_block_ids == (0,)
    assert manager.get_block_table("request-1") == (0, 1)
    with pytest.raises(FrozenInstanceError):
        first.block_tables = ()
