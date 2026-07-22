"""Tests for logical KV-cache block primitives."""

import random

from micro_vllm.core.block import BlockPool, BlockTable, KVCacheBlock

NULL_BLOCK_ID = 99


def test_kv_cache_block_bundles_physical_lifecycle_state() -> None:
    block = KVCacheBlock(block_id=7)

    assert block.block_id == 7
    assert block.ref_count == 0
    assert block.block_hash is None


def test_block_pool_initial_state() -> None:
    pool = BlockPool(4)

    assert pool.num_blocks == 4
    assert pool.num_free_blocks == 3
    assert pool.null_block_id == 3
    assert pool.num_cached_blocks == 0
    assert pool.free_block_ids == (0, 1, 2)
    assert tuple(pool.ref_count(block_id) for block_id in range(4)) == (0, 0, 0, 0)
    assert tuple(pool.block_hash(block_id) for block_id in range(4)) == (
        None,
        None,
        None,
        None,
    )


def test_allocate_single_and_multiple_blocks_in_free_queue_order() -> None:
    pool = BlockPool(5)

    assert pool.allocate(1) == (0,)
    assert pool.allocate(2) == (1, 2)
    assert pool.free_block_ids == (3,)
    assert pool.num_free_blocks == 1
    assert tuple(pool.ref_count(block_id) for block_id in range(5)) == (
        1,
        1,
        1,
        0,
        0,
    )


def test_null_block_never_enters_the_allocatable_lifecycle() -> None:
    pool = BlockPool(3)

    assert pool.allocate(2) == (0, 1)
    pool.release(0)
    pool.release(1)

    assert pool.free_block_ids == (0, 1)
    assert pool.allocate(2) == (0, 1)
    assert pool.null_block_id == 2
    assert pool.ref_count(pool.null_block_id) == 0
    assert pool.block_hash(pool.null_block_id) is None


def test_insufficient_allocation_is_atomic() -> None:
    pool = BlockPool(3)
    assert pool.allocate(2) == (0, 1)
    free_before = pool.free_block_ids
    refcounts_before = tuple(pool.ref_count(block_id) for block_id in range(3))

    assert pool.allocate(2) is None

    assert pool.free_block_ids == free_before
    assert tuple(pool.ref_count(block_id) for block_id in range(3)) == refcounts_before


def test_allocate_retains_cached_blocks_and_allocates_new_blocks_atomically() -> None:
    pool = BlockPool(4)
    assert pool.allocate(2) == (0, 1)
    pool.cache_block(0, b"a" * 32)
    pool.cache_block(1, b"b" * 32)
    pool.release(0)

    assert pool.allocate(1, (0, 1)) == (2,)

    assert pool.free_block_ids == ()
    assert tuple(pool.ref_count(block_id) for block_id in range(4)) == (1, 2, 1, 0)
    assert pool.get_cached_block(b"a" * 32) is not None
    assert pool.get_cached_block(b"b" * 32) is not None


def test_cached_free_blocks_count_towards_atomic_allocation_capacity() -> None:
    pool = BlockPool(3)
    assert pool.allocate(2) == (0, 1)
    pool.cache_block(0, b"a" * 32)
    pool.cache_block(1, b"b" * 32)
    pool.release(0)
    pool.release(1)
    free_before = pool.free_block_ids
    hashes_before = tuple(pool.block_hash(block_id) for block_id in range(3))

    assert pool.allocate(1, (0, 1)) is None

    assert pool.free_block_ids == free_before
    assert tuple(pool.ref_count(block_id) for block_id in range(3)) == (0, 0, 0)
    assert tuple(pool.block_hash(block_id) for block_id in range(3)) == hashes_before


def test_capacity_preflight_has_no_side_effects() -> None:
    pool = BlockPool(4)
    assert pool.allocate(2) == (0, 1)
    pool.cache_block(0, b"a" * 32)
    pool.release(0)
    free_before = pool.free_block_ids
    refcounts_before = tuple(pool.ref_count(block_id) for block_id in range(4))
    hashes_before = tuple(pool.block_hash(block_id) for block_id in range(4))

    assert pool.can_allocate(1, (0,)) is True
    assert pool.can_allocate(2, (0,)) is False

    assert pool.free_block_ids == free_before
    assert tuple(pool.ref_count(block_id) for block_id in range(4)) == refcounts_before
    assert tuple(pool.block_hash(block_id) for block_id in range(4)) == hashes_before


def test_allocate_allows_zero_new_blocks() -> None:
    assert BlockPool(2).allocate(0) == ()


def test_cached_block_remains_indexed_until_free_block_is_reused() -> None:
    pool = BlockPool(2)
    block_hash = b"a" * 32
    assert pool.allocate(1) == (0,)

    block = pool.cache_block(0, block_hash)

    assert pool.ref_count(0) == 1
    assert pool.num_cached_blocks == 1
    assert pool.block_hash(0) == block_hash
    assert pool.get_cached_block(block_hash) is block

    pool.release(0)

    assert pool.ref_count(0) == 0
    assert pool.free_block_ids == (0,)
    assert pool.get_cached_block(block_hash) is block

    assert pool.acquire_cached_block(block_hash) is block
    assert pool.ref_count(0) == 1
    assert pool.free_block_ids == ()

    pool.release(0)
    assert pool.allocate(1) == (0,)
    assert pool.get_cached_block(block_hash) is None
    assert pool.num_cached_blocks == 0
    assert pool.block_hash(0) is None


def test_cache_block_is_idempotent_and_updates_identity() -> None:
    pool = BlockPool(2)
    first_hash = b"a" * 32
    second_hash = b"b" * 32
    assert pool.allocate(1) == (0,)

    first = pool.cache_block(0, first_hash)
    assert pool.cache_block(0, first_hash) is first
    assert pool.ref_count(0) == 1

    assert pool.cache_block(0, second_hash) is first
    assert pool.get_cached_block(first_hash) is None
    assert pool.get_cached_block(second_hash) is first


def test_acquire_cached_active_block_adds_request_reference() -> None:
    pool = BlockPool(2)
    block_hash = b"a" * 32
    assert pool.allocate(1) == (0,)
    block = pool.cache_block(0, block_hash)

    assert pool.acquire_cached_block(block_hash) is block
    assert pool.ref_count(0) == 2
    assert pool.num_free_blocks == 0


def test_free_block_snapshot_is_immutable_and_stable() -> None:
    pool = BlockPool(4)
    snapshot = pool.free_block_ids

    assert isinstance(snapshot, tuple)
    assert pool.allocate(1) == (0,)
    assert snapshot == (0, 1, 2)
    assert pool.free_block_ids == (1, 2)


def test_deterministic_randomized_operations_preserve_pool_invariants() -> None:
    rng = random.Random(0)
    num_blocks = 8
    pool = BlockPool(num_blocks)
    expected_free = list(range(num_blocks - 1))
    expected_refcounts = [0] * num_blocks

    for _ in range(500):
        operation = rng.choice(("allocate", "release"))
        allocated_ids = [
            block_id
            for block_id, refcount in enumerate(expected_refcounts)
            if refcount > 0
        ]

        if operation == "allocate":
            count = rng.randint(1, 3)
            allocated = pool.allocate(count)
            if count > len(expected_free):
                assert allocated is None
            else:
                expected = tuple(expected_free[:count])
                del expected_free[:count]
                assert allocated == expected
                for block_id in expected:
                    expected_refcounts[block_id] = 1
        elif allocated_ids:
            block_id = rng.choice(allocated_ids)
            pool.release(block_id)
            expected_refcounts[block_id] -= 1
            if expected_refcounts[block_id] == 0:
                expected_free.append(block_id)

        actual_refcounts = [
            pool.ref_count(block_id) for block_id in range(num_blocks)
        ]
        num_allocated_blocks = sum(refcount > 0 for refcount in actual_refcounts)
        assert actual_refcounts == expected_refcounts
        assert pool.free_block_ids == tuple(expected_free)
        assert len(set(pool.free_block_ids)) == pool.num_free_blocks
        assert num_allocated_blocks + pool.num_free_blocks + 1 == pool.num_blocks
        assert pool.ref_count(pool.null_block_id) == 0
        assert pool.block_hash(pool.null_block_id) is None


def test_block_table_initializes_empty_zero_based_mapping() -> None:
    table = BlockTable(NULL_BLOCK_ID)

    assert table.num_logical_blocks == 0
    assert table.snapshot() == ()


def test_block_table_extends_mappings_in_order() -> None:
    table = BlockTable(NULL_BLOCK_ID)

    table.extend([7, NULL_BLOCK_ID, 11, 13])

    assert tuple(table.physical_block(index) for index in range(4)) == (
        7,
        NULL_BLOCK_ID,
        11,
        13,
    )


def test_replace_updates_one_mapping_and_returns_previous_block() -> None:
    table = BlockTable(NULL_BLOCK_ID)
    table.extend([7, 11])

    assert table.replace(1, 13) == 11
    assert table.snapshot() == (7, 13)


def test_unmapped_physical_block_uses_null_block() -> None:
    table = BlockTable(NULL_BLOCK_ID)
    table.extend([7, NULL_BLOCK_ID])

    assert table.physical_block(1) == NULL_BLOCK_ID


def test_fill_missing_uses_absolute_logical_indices() -> None:
    table = BlockTable(NULL_BLOCK_ID)
    table.extend([7, NULL_BLOCK_ID])

    table.fill_missing((1, 3), (11, 13))

    assert table.snapshot() == (7, 11, NULL_BLOCK_ID, 13)


def test_release_before_preserves_absolute_logical_indices() -> None:
    table = BlockTable(NULL_BLOCK_ID)
    table.extend([7, 11, NULL_BLOCK_ID, 13, 17])

    assert table.release_before(3) == (7, 11)
    assert table.snapshot() == (
        NULL_BLOCK_ID,
        NULL_BLOCK_ID,
        NULL_BLOCK_ID,
        13,
        17,
    )
    assert table.release_before(4) == (13,)
    assert table.snapshot() == (
        NULL_BLOCK_ID,
        NULL_BLOCK_ID,
        NULL_BLOCK_ID,
        NULL_BLOCK_ID,
        17,
    )


def test_block_table_snapshot_is_immutable_and_stable() -> None:
    mappings = [7, NULL_BLOCK_ID]
    table = BlockTable(NULL_BLOCK_ID)
    table.extend(mappings)
    snapshot = table.snapshot()

    mappings[0] = 99
    mappings.append(11)
    table.extend((13,))

    assert isinstance(snapshot, tuple)
    assert snapshot == (7, NULL_BLOCK_ID)
    assert table.snapshot() == (7, NULL_BLOCK_ID, 13)


def test_clear_returns_non_empty_mappings_in_logical_order() -> None:
    table = BlockTable(NULL_BLOCK_ID)
    table.extend([7, NULL_BLOCK_ID, 13, 17])

    assert table.clear() == (7, 13, 17)
    assert table.num_logical_blocks == 0


def test_block_table_operations_do_not_change_block_pool_refcounts() -> None:
    pool = BlockPool(4)
    assert pool.allocate(3) == (0, 1, 2)
    table = BlockTable(pool.null_block_id)

    table.extend([0, pool.null_block_id, 1, 2])
    assert table.clear() == (0, 1, 2)

    assert tuple(pool.ref_count(block_id) for block_id in range(4)) == (1, 1, 1, 0)
    assert pool.num_free_blocks == 0
