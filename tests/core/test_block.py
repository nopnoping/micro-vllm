"""Tests for logical KV-cache block primitives."""

import random

import pytest

from micro_vllm.core.block import BlockPool


def test_block_pool_initial_state() -> None:
    pool = BlockPool(4)

    assert pool.num_blocks == 4
    assert pool.num_free_blocks == 4
    assert pool.free_block_ids == (0, 1, 2, 3)
    assert tuple(pool.ref_count(block_id) for block_id in range(4)) == (0, 0, 0, 0)


@pytest.mark.parametrize("num_blocks", [0, -1])
def test_block_pool_rejects_non_positive_capacity(num_blocks: int) -> None:
    with pytest.raises(ValueError, match="num_blocks must be greater than zero"):
        BlockPool(num_blocks)


def test_allocate_single_and_multiple_blocks_in_free_queue_order() -> None:
    pool = BlockPool(5)

    assert pool.allocate(1) == (0,)
    assert pool.allocate(2) == (1, 2)
    assert pool.free_block_ids == (3, 4)
    assert pool.num_free_blocks == 2
    assert tuple(pool.ref_count(block_id) for block_id in range(5)) == (
        1,
        1,
        1,
        0,
        0,
    )


def test_insufficient_allocation_is_atomic() -> None:
    pool = BlockPool(3)
    assert pool.allocate(2) == (0, 1)
    free_before = pool.free_block_ids
    refcounts_before = tuple(pool.ref_count(block_id) for block_id in range(3))

    assert pool.allocate(2) is None

    assert pool.free_block_ids == free_before
    assert tuple(pool.ref_count(block_id) for block_id in range(3)) == refcounts_before


@pytest.mark.parametrize("count", [0, -1])
def test_allocate_rejects_non_positive_count(count: int) -> None:
    pool = BlockPool(2)

    with pytest.raises(ValueError, match="count must be greater than zero"):
        pool.allocate(count)


def test_retain_and_release_shared_block() -> None:
    pool = BlockPool(2)
    assert pool.allocate(1) == (0,)

    pool.retain(0)
    assert pool.ref_count(0) == 2
    assert pool.free_block_ids == (1,)

    pool.release(0)
    assert pool.ref_count(0) == 1
    assert pool.free_block_ids == (1,)

    pool.release(0)
    assert pool.ref_count(0) == 0
    assert pool.free_block_ids == (1, 0)
    assert pool.allocate(2) == (1, 0)


def test_release_only_recycles_block_whose_refcount_reaches_zero() -> None:
    pool = BlockPool(3)
    assert pool.allocate(2) == (0, 1)
    pool.retain(0)

    pool.release(0)
    pool.release(1)

    assert pool.ref_count(0) == 1
    assert pool.ref_count(1) == 0
    assert pool.free_block_ids == (2, 1)


@pytest.mark.parametrize("block_id", [-1, 3])
@pytest.mark.parametrize("operation", ["ref_count", "retain", "release"])
def test_block_pool_rejects_invalid_block_id(
    operation: str, block_id: int
) -> None:
    pool = BlockPool(3)

    with pytest.raises(ValueError, match=rf"invalid block_id {block_id}"):
        getattr(pool, operation)(block_id)


def test_retain_rejects_free_block() -> None:
    pool = BlockPool(2)

    with pytest.raises(RuntimeError, match="cannot retain block_id 0: block is free"):
        pool.retain(0)


def test_release_rejects_double_free_without_duplicating_queue_entry() -> None:
    pool = BlockPool(2)
    assert pool.allocate(1) == (0,)
    pool.release(0)
    free_before = pool.free_block_ids

    with pytest.raises(RuntimeError, match="cannot release block_id 0: block is free"):
        pool.release(0)

    assert pool.free_block_ids == free_before
    assert pool.free_block_ids.count(0) == 1


def test_free_block_snapshot_is_immutable_and_stable() -> None:
    pool = BlockPool(3)
    snapshot = pool.free_block_ids

    assert isinstance(snapshot, tuple)
    assert pool.allocate(1) == (0,)
    assert snapshot == (0, 1, 2)
    assert pool.free_block_ids == (1, 2)


def test_deterministic_randomized_operations_preserve_pool_invariants() -> None:
    rng = random.Random(0)
    num_blocks = 8
    pool = BlockPool(num_blocks)
    expected_free = list(range(num_blocks))
    expected_refcounts = [0] * num_blocks

    for _ in range(500):
        operation = rng.choice(("allocate", "retain", "release"))
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
        elif operation == "retain" and allocated_ids:
            block_id = rng.choice(allocated_ids)
            pool.retain(block_id)
            expected_refcounts[block_id] += 1
        elif operation == "release" and allocated_ids:
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
        assert num_allocated_blocks + pool.num_free_blocks == pool.num_blocks
