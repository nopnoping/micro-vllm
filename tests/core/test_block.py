"""Tests for logical KV-cache block primitives."""

import random

import pytest

from micro_vllm.core.block import BlockPool, BlockTable


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


def test_block_table_initializes_empty_zero_based_mapping() -> None:
    table = BlockTable(block_size=4)

    assert table.block_size == 4
    assert table.num_logical_blocks == 0
    assert table.capacity_tokens == 0
    assert table.snapshot() == ()


@pytest.mark.parametrize("block_size", [0, -1])
def test_block_table_rejects_non_positive_block_size(block_size: int) -> None:
    with pytest.raises(ValueError, match="block_size must be greater than zero"):
        BlockTable(block_size)


def test_block_table_appends_single_and_multiple_mappings_in_order() -> None:
    table = BlockTable(block_size=4)

    table.append(7)
    table.append(None)
    table.extend([11, 13])

    assert tuple(table.physical_block(index) for index in range(4)) == (
        7,
        None,
        11,
        13,
    )


def test_block_table_extend_is_atomic_for_invalid_mapping() -> None:
    table = BlockTable(block_size=4)
    table.append(7)

    with pytest.raises(
        ValueError, match="physical_block_id must be non-negative or None"
    ):
        table.extend([11, -1, 13])

    assert table.physical_block(0) == 7
    with pytest.raises(ValueError, match="invalid logical_block_index 1"):
        table.physical_block(1)


@pytest.mark.parametrize("logical_block_index", [-1, 2])
def test_physical_block_rejects_out_of_range_logical_index(
    logical_block_index: int,
) -> None:
    table = BlockTable(block_size=4)
    table.extend([7, 11])

    with pytest.raises(
        ValueError, match=rf"invalid logical_block_index {logical_block_index}"
    ):
        table.physical_block(logical_block_index)


@pytest.mark.parametrize(
    ("block_size", "token_position", "expected_slot"),
    [
        (1, 0, (7, 0)),
        (1, 1, (11, 0)),
        (4, 0, (7, 0)),
        (4, 3, (7, 3)),
        (4, 4, (11, 0)),
        (4, 7, (11, 3)),
        (8, 7, (7, 7)),
        (8, 8, (11, 0)),
    ],
)
def test_token_slot_uses_divmod_mapping(
    block_size: int,
    token_position: int,
    expected_slot: tuple[int, int],
) -> None:
    table = BlockTable(block_size)
    table.extend([7, 11])

    assert table.token_slot(token_position) == expected_slot


@pytest.mark.parametrize("token_position", [-1, 8])
def test_token_slot_rejects_position_outside_table_capacity(
    token_position: int,
) -> None:
    table = BlockTable(block_size=4)
    table.extend([7, 11])

    with pytest.raises(ValueError, match="token_position"):
        table.token_slot(token_position)


def test_unmapped_queries_consistently_return_none() -> None:
    table = BlockTable(block_size=4)
    table.extend([7, None])

    assert table.physical_block(1) is None
    assert table.token_slot(4) is None
    assert table.token_slot(7) is None


def test_block_table_snapshot_is_immutable_and_stable() -> None:
    mappings = [7, None]
    table = BlockTable(block_size=4)
    table.extend(mappings)
    snapshot = table.snapshot()

    mappings[0] = 99
    mappings.append(11)
    table.append(13)

    assert isinstance(snapshot, tuple)
    assert snapshot == (7, None)
    assert table.snapshot() == (7, None, 13)


def test_truncate_at_block_boundary_removes_tail_mappings() -> None:
    table = BlockTable(block_size=4)
    table.extend([7, 11, 13])

    removed_block_ids = table.truncate(8)

    assert removed_block_ids == (13,)
    assert table.snapshot() == (7, 11)
    assert table.num_logical_blocks == 2
    assert table.capacity_tokens == 8


def test_truncate_to_partial_block_keeps_covering_mapping() -> None:
    table = BlockTable(block_size=4)
    table.extend([7, 11, 13])

    removed_block_ids = table.truncate(5)

    assert removed_block_ids == (13,)
    assert table.snapshot() == (7, 11)
    assert table.capacity_tokens == 8


@pytest.mark.parametrize("num_tokens", [-1, 9])
def test_truncate_rejects_invalid_token_length(num_tokens: int) -> None:
    table = BlockTable(block_size=4)
    table.extend([7, 11])
    snapshot_before = table.snapshot()

    with pytest.raises(ValueError, match="num_tokens"):
        table.truncate(num_tokens)

    assert table.snapshot() == snapshot_before


def test_truncate_to_zero_is_equivalent_to_clear() -> None:
    truncated = BlockTable(block_size=4)
    truncated.extend([7, None, 13])
    cleared = BlockTable(block_size=4)
    cleared.extend([7, None, 13])

    assert truncated.truncate(0) == (7, 13)
    assert cleared.clear() == (7, 13)
    assert truncated.snapshot() == cleared.snapshot() == ()
    assert truncated.capacity_tokens == cleared.capacity_tokens == 0


def test_clear_returns_non_empty_mappings_in_logical_order() -> None:
    table = BlockTable(block_size=4)
    table.extend([7, None, 13, 17])

    assert table.clear() == (7, 13, 17)
    assert table.num_logical_blocks == 0
    assert table.capacity_tokens == 0


def test_block_table_operations_do_not_change_block_pool_refcounts() -> None:
    pool = BlockPool(3)
    assert pool.allocate(3) == (0, 1, 2)
    table = BlockTable(block_size=4)

    table.extend([0, None, 1, 2])
    assert table.token_slot(0) == (0, 0)
    assert table.truncate(8) == (1, 2)
    assert table.clear() == (0,)

    assert tuple(pool.ref_count(block_id) for block_id in range(3)) == (1, 1, 1)
    assert pool.num_free_blocks == 0
