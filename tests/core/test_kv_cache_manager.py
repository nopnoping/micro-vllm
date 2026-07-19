"""Tests for logical request KV-cache lifecycle management."""

from dataclasses import FrozenInstanceError

import pytest

from micro_vllm.config import CacheConfig
from micro_vllm.core.kv_cache_manager import KVCacheAllocation, KVCacheManager


def make_manager(block_size: int = 4, num_blocks: int = 8) -> KVCacheManager:
    return KVCacheManager(
        CacheConfig(
            block_size=block_size,
            num_blocks=num_blocks,
            enable_prefix_cache=False,
        )
    )


def test_manager_owns_one_pool_and_empty_table_registry() -> None:
    manager = make_manager()

    assert manager._block_pool.num_blocks == 8
    assert manager._block_pool.num_free_blocks == 8
    assert manager._block_tables == {}


def test_register_request_creates_isolated_empty_block_tables() -> None:
    manager = make_manager()

    manager.register_request("request-1")
    manager.register_request("request-2")

    assert manager.get_block_table("request-1") == ()
    assert manager.get_block_table("request-2") == ()
    assert manager._block_tables["request-1"] is not manager._block_tables["request-2"]


def test_register_request_rejects_empty_id() -> None:
    manager = make_manager()

    with pytest.raises(ValueError, match="request_id must not be empty"):
        manager.register_request("")


def test_register_request_rejects_duplicate_id_without_replacing_table() -> None:
    manager = make_manager()
    manager.register_request("request-1")
    original_table = manager._block_tables["request-1"]

    with pytest.raises(ValueError, match="duplicate request_id: request-1"):
        manager.register_request("request-1")

    assert manager._block_tables["request-1"] is original_table


@pytest.mark.parametrize("operation", ["get_block_table", "free_request"])
def test_unknown_request_operations_raise_key_error(operation: str) -> None:
    manager = make_manager()

    with pytest.raises(KeyError, match="unknown request_id: missing"):
        getattr(manager, operation)("missing")


def test_free_registered_request_removes_its_table() -> None:
    manager = make_manager()
    manager.register_request("request-1")

    manager.free_request("request-1")

    with pytest.raises(KeyError, match="unknown request_id: request-1"):
        manager.get_block_table("request-1")
    manager.register_request("request-1")
    assert manager.get_block_table("request-1") == ()


@pytest.mark.parametrize(
    ("num_computed_tokens", "num_scheduled_tokens", "expected_blocks"),
    [
        (0, 0, 0),
        (0, 1, 1),
        (0, 4, 1),
        (0, 5, 2),
        (3, 2, 2),
        (4, 4, 2),
        (4, 5, 3),
    ],
)
def test_num_blocks_to_allocate_uses_ceil_div_of_target_tokens(
    num_computed_tokens: int,
    num_scheduled_tokens: int,
    expected_blocks: int,
) -> None:
    manager = make_manager()
    manager.register_request("request-1")

    assert (
        manager.get_num_blocks_to_allocate(
            "request-1",
            num_computed_tokens,
            num_scheduled_tokens,
        )
        == expected_blocks
    )


@pytest.mark.parametrize(
    ("num_computed_tokens", "num_scheduled_tokens", "expected_blocks"),
    [
        (1, 1, 0),
        (3, 1, 0),
        (4, 1, 1),
        (4, 4, 1),
        (3, 6, 2),
    ],
)
def test_num_blocks_to_allocate_only_counts_blocks_missing_from_table(
    num_computed_tokens: int,
    num_scheduled_tokens: int,
    expected_blocks: int,
) -> None:
    manager = make_manager()
    manager.register_request("request-1")
    manager._block_tables["request-1"].append(0)

    assert (
        manager.get_num_blocks_to_allocate(
            "request-1",
            num_computed_tokens,
            num_scheduled_tokens,
        )
        == expected_blocks
    )


def test_num_blocks_to_allocate_is_a_read_only_calculation() -> None:
    manager = make_manager()
    manager.register_request("request-1")
    manager._block_tables["request-1"].append(0)
    table_before = manager.get_block_table("request-1")
    num_free_blocks_before = manager._block_pool.num_free_blocks

    assert manager.get_num_blocks_to_allocate("request-1", 3, 6) == 2

    assert manager.get_block_table("request-1") == table_before
    assert manager._block_pool.num_free_blocks == num_free_blocks_before


@pytest.mark.parametrize(
    ("num_computed_tokens", "num_scheduled_tokens", "message"),
    [
        (-1, 1, "num_computed_tokens must be non-negative"),
        (0, -1, "num_scheduled_tokens must be non-negative"),
    ],
)
def test_num_blocks_to_allocate_rejects_negative_token_counts(
    num_computed_tokens: int,
    num_scheduled_tokens: int,
    message: str,
) -> None:
    manager = make_manager()
    manager.register_request("request-1")

    with pytest.raises(ValueError, match=message):
        manager.get_num_blocks_to_allocate(
            "request-1",
            num_computed_tokens,
            num_scheduled_tokens,
        )


def test_allocate_slots_reserves_first_partial_block() -> None:
    manager = make_manager()
    manager.register_request("request-1")

    allocation = manager.allocate_slots("request-1", 0, 1)

    assert allocation == KVCacheAllocation(
        block_table=(0,),
        new_block_ids=(0,),
    )
    assert manager.get_block_table("request-1") == (0,)
    assert manager._block_pool.ref_count(0) == 1


def test_allocate_slots_reuses_capacity_in_partial_block() -> None:
    manager = make_manager()
    manager.register_request("request-1")
    assert manager.allocate_slots("request-1", 0, 1) is not None

    allocation = manager.allocate_slots("request-1", 1, 2)

    assert allocation == KVCacheAllocation(
        block_table=(0,),
        new_block_ids=(),
    )
    assert manager._block_pool.num_free_blocks == 7


def test_allocate_slots_crosses_one_block_boundary() -> None:
    manager = make_manager()
    manager.register_request("request-1")
    assert manager.allocate_slots("request-1", 0, 3) is not None

    allocation = manager.allocate_slots("request-1", 3, 2)

    assert allocation == KVCacheAllocation(
        block_table=(0, 1),
        new_block_ids=(1,),
    )


def test_allocate_slots_crosses_multiple_block_boundaries() -> None:
    manager = make_manager()
    manager.register_request("request-1")

    allocation = manager.allocate_slots("request-1", 0, 9)

    assert allocation == KVCacheAllocation(
        block_table=(0, 1, 2),
        new_block_ids=(0, 1, 2),
    )


def test_allocate_slots_isolates_request_tables_while_sharing_pool_capacity() -> None:
    manager = make_manager(num_blocks=4)
    manager.register_request("request-1")
    manager.register_request("request-2")

    first = manager.allocate_slots("request-1", 0, 5)
    second = manager.allocate_slots("request-2", 0, 4)

    assert first == KVCacheAllocation((0, 1), (0, 1))
    assert second == KVCacheAllocation((2,), (2,))
    assert manager.get_block_table("request-1") == (0, 1)
    assert manager.get_block_table("request-2") == (2,)
    assert manager._block_pool.num_free_blocks == 1


def test_allocate_slots_capacity_failure_is_atomic() -> None:
    manager = make_manager(num_blocks=2)
    manager.register_request("request-1")
    manager.register_request("request-2")
    assert manager.allocate_slots("request-1", 0, 4) is not None
    first_table_before = manager.get_block_table("request-1")
    second_table_before = manager.get_block_table("request-2")
    free_blocks_before = manager._block_pool.free_block_ids

    allocation = manager.allocate_slots("request-2", 0, 5)

    assert allocation is None
    assert manager.get_block_table("request-1") == first_table_before
    assert manager.get_block_table("request-2") == second_table_before
    assert manager._block_pool.free_block_ids == free_blocks_before
    assert tuple(manager._block_pool.ref_count(block_id) for block_id in range(2)) == (
        1,
        0,
    )


def test_allocation_snapshot_does_not_change_after_later_allocation() -> None:
    manager = make_manager()
    manager.register_request("request-1")
    first = manager.allocate_slots("request-1", 0, 1)
    assert first is not None

    assert manager.allocate_slots("request-1", 4, 1) is not None

    assert first.block_table == (0,)
    assert first.new_block_ids == (0,)
    assert manager.get_block_table("request-1") == (0, 1)


def test_kv_cache_allocation_is_frozen_and_canonicalizes_sequences() -> None:
    allocation = KVCacheAllocation(
        block_table=[0, None],  # type: ignore[arg-type]
        new_block_ids=[0],  # type: ignore[arg-type]
    )

    assert allocation.block_table == (0, None)
    assert allocation.new_block_ids == (0,)
    with pytest.raises(FrozenInstanceError):
        allocation.block_table = ()


def test_free_request_recycles_all_request_owned_blocks() -> None:
    manager = make_manager(num_blocks=4)
    manager.register_request("request-1")
    assert manager.allocate_slots("request-1", 0, 9) is not None

    manager.free_request("request-1")

    assert manager._block_pool.num_free_blocks == 4
    assert manager._block_pool.free_block_ids == (3, 0, 1, 2)
    assert tuple(manager._block_pool.ref_count(block_id) for block_id in range(4)) == (
        0,
        0,
        0,
        0,
    )
    with pytest.raises(KeyError, match="unknown request_id: request-1"):
        manager.get_block_table("request-1")


def test_free_request_does_not_affect_other_request() -> None:
    manager = make_manager(num_blocks=4)
    manager.register_request("request-1")
    manager.register_request("request-2")
    assert manager.allocate_slots("request-1", 0, 5) is not None
    assert manager.allocate_slots("request-2", 0, 4) is not None
    second_table_before = manager.get_block_table("request-2")

    manager.free_request("request-1")

    assert manager.get_block_table("request-2") == second_table_before == (2,)
    assert manager._block_pool.ref_count(2) == 1
    assert manager._block_pool.free_block_ids == (3, 0, 1)


def test_free_request_releases_one_reference_from_shared_block() -> None:
    manager = make_manager(num_blocks=2)
    manager.register_request("request-1")
    manager.register_request("request-2")
    shared_block_ids = manager._block_pool.allocate(1)
    assert shared_block_ids == (0,)
    manager._block_pool.retain(0)
    manager._block_tables["request-1"].append(0)
    manager._block_tables["request-2"].append(0)

    manager.free_request("request-1")

    assert manager._block_pool.ref_count(0) == 1
    assert manager._block_pool.free_block_ids == (1,)
    assert manager.get_block_table("request-2") == (0,)

    manager.free_request("request-2")
    assert manager._block_pool.ref_count(0) == 0
    assert manager._block_pool.free_block_ids == (1, 0)


def test_repeated_free_is_explicit_and_does_not_double_release() -> None:
    manager = make_manager(num_blocks=2)
    manager.register_request("request-1")
    assert manager.allocate_slots("request-1", 0, 1) is not None
    manager.free_request("request-1")
    free_blocks_before = manager._block_pool.free_block_ids

    with pytest.raises(KeyError, match="unknown request_id: request-1"):
        manager.free_request("request-1")

    assert manager._block_pool.free_block_ids == free_blocks_before
    assert manager._block_pool.ref_count(0) == 0


def test_get_block_table_returns_stable_read_only_snapshot() -> None:
    manager = make_manager()
    manager.register_request("request-1")
    assert manager.allocate_slots("request-1", 0, 1) is not None
    snapshot = manager.get_block_table("request-1")

    assert isinstance(snapshot, tuple)
    assert manager.allocate_slots("request-1", 4, 1) is not None
    assert snapshot == (0,)
    assert manager.get_block_table("request-1") == (0, 1)
