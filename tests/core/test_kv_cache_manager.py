"""Tests for logical request KV-cache lifecycle management."""

from dataclasses import FrozenInstanceError

import pytest

from micro_vllm.config import CacheConfig, LLMConfig, ModelConfig
from micro_vllm.core.kv_cache_manager import KVCacheAllocation, KVCacheManager
from micro_vllm.core.prefix_cache import hash_block


def make_manager(
    block_size: int = 4,
    num_blocks: int = 9,
    *,
    enable_prefix_cache: bool = False,
) -> KVCacheManager:
    return KVCacheManager(
        LLMConfig(
            model_config=ModelConfig(model="unused"),
            cache_config=CacheConfig(
                block_size=block_size,
                num_blocks=num_blocks,
                enable_prefix_cache=enable_prefix_cache,
            ),
        )
    )


def cache_token_blocks(
    manager: KVCacheManager,
    token_blocks: tuple[tuple[int, ...], ...],
    *,
    namespace: str = "model-a",
) -> tuple[bytes, ...]:
    parent_hash = None
    block_hashes: list[bytes] = []

    for token_block in token_blocks:
        block_ids = manager._block_pool.allocate(1)
        assert block_ids is not None
        block_hash = hash_block(
            parent_hash,
            token_block,
            block_size=len(token_block),
            cache_namespace=namespace,
        )
        manager._block_pool.cache_block(block_ids[0], block_hash)
        manager._block_pool.release(block_ids[0])
        block_hashes.append(block_hash)
        parent_hash = block_hash

    return tuple(block_hashes)


def test_manager_owns_one_pool_and_empty_table_registry() -> None:
    manager = make_manager()

    assert manager._block_pool.num_blocks == 9
    assert manager._block_pool.num_free_blocks == 8
    assert manager._block_pool.null_block_id == 8
    assert manager._block_tables == {}


def test_register_request_creates_isolated_empty_block_tables() -> None:
    manager = make_manager()

    manager.register_request("request-1")
    manager.register_request("request-2")

    assert manager.get_block_table("request-1") == ()
    assert manager.get_block_table("request-2") == ()
    assert manager._block_tables["request-1"] is not manager._block_tables["request-2"]


def test_find_then_allocate_prefix_hit_separates_discovery_from_ownership() -> None:
    manager = make_manager(
        block_size=2,
        num_blocks=5,
        enable_prefix_cache=True,
    )
    block_hashes = cache_token_blocks(manager, ((1, 2), (3, 4)))
    manager.register_request("request-1")

    hit_block_ids, num_hit_tokens = manager.find_prefix_hit(
        "request-1",
        (1, 2, 3, 4, 5),
        cache_namespace="model-a",
    )

    assert hit_block_ids == (0, 1)
    assert num_hit_tokens == 4
    assert manager.get_block_table("request-1") == ()
    assert manager._block_pool.ref_count(0) == 0
    assert manager._block_pool.ref_count(1) == 0

    allocation = manager.allocate_slots(
        "request-1",
        num_hit_tokens,
        1,
        hit_block_ids,
    )

    assert allocation == KVCacheAllocation((0, 1, 2), (2,))
    assert manager.get_block_table("request-1") == (0, 1, 2)
    assert manager._block_pool.ref_count(0) == 1
    assert manager._block_pool.ref_count(1) == 1

    manager.free_request("request-1")
    assert manager._block_pool.ref_count(0) == 0
    assert manager._block_pool.ref_count(1) == 0
    assert manager._block_pool.get_cached_block(block_hashes[0]) is not None
    assert manager._block_pool.get_cached_block(block_hashes[1]) is not None


def test_find_prefix_hit_ignores_partial_token_block() -> None:
    manager = make_manager(block_size=2, enable_prefix_cache=True)
    manager.register_request("request-1")

    assert manager.find_prefix_hit(
        "request-1",
        (1,),
        cache_namespace="model-a",
    ) == ((), 0)
    assert manager.get_block_table("request-1") == ()


@pytest.mark.parametrize(
    ("token_ids", "namespace"),
    [
        ((1, 3, 9), "model-a"),
        ((1, 2, 9), "model-b"),
    ],
)
def test_find_prefix_hit_rejects_different_token_or_namespace(
    token_ids: tuple[int, ...],
    namespace: str,
) -> None:
    manager = make_manager(block_size=2, enable_prefix_cache=True)
    cache_token_blocks(manager, ((1, 2),))
    manager.register_request("request-1")

    assert manager.find_prefix_hit(
        "request-1",
        token_ids,
        cache_namespace=namespace,
    ) == ((), 0)
    assert manager.get_block_table("request-1") == ()
    assert manager._block_pool.ref_count(0) == 0


def test_find_prefix_hit_stops_at_first_missing_parent_chain_block() -> None:
    manager = make_manager(block_size=2, enable_prefix_cache=True)
    first_hash = cache_token_blocks(manager, ((1, 2),))[0]
    missing_hash = hash_block(
        first_hash,
        (3, 4),
        block_size=2,
        cache_namespace="model-a",
    )
    third_hash = hash_block(
        missing_hash,
        (5, 6),
        block_size=2,
        cache_namespace="model-a",
    )
    third_block_ids = manager._block_pool.allocate(1)
    assert third_block_ids == (1,)
    manager._block_pool.cache_block(1, third_hash)
    manager._block_pool.release(1)
    manager.register_request("request-1")

    assert manager.find_prefix_hit(
        "request-1",
        (1, 2, 3, 4, 5, 6, 7),
        cache_namespace="model-a",
    ) == ((0,), 2)
    assert manager.get_block_table("request-1") == ()
    assert manager._block_pool.ref_count(0) == 0
    assert manager._block_pool.ref_count(1) == 0


def test_allocate_prefix_hit_is_atomic_when_capacity_is_insufficient() -> None:
    manager = make_manager(
        block_size=2,
        num_blocks=3,
        enable_prefix_cache=True,
    )
    block_hashes = cache_token_blocks(manager, ((1, 2), (3, 4)))
    manager.register_request("request-1")
    hit_block_ids, num_hit_tokens = manager.find_prefix_hit(
        "request-1",
        (1, 2, 3, 4, 5),
        cache_namespace="model-a",
    )
    assert hit_block_ids == (0, 1)

    assert manager.allocate_slots(
        "request-1",
        num_hit_tokens,
        1,
        hit_block_ids,
    ) is None

    assert manager.get_block_table("request-1") == ()
    assert manager._block_pool.ref_count(0) == 0
    assert manager._block_pool.ref_count(1) == 0
    assert manager._block_pool.get_cached_block(block_hashes[0]) is not None
    assert manager._block_pool.get_cached_block(block_hashes[1]) is not None


def test_find_prefix_hit_is_disabled_by_cache_config() -> None:
    manager = make_manager(block_size=2, enable_prefix_cache=False)
    cache_token_blocks(manager, ((1, 2),))
    manager.register_request("request-1")

    assert manager.find_prefix_hit(
        "request-1",
        (1, 2),
        cache_namespace="model-a",
    ) == ((), 0)
    assert manager.get_block_table("request-1") == ()
    assert manager._block_pool.ref_count(0) == 0


@pytest.mark.parametrize(
    ("token_blocks", "token_ids", "expected_hit"),
    [
        (((1, 2),), (1, 2), ((), 0)),
        (((1, 2), (3, 4)), (1, 2, 3, 4), ((0,), 2)),
    ],
)
def test_find_prefix_hit_keeps_last_aligned_block_for_model_computation(
    token_blocks: tuple[tuple[int, ...], ...],
    token_ids: tuple[int, ...],
    expected_hit: tuple[tuple[int, ...], int],
) -> None:
    manager = make_manager(block_size=2, enable_prefix_cache=True)
    cache_token_blocks(manager, token_blocks)
    manager.register_request("request-1")

    assert manager.find_prefix_hit(
        "request-1",
        token_ids,
        cache_namespace="model-a",
    ) == expected_hit
    assert manager.get_block_table("request-1") == ()
    assert manager._block_pool.ref_count(0) == 0
    if len(token_blocks) == 2:
        assert manager._block_pool.ref_count(1) == 0


def test_commit_prefix_blocks_caches_only_complete_blocks_and_is_idempotent() -> None:
    manager = make_manager(
        block_size=2,
        num_blocks=5,
        enable_prefix_cache=True,
    )
    manager.register_request("request-1")
    assert manager.allocate_slots("request-1", 0, 5) is not None

    first_commit = manager.commit_prefix_blocks(
        "request-1",
        (1, 2, 3, 4, 5),
        5,
        cache_namespace="model-a",
    )
    second_commit = manager.commit_prefix_blocks(
        "request-1",
        (1, 2, 3, 4, 5),
        5,
        cache_namespace="model-a",
    )

    assert first_commit == second_commit == (0, 1)
    assert manager.get_block_table("request-1") == (0, 1, 2)
    assert manager._block_pool.num_cached_blocks == 2
    assert manager._block_pool.block_hash(0) is not None
    assert manager._block_pool.block_hash(1) is not None
    assert manager._block_pool.block_hash(2) is None
    assert tuple(manager._block_pool.ref_count(block_id) for block_id in range(3)) == (
        1,
        1,
        1,
    )


def test_commit_prefix_blocks_reuses_canonical_block_and_releases_duplicate() -> None:
    manager = make_manager(
        block_size=2,
        num_blocks=3,
        enable_prefix_cache=True,
    )
    manager.register_request("request-1")
    assert manager.allocate_slots("request-1", 0, 2) is not None
    assert manager.commit_prefix_blocks(
        "request-1",
        (1, 2),
        2,
        cache_namespace="model-a",
    ) == (0,)
    manager.free_request("request-1")

    manager.register_request("request-2")
    allocation = manager.allocate_slots("request-2", 0, 2)
    assert allocation == KVCacheAllocation((1,), (1,))

    canonical_block_ids = manager.commit_prefix_blocks(
        "request-2",
        (1, 2),
        2,
        cache_namespace="model-a",
    )

    assert canonical_block_ids == (0,)
    assert manager.get_block_table("request-2") == (0,)
    assert manager._block_pool.ref_count(0) == 1
    assert manager._block_pool.ref_count(1) == 0
    assert manager._block_pool.free_block_ids == (1,)
    assert manager._block_pool.num_cached_blocks == 1


def test_commit_prefix_blocks_is_disabled_by_cache_config() -> None:
    manager = make_manager(block_size=2, enable_prefix_cache=False)
    manager.register_request("request-1")
    assert manager.allocate_slots("request-1", 0, 2) is not None

    assert manager.commit_prefix_blocks(
        "request-1",
        (1, 2),
        2,
        cache_namespace="model-a",
    ) == ()
    assert manager._block_pool.num_cached_blocks == 0
    assert manager._block_pool.block_hash(0) is None


@pytest.mark.parametrize("operation", ["get_block_table", "free_request"])
def test_unknown_request_operations_raise_key_error(operation: str) -> None:
    manager = make_manager()

    with pytest.raises(KeyError):
        getattr(manager, operation)("missing")


def test_free_registered_request_removes_its_table() -> None:
    manager = make_manager()
    manager.register_request("request-1")

    manager.free_request("request-1")

    with pytest.raises(KeyError):
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
    manager._block_tables["request-1"].extend((0,))

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
    manager._block_tables["request-1"].extend((0,))
    table_before = manager.get_block_table("request-1")
    num_free_blocks_before = manager._block_pool.num_free_blocks

    assert manager.get_num_blocks_to_allocate("request-1", 3, 6) == 2

    assert manager.get_block_table("request-1") == table_before
    assert manager._block_pool.num_free_blocks == num_free_blocks_before


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
    manager = make_manager(num_blocks=5)
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
    manager = make_manager(num_blocks=3)
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
    assert tuple(manager._block_pool.ref_count(block_id) for block_id in range(3)) == (
        1,
        0,
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
        block_table=[0, 8],  # type: ignore[arg-type]
        new_block_ids=[0],  # type: ignore[arg-type]
    )

    assert allocation.block_table == (0, 8)
    assert allocation.new_block_ids == (0,)
    with pytest.raises(FrozenInstanceError):
        allocation.block_table = ()


def test_free_request_blocks_keeps_an_empty_table_for_recompute() -> None:
    manager = make_manager(num_blocks=3)
    manager.register_request("request-1")
    assert manager.allocate_slots("request-1", 0, 5) is not None

    manager.free_request_blocks("request-1")

    assert manager.get_block_table("request-1") == ()
    assert manager.num_free_blocks == 2
    assert manager.allocate_slots("request-1", 0, 1) is not None


def test_repeated_recompute_cycles_release_each_reference_once() -> None:
    manager = make_manager(num_blocks=3)
    manager.register_request("request-1")

    for _ in range(3):
        assert manager.allocate_slots("request-1", 0, 3) is not None
        manager.free_request_blocks("request-1")
        assert manager.get_block_table("request-1") == ()
        assert manager.num_free_blocks == 2
        ref_counts = tuple(
            manager._block_pool.ref_count(block_id) for block_id in range(3)
        )
        assert ref_counts == (
            0,
            0,
            0,
        )


def test_free_request_recycles_all_request_owned_blocks() -> None:
    manager = make_manager(num_blocks=5)
    manager.register_request("request-1")
    assert manager.allocate_slots("request-1", 0, 9) is not None

    manager.free_request("request-1")

    assert manager._block_pool.num_free_blocks == 4
    assert manager._block_pool.free_block_ids == (3, 0, 1, 2)
    assert tuple(manager._block_pool.ref_count(block_id) for block_id in range(5)) == (
        0,
        0,
        0,
        0,
        0,
    )
    with pytest.raises(KeyError):
        manager.get_block_table("request-1")


def test_free_request_does_not_affect_other_request() -> None:
    manager = make_manager(num_blocks=5)
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
    manager = make_manager(num_blocks=3)
    manager.register_request("request-1")
    manager.register_request("request-2")
    shared_block_ids = manager._block_pool.allocate(1)
    assert shared_block_ids == (0,)
    block_hash = b"a" * 32
    manager._block_pool.cache_block(0, block_hash)
    assert manager._block_pool.acquire_cached_block(block_hash) is not None
    manager._block_tables["request-1"].extend((0,))
    manager._block_tables["request-2"].extend((0,))

    manager.free_request("request-1")

    assert manager._block_pool.ref_count(0) == 1
    assert manager._block_pool.free_block_ids == (1,)
    assert manager.get_block_table("request-2") == (0,)

    manager.free_request("request-2")
    assert manager._block_pool.ref_count(0) == 0
    assert manager._block_pool.free_block_ids == (1, 0)


def test_repeated_free_is_explicit_and_does_not_double_release() -> None:
    manager = make_manager(num_blocks=3)
    manager.register_request("request-1")
    assert manager.allocate_slots("request-1", 0, 1) is not None
    manager.free_request("request-1")
    free_blocks_before = manager._block_pool.free_block_ids

    with pytest.raises(KeyError):
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
