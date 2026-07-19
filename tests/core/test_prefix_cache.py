"""Tests for deterministic full-block prefix-cache identities."""

import builtins

import pytest

from micro_vllm.core.prefix_cache import (
    SINGLE_CACHE_GROUP_ID,
    hash_block,
)


def make_hash(
    token_ids: tuple[int, ...],
    *,
    parent: bytes | None = None,
    namespace: str = "model-a",
) -> bytes:
    return hash_block(
        parent,
        token_ids,
        block_size=len(token_ids),
        cache_namespace=namespace,
    )


def test_same_prefix_and_namespace_have_stable_hash() -> None:
    first = make_hash((1, 2, 3, 4))
    second = make_hash((1, 2, 3, 4))

    assert first == second
    assert first.hex() == (
        "fe7ca7b417786c985f1887b3d0ac9dace5ab4cbb74e201f115d7ee22cd15a52e"
    )


def test_hash_does_not_use_process_randomized_builtin_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_builtin_hash(value: object) -> int:
        raise AssertionError(f"unexpected builtin hash call for {value!r}")

    with monkeypatch.context() as context:
        context.setattr(builtins, "hash", reject_builtin_hash)
        digest = make_hash((1, 2, 3, 4))

    assert len(digest) == 32


def test_same_token_block_with_different_parent_does_not_share_hash() -> None:
    first_parent = make_hash((1, 2))
    second_parent = make_hash((3, 4))

    first_child = make_hash((5, 6), parent=first_parent)
    second_child = make_hash((5, 6), parent=second_parent)

    assert first_child != second_child


def test_hash_changes_with_current_token_block() -> None:
    parent = make_hash((1, 2))

    assert make_hash((3, 4), parent=parent) != make_hash((3, 5), parent=parent)


def test_hash_changes_with_cache_namespace() -> None:
    assert make_hash((1, 2), namespace="model-a") != make_hash(
        (1, 2), namespace="model-b"
    )


def test_cache_group_id_is_part_of_the_hash() -> None:
    assert SINGLE_CACHE_GROUP_ID == 0
    assert hash_block(
        None,
        (1, 2),
        block_size=2,
        cache_namespace="model-a",
        cache_group_id=0,
    ) == make_hash((1, 2))

    assert hash_block(
        None,
        (1, 2),
        block_size=2,
        cache_namespace="model-a",
        cache_group_id=1,
    ) != make_hash((1, 2))
