"""Tests for immutable multi-group KV-cache layout contracts."""

from dataclasses import FrozenInstanceError

import pytest

from micro_vllm.attention.spec import (
    ChunkedLocalSpec,
    FullAttentionSpec,
)
from micro_vllm.core.kv_cache_spec import KVCacheGroupSpec, KVCacheLayout


def make_full_group(
    group_id: int = 0,
    layer_indices: tuple[int, ...] = (0, 1),
) -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        group_id=group_id,
        attention_spec=FullAttentionSpec(),
        layer_indices=layer_indices,
    )


def test_full_only_layout_uses_the_same_single_group_contract() -> None:
    layout = KVCacheLayout(
        block_size=16,
        num_blocks=32,
        groups=(make_full_group(),),
    )

    assert layout.num_layers == 2
    assert layout.layer_attention_group_ids == (0, 0)
    assert layout.groups[0].group_id == 0


def test_hybrid_layout_uses_stable_first_layer_group_order() -> None:
    layout = KVCacheLayout(
        block_size=4,
        num_blocks=8,
        groups=(
            make_full_group(layer_indices=(0, 2)),
            KVCacheGroupSpec(
                group_id=1,
                attention_spec=ChunkedLocalSpec(4),
                layer_indices=(1, 3),
            ),
        ),
    )

    assert layout.layer_attention_group_ids == (0, 1, 0, 1)
    assert layout.groups[0].attention_spec == FullAttentionSpec()
    assert layout.groups[1].attention_spec == ChunkedLocalSpec(4)
    assert tuple(range(layout.num_blocks)) == tuple(range(8))


def test_layout_and_nested_groups_are_frozen() -> None:
    layout = KVCacheLayout(4, 8, (make_full_group(),))

    with pytest.raises(FrozenInstanceError):
        layout.block_size = 8

    with pytest.raises(FrozenInstanceError):
        layout.groups[0].group_id = 1

    assert layout.groups[0].layer_indices == (0, 1)
