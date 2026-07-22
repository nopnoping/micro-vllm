"""Tests for immutable multi-group KV-cache topology contracts."""

from dataclasses import FrozenInstanceError

import pytest

from micro_vllm.attention.spec import (
    ChunkedLocalSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
)
from micro_vllm.core.kv_cache_spec import (
    KVCacheLayout,
    build_kv_cache_topology,
)


def test_full_only_topology_uses_one_group() -> None:
    topology = build_kv_cache_topology(
        (FullAttentionSpec(), FullAttentionSpec())
    )

    assert len(topology.groups) == 1
    assert topology.groups[0].group_id == 0
    assert topology.groups[0].attention_spec == FullAttentionSpec()
    assert topology.groups[0].layer_indices == (0, 1)
    assert topology.page_layer_count == 2
    assert topology.num_layers == 2


def test_hybrid_topology_has_stable_groups_and_local_layer_indices() -> None:
    topology = build_kv_cache_topology(
        (
            FullAttentionSpec(),
            ChunkedLocalSpec(4),
            ChunkedLocalSpec(4),
            FullAttentionSpec(),
            ChunkedLocalSpec(4),
        )
    )

    assert tuple(group.attention_spec for group in topology.groups) == (
        FullAttentionSpec(),
        ChunkedLocalSpec(4),
    )
    assert tuple(group.layer_indices for group in topology.groups) == (
        (0, 3),
        (1, 2, 4),
    )
    assert topology.page_layer_count == 3


def test_different_attention_parameters_create_different_groups() -> None:
    topology = build_kv_cache_topology(
        (
            SlidingWindowSpec(128),
            SlidingWindowSpec(256),
            SlidingWindowSpec(128),
        )
    )

    assert tuple(group.attention_spec for group in topology.groups) == (
        SlidingWindowSpec(128),
        SlidingWindowSpec(256),
    )
    assert tuple(group.layer_indices for group in topology.groups) == (
        (0, 2),
        (1,),
    )
    assert topology.page_layer_count == 2


def test_layout_composes_frozen_topology_and_capacity() -> None:
    topology = build_kv_cache_topology(
        (FullAttentionSpec(), ChunkedLocalSpec(4))
    )
    layout = KVCacheLayout(
        block_size=4,
        num_blocks=8,
        topology=topology,
    )

    assert layout.groups == topology.groups
    assert layout.page_layer_count == 1
    assert layout.num_layers == 2

    with pytest.raises(FrozenInstanceError):
        layout.block_size = 8

    with pytest.raises(FrozenInstanceError):
        topology.page_layer_count = 2

    with pytest.raises(FrozenInstanceError):
        topology.groups[0].group_id = 1
