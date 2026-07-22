"""Immutable KV-cache topology shared by control and data planes."""

from __future__ import annotations

from dataclasses import dataclass

from micro_vllm.attention.spec import AttentionSpec


@dataclass(frozen=True, slots=True)
class KVCacheGroupSpec:
    """Layers sharing one attention and logical cache semantic."""

    group_id: int
    attention_spec: AttentionSpec
    layer_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class KVCacheTopology:
    """Stable cache groups derived from model-layer attention specs."""

    groups: tuple[KVCacheGroupSpec, ...]
    page_layer_count: int

    @property
    def num_layers(self) -> int:
        return sum(len(group.layer_indices) for group in self.groups)


@dataclass(frozen=True, slots=True)
class KVCacheLayout:
    """A model topology with its initialized physical block capacity."""

    block_size: int
    num_blocks: int
    topology: KVCacheTopology

    @property
    def groups(self) -> tuple[KVCacheGroupSpec, ...]:
        return self.topology.groups

    @property
    def page_layer_count(self) -> int:
        return self.topology.page_layer_count

    @property
    def num_layers(self) -> int:
        return self.topology.num_layers


def build_kv_cache_topology(
    layer_attention_specs: tuple[AttentionSpec, ...],
) -> KVCacheTopology:
    """Group equal per-layer specs in stable first-occurrence order."""

    group_ids_by_spec: dict[AttentionSpec, int] = {}
    group_layer_indices: list[list[int]] = []

    for layer_index, attention_spec in enumerate(layer_attention_specs):
        group_id = group_ids_by_spec.get(attention_spec)
        if group_id is None:
            group_id = len(group_ids_by_spec)
            group_ids_by_spec[attention_spec] = group_id
            group_layer_indices.append([])

        group_layer_indices[group_id].append(layer_index)

    groups = tuple(
        KVCacheGroupSpec(
            group_id=group_id,
            attention_spec=attention_spec,
            layer_indices=tuple(group_layer_indices[group_id]),
        )
        for attention_spec, group_id in group_ids_by_spec.items()
    )
    return KVCacheTopology(
        groups=groups,
        page_layer_count=max(len(group.layer_indices) for group in groups),
    )
