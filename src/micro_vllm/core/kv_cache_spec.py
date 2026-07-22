"""Immutable KV-cache layout shared by control and data planes."""

from __future__ import annotations

from dataclasses import dataclass

from micro_vllm.attention.spec import AttentionSpec


@dataclass(frozen=True, slots=True)
class KVCacheGroupSpec:
    """One group of layers sharing attention and block-cache semantics."""

    group_id: int
    attention_spec: AttentionSpec
    layer_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class KVCacheLayout:
    """Stable group ordering over one shared physical block namespace.

    All groups allocate block IDs from the same ``[0, num_blocks)`` pool.
    The union of ``layer_indices`` is exactly ``range(num_layers)`` where
    ``num_layers`` is inferred from the immutable group declarations.
    """

    block_size: int
    num_blocks: int
    groups: tuple[KVCacheGroupSpec, ...]

    @property
    def num_layers(self) -> int:
        """Number of consecutively covered model layers."""

        return sum(len(group.layer_indices) for group in self.groups)

    @property
    def layer_attention_group_ids(self) -> tuple[int, ...]:
        """Return the group ID selected by every layer in model order."""

        group_ids = [0] * self.num_layers
        for group in self.groups:
            for layer_index in group.layer_indices:
                group_ids[layer_index] = group.group_id
        return tuple(group_ids)
