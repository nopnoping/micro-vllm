"""Runtime metadata for full causal attention."""

from dataclasses import dataclass

import torch

from micro_vllm.attention.base import AttentionMetadata


@dataclass(frozen=True, slots=True, eq=False)
class FullAttentionMetadata(AttentionMetadata):
    """Flattened query and paged-cache inputs for full attention."""

    query_positions: torch.Tensor
    query_start_locations: torch.Tensor
    context_lengths: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int

    @property
    def num_requests(self) -> int:
        """Number of requests represented by this batch."""

        return self.context_lengths.numel()
