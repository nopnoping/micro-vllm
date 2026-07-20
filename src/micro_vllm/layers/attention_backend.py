"""Reference and optimized attention backend contracts."""

from dataclasses import dataclass
from typing import Final

import torch


PAD_BLOCK_ID: Final = -1


@dataclass(frozen=True, slots=True, eq=False)
class AttentionMetadata:
    """Flattened query and physical-cache indexing metadata."""

    query_positions: torch.Tensor
    query_start_locations: torch.Tensor
    context_lengths: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int

    def __post_init__(self) -> None:
        tensors = (
            ("query_positions", self.query_positions),
            ("query_start_locations", self.query_start_locations),
            ("context_lengths", self.context_lengths),
            ("block_table", self.block_table),
            ("slot_mapping", self.slot_mapping),
        )
        for field_name, tensor in tensors:
            if tensor.dtype is not torch.int64:
                raise ValueError(f"{field_name} must use torch.int64")
            if tensor.device != self.query_positions.device:
                raise ValueError("attention metadata tensors must share one device")

        for field_name, tensor in tensors[:3] + tensors[4:]:
            if tensor.ndim != 1:
                raise ValueError(f"{field_name} must be one-dimensional")
        if self.block_table.ndim != 2:
            raise ValueError("block_table must be two-dimensional")
        if self.block_size <= 0:
            raise ValueError("block_size must be greater than zero")

        if self.query_start_locations.numel() != self.num_requests + 1:
            raise ValueError(
                "query_start_locations length must equal num_requests + 1"
            )
        if self.block_table.shape[0] != self.num_requests:
            raise ValueError("block_table rows must equal num_requests")
        if self.slot_mapping.numel() != self.num_query_tokens:
            raise ValueError(
                "slot_mapping length must equal num_query_tokens"
            )

    @property
    def num_requests(self) -> int:
        """Number of requests represented by the metadata."""

        return self.context_lengths.numel()

    @property
    def num_query_tokens(self) -> int:
        """Number of flattened query tokens."""

        return self.query_positions.numel()
