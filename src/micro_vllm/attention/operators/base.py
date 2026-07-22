"""Tensor operator contract shared by all attention semantics."""

from abc import ABC, abstractmethod

import torch

from micro_vllm.attention.attention import AttentionMetadata


class AttentionOperator(ABC):
    """Compute attention from Q/K/V tensors and semantic metadata."""

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_kv_cache: torch.Tensor,
        metadata: AttentionMetadata,
        /,
    ) -> torch.Tensor:
        return self.forward(
            query,
            key,
            value,
            layer_kv_cache,
            metadata,
        )

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_kv_cache: torch.Tensor,
        metadata: AttentionMetadata,
        /,
    ) -> torch.Tensor:
        """Compute attention for one layer."""
