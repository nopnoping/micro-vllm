"""Model-facing attention layer composition."""

import torch
from torch import nn

from micro_vllm.layers.attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from micro_vllm.layers.linear import Linear
from micro_vllm.layers.rotary_embedding import RotaryEmbedding


class LlamaAttention(nn.Module):
    """Dense Llama projections around an injected attention backend."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rope_theta: float,
        backend: AttentionBackend,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.q_proj = Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = Linear(
            hidden_size,
            num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = Linear(
            hidden_size,
            num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = Linear(hidden_size, hidden_size, bias=False)
        self.rotary_embedding = RotaryEmbedding(self.head_dim, base=rope_theta)
        self.backend = backend

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        layer_kv_cache: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        query = self.q_proj(hidden_states).view(
            -1,
            self.num_attention_heads,
            self.head_dim,
        )
        key = self.k_proj(hidden_states).view(
            -1,
            self.num_key_value_heads,
            self.head_dim,
        )
        value = self.v_proj(hidden_states).view(
            -1,
            self.num_key_value_heads,
            self.head_dim,
        )
        query, key = self.rotary_embedding(positions, query, key)
        attention = self.backend(
            query,
            key,
            value,
            layer_kv_cache,
            metadata,
        )
        return self.o_proj(attention.flatten(start_dim=1))
