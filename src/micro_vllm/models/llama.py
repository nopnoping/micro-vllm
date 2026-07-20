"""Llama-family model configuration and assembly."""

from dataclasses import dataclass

import torch
from torch import nn

from micro_vllm.layers.attention import LlamaAttention
from micro_vllm.layers.attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from micro_vllm.layers.embedding import LMHead, TokenEmbedding
from micro_vllm.layers.mlp import SwiGLUMLP
from micro_vllm.layers.normalization import RMSNorm


@dataclass(frozen=True, slots=True)
class LlamaModelConfig:
    """Minimal architecture contract for the V1 dense Llama model."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be greater than zero")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.head_dim % 2 != 0:
            raise ValueError("attention head_dim must be even for rotary embedding")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be greater than zero")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be greater than zero")
        if self.hidden_act != "silu":
            raise ValueError("hidden_act must be 'silu' for the V1 Llama model")

    @property
    def head_dim(self) -> int:
        """Per-head hidden dimension."""

        return self.hidden_size // self.num_attention_heads

    @property
    def num_query_heads_per_kv_head(self) -> int:
        """Number of query heads sharing each key/value head."""

        return self.num_attention_heads // self.num_key_value_heads


class LlamaDecoderLayer(nn.Module):
    """Pre-norm Llama decoder layer over one physical KV cache."""

    def __init__(
        self,
        config: LlamaModelConfig,
        attention_backend: AttentionBackend,
    ) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            rope_theta=config.rope_theta,
            backend=attention_backend,
        )
        self.mlp = SwiGLUMLP(
            config.hidden_size,
            config.intermediate_size,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        layer_kv_cache: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            positions,
            layer_kv_cache,
            metadata,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class LlamaModel(nn.Module):
    """Tiny dense Llama model for flattened inference execution."""

    def __init__(
        self,
        config: LlamaModelConfig,
        attention_backend: AttentionBackend,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = TokenEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            LlamaDecoderLayer(config, attention_backend)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = LMHead(
            config.hidden_size,
            config.vocab_size,
            tied_embedding=(
                self.embed_tokens if config.tie_word_embeddings else None
            ),
        )
        self.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        layer_kv_caches: tuple[torch.Tensor, ...],
        attention_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer, layer_kv_cache in zip(self.layers, layer_kv_caches):
            hidden_states = layer(
                hidden_states,
                positions,
                layer_kv_cache,
                attention_metadata,
            )
        return self.norm(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits only for hidden-state rows selected by the caller."""

        return self.lm_head(hidden_states)
