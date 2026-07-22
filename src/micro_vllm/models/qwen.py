"""Qwen3 dense architecture and parameter loading."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from micro_vllm.attention.attention import Attention, AttentionMetadata
from micro_vllm.attention.spec import FullAttentionSpec
from micro_vllm.config import ModelConfig
from micro_vllm.layers.embedding import LMHead, TokenEmbedding
from micro_vllm.layers.linear import Linear
from micro_vllm.layers.mlp import SwiGLUMLP
from micro_vllm.layers.normalization import RMSNorm
from micro_vllm.layers.rotary_embedding import RotaryEmbedding


class Qwen3Attention(nn.Module):
    """Qwen3 projections, per-head Q/K normalization, and paged attention."""

    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rope_theta: float,
        rms_norm_eps: float,
        attention_bias: bool,
        attention: Attention,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        query_size = num_attention_heads * head_dim
        key_value_size = num_key_value_heads * head_dim
        self.q_proj = Linear(
            hidden_size,
            query_size,
            bias=attention_bias,
        )
        self.k_proj = Linear(
            hidden_size,
            key_value_size,
            bias=attention_bias,
        )
        self.v_proj = Linear(
            hidden_size,
            key_value_size,
            bias=attention_bias,
        )
        self.o_proj = Linear(
            query_size,
            hidden_size,
            bias=attention_bias,
        )
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.rotary_embedding = RotaryEmbedding(head_dim, base=rope_theta)
        self.attention = attention

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
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = self.rotary_embedding(positions, query, key)
        attention_output = self.attention(
            query,
            key,
            value,
            layer_kv_cache,
            metadata,
        )
        return self.o_proj(attention_output.flatten(start_dim=1))


class Qwen3DecoderLayer(nn.Module):
    """Pre-norm Qwen3 decoder layer over one physical KV cache."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rope_theta: float,
        rms_norm_eps: float,
        attention_bias: bool,
        attention: Attention,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            attention=attention,
        )
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            eps=rms_norm_eps,
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


class Qwen3Model(nn.Module):
    """Single-device dense Qwen3 model for flattened inference execution."""

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        values = config.values
        vocab_size = config.require_positive_int("vocab_size")
        hidden_size = config.require_positive_int("hidden_size")
        intermediate_size = config.require_positive_int("intermediate_size")
        num_hidden_layers = config.require_positive_int("num_hidden_layers")
        num_attention_heads = config.require_positive_int(
            "num_attention_heads"
        )
        num_key_value_heads = config.positive_int(
            "num_key_value_heads",
            num_attention_heads,
        )
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        head_dim = config.positive_int(
            "head_dim",
            hidden_size // num_attention_heads,
        )
        rms_norm_eps = values.get("rms_norm_eps", 1e-6)
        rope_theta = values.get("rope_theta", 1_000_000.0)
        attention_bias = values.get("attention_bias", False)
        self._tie_word_embeddings = values.get("tie_word_embeddings", False)

        attention = Attention(FullAttentionSpec(), config.attention_operator)
        self.config = config
        self.layer_attentions = (attention,) * num_hidden_layers
        self.embed_tokens = TokenEmbedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                head_dim=head_dim,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
                attention_bias=attention_bias,
                attention=attention,
            )
            for _ in range(num_hidden_layers)
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.lm_head = LMHead(
            hidden_size,
            vocab_size,
            tied_embedding=(
                self.embed_tokens if self._tie_word_embeddings else None
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

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> None:
        """Copy checkpoint tensors directly into model parameters."""
        parameters = dict(self.named_parameters())

        with torch.no_grad():
            for checkpoint_name, checkpoint_tensor in weights:
                parameter_name = _qwen3_parameter_name(
                    checkpoint_name,
                    tied_embeddings=self._tie_word_embeddings,
                )
                parameters[parameter_name].copy_(checkpoint_tensor)


def _qwen3_parameter_name(
    checkpoint_name: str,
    *,
    tied_embeddings: bool,
) -> str:
    if checkpoint_name.startswith("model."):
        parameter_name = checkpoint_name.removeprefix("model.")
    else:
        parameter_name = checkpoint_name
    if tied_embeddings and parameter_name == "lm_head.weight":
        return "embed_tokens.weight"
    return parameter_name
