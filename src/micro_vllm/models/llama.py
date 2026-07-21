"""Llama-family architecture and parameter loading."""

from __future__ import annotations

from collections.abc import Iterable

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
from micro_vllm.config import ModelConfig


class LlamaDecoderLayer(nn.Module):
    """Pre-norm Llama decoder layer over one physical KV cache."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rope_theta: float,
        rms_norm_eps: float,
        attention_backend: AttentionBackend,
    ) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            rope_theta=rope_theta,
            backend=attention_backend,
        )
        self.mlp = SwiGLUMLP(
            hidden_size,
            intermediate_size,
        )
        self.input_layernorm = RMSNorm(
            hidden_size,
            eps=rms_norm_eps,
        )
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


class LlamaModel(nn.Module):
    """Tiny dense Llama model for flattened inference execution."""

    def __init__(
        self,
        config: ModelConfig,
        attention_backend: AttentionBackend,
    ) -> None:
        super().__init__()
        values = config.values
        vocab_size = config.require_positive_int("vocab_size")
        hidden_size = config.require_positive_int("hidden_size")
        intermediate_size = config.require_positive_int("intermediate_size")
        num_hidden_layers = config.require_positive_int("num_hidden_layers")
        num_attention_heads = config.require_positive_int("num_attention_heads")
        num_key_value_heads = config.positive_int(
            "num_key_value_heads",
            num_attention_heads,
        )
        rms_norm_eps = values.get("rms_norm_eps", 1e-6)
        rope_theta = values.get("rope_theta", 10_000.0)
        self._tie_word_embeddings = values.get("tie_word_embeddings", False)

        self.config = config
        self.embed_tokens = TokenEmbedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            LlamaDecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
                attention_backend=attention_backend,
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
                parameter_name = _llama_parameter_name(
                    checkpoint_name,
                    tied_embeddings=self._tie_word_embeddings,
                )
                parameters[parameter_name].copy_(checkpoint_tensor)


def _llama_parameter_name(
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
