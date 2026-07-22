"""Mistral-family sliding-window architecture and parameter loading."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from micro_vllm.attention.attention import Attention, AttentionMetadata
from micro_vllm.attention.spec import SlidingWindowSpec
from micro_vllm.config import ModelConfig
from micro_vllm.layers.embedding import LMHead, TokenEmbedding
from micro_vllm.layers.normalization import RMSNorm
from micro_vllm.models.llama import LlamaDecoderLayer


class MistralModel(nn.Module):
    """Single-device Mistral model with per-layer sliding-window attention."""

    def __init__(self, config: ModelConfig) -> None:
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
        sliding_window = config.require_positive_int("sliding_window")
        rms_norm_eps = values.get("rms_norm_eps", 1e-6)
        rope_theta = values.get("rope_theta", 10_000.0)
        self._tie_word_embeddings = values.get("tie_word_embeddings", False)

        attention = Attention(
            SlidingWindowSpec(sliding_window),
            config.attention_operator,
        )
        self.config = config
        self.layer_attentions = (attention,) * num_hidden_layers
        self.embed_tokens = TokenEmbedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            LlamaDecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
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
        layer_attention_metadata: tuple[AttentionMetadata, ...],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer, layer_kv_cache, metadata in zip(
            self.layers,
            layer_kv_caches,
            layer_attention_metadata,
            strict=True,
        ):
            hidden_states = layer(
                hidden_states,
                positions,
                layer_kv_cache,
                metadata,
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
                parameter_name = _mistral_parameter_name(
                    checkpoint_name,
                    tied_embeddings=self._tie_word_embeddings,
                )
                parameters[parameter_name].copy_(checkpoint_tensor)


def _mistral_parameter_name(
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
