"""Llama-family model configuration and assembly."""

from dataclasses import dataclass


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
