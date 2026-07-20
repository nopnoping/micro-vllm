"""Token embedding and language-model head layers."""

from torch import nn

from micro_vllm.layers.linear import Linear


class TokenEmbedding(nn.Embedding):
    """Single-device token embedding table."""


class LMHead(Linear):
    """Bias-free dense language-model head with optional tied weights."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        tied_embedding: TokenEmbedding | None = None,
    ) -> None:
        super().__init__(hidden_size, vocab_size, bias=False)
        if tied_embedding is not None:
            self.weight = tied_embedding.weight
