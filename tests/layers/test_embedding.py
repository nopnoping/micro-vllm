"""Tests for token embedding and language-model head layers."""

import pytest
import torch
from torch.nn import functional as F

from micro_vllm.layers.embedding import LMHead, TokenEmbedding


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_embedding_weight_tying_is_explicit(
    tie_word_embeddings: bool,
) -> None:
    vocab_size = 16
    hidden_size = 8
    embedding = TokenEmbedding(vocab_size, hidden_size)
    lm_head = LMHead(
        hidden_size,
        vocab_size,
        tied_embedding=embedding if tie_word_embeddings else None,
    )

    assert (lm_head.weight is embedding.weight) is tie_word_embeddings


def test_token_embedding_and_lm_head_match_dense_torch_operations() -> None:
    embedding = TokenEmbedding(num_embeddings=16, embedding_dim=8)
    lm_head = LMHead(hidden_size=8, vocab_size=16)
    input_ids = torch.tensor([[1, 3], [5, 7]], dtype=torch.int64)
    hidden_states = embedding(input_ids)

    logits = lm_head(hidden_states)

    assert torch.equal(
        hidden_states,
        F.embedding(input_ids, embedding.weight),
    )
    assert torch.equal(logits, F.linear(hidden_states, lm_head.weight))
    assert lm_head.bias is None
