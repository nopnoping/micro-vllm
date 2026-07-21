"""Tests for token embedding and language-model head layers."""

import pytest
import torch
from torch.nn import functional as F

from micro_vllm.layers.embedding import LMHead, TokenEmbedding
from micro_vllm.config import ModelConfig


def make_config(*, tie_word_embeddings: bool) -> ModelConfig:
    return ModelConfig(
        {
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 32,
            "tie_word_embeddings": tie_word_embeddings,
        }
    )


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_embedding_weight_tying_follows_model_config(
    tie_word_embeddings: bool,
) -> None:
    config = make_config(tie_word_embeddings=tie_word_embeddings)
    vocab_size = config.require_positive_int("vocab_size")
    hidden_size = config.require_positive_int("hidden_size")
    tied = config.get("tie_word_embeddings", False)
    embedding = TokenEmbedding(vocab_size, hidden_size)
    lm_head = LMHead(
        hidden_size,
        vocab_size,
        tied_embedding=embedding if tied else None,
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
