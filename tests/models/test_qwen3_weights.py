"""Tests for Qwen3-specific checkpoint parameter semantics."""

import pytest
import torch

from micro_vllm.config import ModelConfig
from micro_vllm.models.qwen import Qwen3Model


def make_model(*, tied: bool = True) -> Qwen3Model:
    config = ModelConfig(
        model="unused",
        metadata={
            "architectures": ["Qwen3ForCausalLM"],
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 32,
            "tie_word_embeddings": tied,
            "attention_bias": False,
        },
    )
    return Qwen3Model(config)


def checkpoint_tensors(
    model: Qwen3Model,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    weights = {}
    expected = {}
    for index, (parameter_name, parameter) in enumerate(
        model.named_parameters(),
        start=1,
    ):
        checkpoint_name = (
            parameter_name
            if parameter_name == "lm_head.weight"
            else f"model.{parameter_name}"
        )
        tensor = torch.full(parameter.shape, float(index), dtype=torch.float16)
        weights[checkpoint_name] = tensor
        expected[parameter_name] = tensor
    return weights, expected


def test_qwen3_load_weights_copies_projections_and_qk_norm() -> None:
    model = make_model(tied=False)
    weights, expected = checkpoint_tensors(model)

    model.load_weights(weights.items())

    for parameter_name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter,
            expected[parameter_name].to(parameter.dtype),
        )
    assert "model.layers.0.self_attn.q_norm.weight" in weights
    assert "model.layers.0.self_attn.k_norm.weight" in weights


def test_qwen3_load_weights_accepts_both_tied_checkpoint_names() -> None:
    model = make_model(tied=True)
    embedding = torch.full_like(model.embed_tokens.weight, 3.0)

    model.load_weights(
        [
            ("lm_head.weight", embedding),
            ("model.embed_tokens.weight", embedding),
        ]
    )

    assert model.embed_tokens.weight is model.lm_head.weight
    torch.testing.assert_close(model.embed_tokens.weight, embedding)


def test_qwen3_load_weights_unknown_name_preserves_parameter_name() -> None:
    model = make_model()

    with pytest.raises(KeyError, match="unexpected.weight"):
        model.load_weights([("model.unexpected.weight", torch.zeros(1))])
