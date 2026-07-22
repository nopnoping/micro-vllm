"""Tests for Llama-specific checkpoint parameter semantics."""

import pytest
import torch

from micro_vllm.attention.registry import create_attention
from micro_vllm.config import ModelConfig
from micro_vllm.models.llama import LlamaModel


def make_model(*, tied: bool = False) -> LlamaModel:
    config = ModelConfig(
        model="unused",
        metadata={
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 32,
            "tie_word_embeddings": tied,
        },
    )
    return LlamaModel(config, create_attention("reference"))


def checkpoint_tensors(
    model: LlamaModel,
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


def test_llama_load_weights_copies_every_parameter() -> None:
    model = make_model()
    weights, expected = checkpoint_tensors(model)

    model.load_weights(weights.items())

    for parameter_name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter,
            expected[parameter_name].to(parameter.dtype),
        )


def test_llama_load_weights_preserves_tied_embedding() -> None:
    model = make_model(tied=True)
    weights, expected = checkpoint_tensors(model)

    model.load_weights(weights.items())

    assert model.embed_tokens.weight is model.lm_head.weight
    torch.testing.assert_close(
        model.embed_tokens.weight,
        expected["embed_tokens.weight"].to(model.embed_tokens.weight.dtype),
    )


def test_llama_load_weights_accepts_lm_head_alias_when_tied() -> None:
    model = make_model(tied=True)
    weights, expected = checkpoint_tensors(model)
    embedding = weights.pop("model.embed_tokens.weight")
    weights["lm_head.weight"] = embedding

    model.load_weights(weights.items())

    torch.testing.assert_close(
        model.embed_tokens.weight,
        expected["embed_tokens.weight"].to(model.embed_tokens.weight.dtype),
    )


def test_llama_load_weights_unknown_name_preserves_parameter_name() -> None:
    model = make_model()

    with pytest.raises(KeyError, match="unexpected.weight"):
        model.load_weights(
            [("model.unexpected.weight", torch.zeros(1))]
        )
