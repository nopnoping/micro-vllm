"""Tests for Mistral construction, weights, and sliding-window execution."""

import pytest
import torch
from transformers import MistralConfig, MistralForCausalLM

from micro_vllm.attention.attention import AttentionMetadata
from micro_vllm.attention.spec import SlidingWindowSpec
from micro_vllm.config import ModelConfig
from micro_vllm.models.mistral import MistralModel


def make_config(**overrides: object) -> ModelConfig:
    values = {
        "architectures": ["MistralForCausalLM"],
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "sliding_window": 4,
    }
    values.update(overrides)
    return ModelConfig(model="unused", metadata=values)


def make_reference_config(config: ModelConfig) -> MistralConfig:
    return MistralConfig(
        architectures=["MistralForCausalLM"],
        vocab_size=config.require_positive_int("vocab_size"),
        hidden_size=config.require_positive_int("hidden_size"),
        intermediate_size=config.require_positive_int("intermediate_size"),
        num_hidden_layers=config.require_positive_int("num_hidden_layers"),
        num_attention_heads=config.require_positive_int(
            "num_attention_heads"
        ),
        num_key_value_heads=config.positive_int(
            "num_key_value_heads",
            config.require_positive_int("num_attention_heads"),
        ),
        max_position_embeddings=config.require_positive_int(
            "max_position_embeddings"
        ),
        sliding_window=config.require_positive_int("sliding_window"),
        rms_norm_eps=float(config.values.get("rms_norm_eps", 1e-6)),
        rope_theta=float(config.values.get("rope_theta", 10_000.0)),
        hidden_act="silu",
        tie_word_embeddings=config.values.get("tie_word_embeddings", False),
        attention_dropout=0.0,
    )


def mapped_reference_state(
    reference: MistralForCausalLM,
) -> dict[str, torch.Tensor]:
    return {
        (
            parameter_name
            if parameter_name == "lm_head.weight"
            else parameter_name.removeprefix("model.")
        ): parameter
        for parameter_name, parameter in reference.state_dict().items()
    }


def make_model_pair(
    config: ModelConfig,
) -> tuple[MistralModel, MistralForCausalLM]:
    torch.manual_seed(0)
    reference = MistralForCausalLM(make_reference_config(config)).eval()
    model = MistralModel(config)
    model.load_state_dict(mapped_reference_state(reference))
    return model, reference


def make_layer_caches(
    config: ModelConfig,
    *,
    block_size: int,
    num_blocks: int,
) -> tuple[torch.Tensor, ...]:
    num_attention_heads = config.require_positive_int("num_attention_heads")
    num_key_value_heads = config.positive_int(
        "num_key_value_heads",
        num_attention_heads,
    )
    head_dim = config.require_positive_int("hidden_size") // num_attention_heads
    return tuple(
        torch.zeros(
            2,
            num_blocks,
            block_size,
            num_key_value_heads,
            head_dim,
        )
        for _ in range(config.require_positive_int("num_hidden_layers"))
    )


def run_chunk(
    model: MistralModel,
    layer_caches: tuple[torch.Tensor, ...],
    token_ids: tuple[int, ...],
    first_position: int,
    *,
    block_size: int,
    block_table: torch.Tensor,
) -> torch.Tensor:
    positions = torch.arange(
        first_position,
        first_position + len(token_ids),
        dtype=torch.int64,
    )
    metadata = model.layer_attentions[0].build_metadata(
        query_positions=positions,
        query_start_locations=torch.tensor(
            [0, len(token_ids)],
            dtype=torch.int64,
        ),
        context_lengths=torch.tensor(
            [first_position + len(token_ids)],
            dtype=torch.int64,
        ),
        block_table=block_table.unsqueeze(0),
        slot_mapping=positions,
        block_size=block_size,
    )
    return model(
        torch.tensor(token_ids, dtype=torch.int64),
        positions,
        layer_caches,
        (metadata,) * len(layer_caches),
    )


def test_mistral_initializes_sliding_attention_for_every_layer() -> None:
    model = MistralModel(
        make_config(
            vocab_size=64,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=3,
            num_attention_heads=6,
            num_key_value_heads=2,
            sliding_window=8,
            rms_norm_eps=1e-5,
            rope_theta=500_000.0,
            tie_word_embeddings=True,
        )
    )

    attention = model.layer_attentions[0]
    assert len(model.layers) == 3
    assert model.embed_tokens.weight.shape == (64, 24)
    assert model.layer_attentions == (attention, attention, attention)
    assert attention.spec == SlidingWindowSpec(8)
    assert all(
        layer.self_attn.attention is attention for layer in model.layers
    )
    assert model.layers[0].self_attn.num_key_value_heads == 2
    assert model.layers[0].self_attn.rotary_embedding.base == 500_000.0
    assert model.norm.variance_epsilon == 1e-5
    assert model.embed_tokens.weight is model.lm_head.weight


def test_mistral_rejects_incompatible_attention_head_counts() -> None:
    with pytest.raises(
        ValueError,
        match="num_attention_heads must be divisible by num_key_value_heads",
    ):
        MistralModel(
            make_config(num_attention_heads=3, num_key_value_heads=2)
        )


def test_mistral_load_weights_copies_checkpoint_parameters() -> None:
    model = MistralModel(make_config())
    expected: dict[str, torch.Tensor] = {}
    checkpoint: dict[str, torch.Tensor] = {}
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
        checkpoint[checkpoint_name] = tensor
        expected[parameter_name] = tensor

    model.load_weights(checkpoint.items())

    for parameter_name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter,
            expected[parameter_name].to(parameter.dtype),
        )


@pytest.mark.parametrize(
    "chunk_sizes",
    [
        pytest.param((6,), id="full-prefill"),
        pytest.param((2, 2, 2), id="chunked-prefill"),
        pytest.param((4, 1, 1), id="prefill-and-decode"),
    ],
)
@torch.no_grad()
def test_mistral_hidden_and_logits_match_transformers(
    chunk_sizes: tuple[int, ...],
) -> None:
    config = make_config()
    model, reference = make_model_pair(config)
    token_ids = (1, 3, 5, 7, 9, 11)
    block_size = 2
    block_table = torch.arange(3, dtype=torch.int64)
    layer_caches = make_layer_caches(
        config,
        block_size=block_size,
        num_blocks=3,
    )
    first_position = 0
    hidden_states = torch.empty(0)
    for chunk_size in chunk_sizes:
        hidden_states = run_chunk(
            model,
            layer_caches,
            token_ids[first_position : first_position + chunk_size],
            first_position,
            block_size=block_size,
            block_table=block_table,
        )
        first_position += chunk_size

    reference_hidden_states = reference.model(
        torch.tensor([token_ids], dtype=torch.int64),
        use_cache=False,
    ).last_hidden_state.squeeze(0)
    logits = model.compute_logits(hidden_states[-1:])
    reference_logits = reference.lm_head(reference_hidden_states[-1:])

    torch.testing.assert_close(
        hidden_states[-1],
        reference_hidden_states[-1],
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        logits,
        reference_logits,
        atol=1e-5,
        rtol=1e-5,
    )
