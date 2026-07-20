"""Tests for flattened Llama decoder and model execution."""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from micro_vllm.layers.attention_backend import (
    AttentionMetadata,
    ReferenceAttentionBackend,
)
from micro_vllm.models.llama import LlamaModel, LlamaModelConfig


class RecordingZeroBackend:
    def __init__(self) -> None:
        self.layer_caches: list[torch.Tensor] = []

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_kv_cache: torch.Tensor,
        metadata: AttentionMetadata,
        /,
    ) -> torch.Tensor:
        self.layer_caches.append(layer_kv_cache)
        return torch.zeros_like(query)


def make_config(**overrides: object) -> LlamaModelConfig:
    values = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
    }
    values.update(overrides)
    return LlamaModelConfig(**values)


def make_metadata(
    num_tokens: int,
    *,
    block_table: list[int],
    block_size: int,
) -> AttentionMetadata:
    positions = torch.arange(num_tokens, dtype=torch.int64)
    slots = torch.tensor(
        [
            block_table[position // block_size] * block_size
            + position % block_size
            for position in range(num_tokens)
        ],
        dtype=torch.int64,
    )
    return AttentionMetadata(
        query_positions=positions,
        query_start_locations=torch.tensor([0, num_tokens], dtype=torch.int64),
        context_lengths=torch.tensor([num_tokens], dtype=torch.int64),
        block_table=torch.tensor([block_table], dtype=torch.int64),
        slot_mapping=slots,
        block_size=block_size,
    )


def make_layer_caches(
    config: LlamaModelConfig,
    *,
    num_blocks: int,
    block_size: int,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.zeros(
            (
                2,
                num_blocks,
                block_size,
                config.num_key_value_heads,
                config.head_dim,
            )
        )
        for _ in range(config.num_hidden_layers)
    )


def test_llama_model_uses_one_physical_cache_per_layer() -> None:
    torch.manual_seed(0)
    config = make_config()
    backend = RecordingZeroBackend()
    model = LlamaModel(config, backend)
    input_ids = torch.tensor([1, 3, 5], dtype=torch.int64)
    positions = torch.arange(3, dtype=torch.int64)
    metadata = make_metadata(3, block_table=[0], block_size=4)
    layer_caches = make_layer_caches(config, num_blocks=2, block_size=4)

    hidden_states = model(
        input_ids,
        positions,
        layer_caches,
        metadata,
    )

    assert hidden_states.shape == torch.Size((3, config.hidden_size))
    assert len(backend.layer_caches) == len(layer_caches)
    assert all(
        received is expected
        for received, expected in zip(backend.layer_caches, layer_caches)
    )
    assert model.training is False
    assert all(layer.training is False for layer in model.layers)


def test_forward_and_compute_logits_are_separate() -> None:
    torch.manual_seed(0)
    config = make_config(num_hidden_layers=1)
    model = LlamaModel(config, RecordingZeroBackend())
    input_ids = torch.tensor([2, 4, 6], dtype=torch.int64)
    positions = torch.arange(3, dtype=torch.int64)
    metadata = make_metadata(3, block_table=[0], block_size=4)
    layer_caches = make_layer_caches(config, num_blocks=1, block_size=4)
    lm_head_inputs: list[torch.Tensor] = []
    hook = model.lm_head.register_forward_pre_hook(
        lambda _module, inputs: lm_head_inputs.append(inputs[0])
    )

    hidden_states = model(
        input_ids,
        positions,
        layer_caches,
        metadata,
    )
    assert lm_head_inputs == []

    sampling_hidden_states = hidden_states.index_select(
        0,
        torch.tensor([0, 2], dtype=torch.int64),
    )
    logits = model.compute_logits(sampling_hidden_states)
    hook.remove()

    assert len(lm_head_inputs) == 1
    assert lm_head_inputs[0] is sampling_hidden_states
    assert logits.shape == torch.Size((2, config.vocab_size))


def test_tiny_llama_hidden_states_and_logits_match_transformers() -> None:
    torch.manual_seed(0)
    config = make_config()
    model = LlamaModel(config, ReferenceAttentionBackend())
    reference_config = LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        hidden_act=config.hidden_act,
        tie_word_embeddings=config.tie_word_embeddings,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
    )
    reference = LlamaForCausalLM(reference_config).eval()
    model_state = model.state_dict()
    reference.model.load_state_dict(
        {
            name: parameter
            for name, parameter in model_state.items()
            if name != "lm_head.weight"
        }
    )
    reference.lm_head.load_state_dict(
        {"weight": model_state["lm_head.weight"]}
    )
    input_ids = torch.tensor([1, 3, 5, 7], dtype=torch.int64)
    positions = torch.arange(4, dtype=torch.int64)
    metadata = make_metadata(4, block_table=[2, 0], block_size=2)
    layer_caches = make_layer_caches(config, num_blocks=3, block_size=2)

    with torch.no_grad():
        hidden_states = model(
            input_ids,
            positions,
            layer_caches,
            metadata,
        )
        reference_hidden_states = reference.model(
            input_ids.unsqueeze(0),
            position_ids=positions.unsqueeze(0),
            use_cache=False,
        ).last_hidden_state.squeeze(0)

    torch.testing.assert_close(
        hidden_states,
        reference_hidden_states,
        atol=1e-5,
        rtol=1e-5,
    )

    sampling_rows = torch.tensor([1, 3], dtype=torch.int64)
    logits = model.compute_logits(hidden_states.index_select(0, sampling_rows))
    reference_logits = reference.lm_head(
        reference_hidden_states.index_select(0, sampling_rows)
    )
    torch.testing.assert_close(logits, reference_logits, atol=1e-5, rtol=1e-5)
