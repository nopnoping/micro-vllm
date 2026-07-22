"""Tests for flattened Llama decoder and model execution."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from micro_vllm.attention.attention import (
    Attention,
    AttentionMetadata,
    map_cache_slots,
)
from micro_vllm.attention.operators.base import AttentionOperator
from micro_vllm.config import ModelConfig
from micro_vllm.models.llama import LlamaModel

NULL_BLOCK_ID = 0


class RecordingZeroOperator(AttentionOperator):
    def __init__(self) -> None:
        self.layer_caches: list[torch.Tensor] = []

    def forward(
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


def make_config(**overrides: object) -> ModelConfig:
    values = {
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
    }
    values.update(overrides)
    return ModelConfig(model="unused", metadata=values)


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
        query_context_start_positions=torch.zeros_like(positions),
        query_start_locations=torch.tensor([0, num_tokens], dtype=torch.int64),
        context_lengths=torch.tensor([num_tokens], dtype=torch.int64),
        block_table=torch.tensor([block_table], dtype=torch.int64),
        slot_mapping=slots,
        block_size=block_size,
    )


def make_layer_caches(
    config: ModelConfig,
    *,
    num_blocks: int,
    block_size: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.zeros(
            (
                2,
                num_blocks,
                block_size,
                config.positive_int(
                    "num_key_value_heads",
                    config.require_positive_int("num_attention_heads"),
                ),
                config.require_positive_int("hidden_size")
                // config.require_positive_int("num_attention_heads"),
            ),
            device=device,
            dtype=dtype,
        )
        for _ in range(config.require_positive_int("num_hidden_layers"))
    )


def make_reference_config(config: ModelConfig) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=config.require_positive_int("vocab_size"),
        hidden_size=config.require_positive_int("hidden_size"),
        intermediate_size=config.require_positive_int("intermediate_size"),
        num_hidden_layers=config.require_positive_int("num_hidden_layers"),
        num_attention_heads=config.require_positive_int("num_attention_heads"),
        num_key_value_heads=config.positive_int(
            "num_key_value_heads",
            config.require_positive_int("num_attention_heads"),
        ),
        max_position_embeddings=config.require_positive_int("max_position_embeddings"),
        rms_norm_eps=float(config.values.get("rms_norm_eps", 1e-6)),
        rope_theta=float(config.values.get("rope_theta", 10_000.0)),
        hidden_act=config.values.get("hidden_act", "silu"),
        tie_word_embeddings=config.values.get("tie_word_embeddings", False),
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
    )


def map_from_transformers_state_dict(
    reference: LlamaForCausalLM,
) -> dict[str, torch.Tensor]:
    return {
        (
            name
            if name == "lm_head.weight"
            else name.removeprefix("model.")
        ): parameter
        for name, parameter in reference.state_dict().items()
    }


def make_model_pair(
    config: ModelConfig,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[LlamaModel, LlamaForCausalLM]:
    torch.manual_seed(0)
    reference = LlamaForCausalLM(make_reference_config(config)).eval()
    model = LlamaModel(config)
    model.load_state_dict(map_from_transformers_state_dict(reference))
    return (
        model.to(device=device, dtype=dtype),
        reference.to(device=device, dtype=dtype),
    )


def test_llama_model_uses_one_physical_cache_per_layer() -> None:
    torch.manual_seed(0)
    config = make_config()
    operator = RecordingZeroOperator()
    model = LlamaModel(config)
    model.layer_attentions[0].operator = operator
    input_ids = torch.tensor([1, 3, 5], dtype=torch.int64)
    positions = torch.arange(3, dtype=torch.int64)
    metadata = make_metadata(3, block_table=[0], block_size=4)
    layer_caches = make_layer_caches(config, num_blocks=2, block_size=4)

    hidden_states = model(
        input_ids,
        positions,
        layer_caches,
        (metadata,) * len(layer_caches),
    )

    assert hidden_states.shape == torch.Size(
        (3, config.require_positive_int("hidden_size"))
    )
    assert len(operator.layer_caches) == len(layer_caches)
    assert all(
        received is expected
        for received, expected in zip(operator.layer_caches, layer_caches)
    )
    assert model.training is False
    assert all(layer.training is False for layer in model.layers)


def test_forward_and_compute_logits_are_separate() -> None:
    torch.manual_seed(0)
    config = make_config(num_hidden_layers=1)
    model = LlamaModel(config)
    model.layer_attentions[0].operator = RecordingZeroOperator()
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
        (metadata,) * len(layer_caches),
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
    assert logits.shape == torch.Size(
        (2, config.require_positive_int("vocab_size"))
    )


def test_gqa_tiny_llama_layers_hidden_states_and_logits_match_transformers() -> None:
    config = make_config()
    model, reference = make_model_pair(config)
    input_ids = torch.tensor([1, 3, 5, 7], dtype=torch.int64)
    positions = torch.arange(4, dtype=torch.int64)
    metadata = make_metadata(4, block_table=[2, 0], block_size=2)
    layer_caches = make_layer_caches(config, num_blocks=3, block_size=2)
    attention_outputs: list[torch.Tensor] = []
    reference_attention_outputs: list[torch.Tensor] = []
    layer_outputs: list[torch.Tensor] = []
    reference_layer_outputs: list[torch.Tensor] = []
    hooks = (
        model.layers[0].self_attn.register_forward_hook(
            lambda _module, _inputs, output: attention_outputs.append(output)
        ),
        reference.model.layers[0].self_attn.register_forward_hook(
            lambda _module, _inputs, output: reference_attention_outputs.append(
                output[0]
            )
        ),
        model.layers[0].register_forward_hook(
            lambda _module, _inputs, output: layer_outputs.append(output)
        ),
        reference.model.layers[0].register_forward_hook(
            lambda _module, _inputs, output: reference_layer_outputs.append(output)
        ),
    )

    with torch.no_grad():
        hidden_states = model(
            input_ids,
            positions,
            layer_caches,
            (metadata,) * len(layer_caches),
        )
        reference_hidden_states = reference.model(
            input_ids.unsqueeze(0),
            position_ids=positions.unsqueeze(0),
            use_cache=False,
        ).last_hidden_state.squeeze(0)
    for hook in hooks:
        hook.remove()

    torch.testing.assert_close(
        attention_outputs[0],
        reference_attention_outputs[0].squeeze(0),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        layer_outputs[0],
        reference_layer_outputs[0].squeeze(0),
        atol=1e-5,
        rtol=1e-5,
    )

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
    assert torch.equal(logits.argmax(dim=-1), reference_logits.argmax(dim=-1))


def test_transformers_parameter_names_add_only_the_model_prefix() -> None:
    model, reference = make_model_pair(make_config())

    mapped_state = map_from_transformers_state_dict(reference)

    assert set(mapped_state) == set(model.state_dict())
    assert "model.embed_tokens.weight" in reference.state_dict()
    assert "embed_tokens.weight" in mapped_state
    assert "lm_head.weight" in mapped_state


def make_batch_metadata(
    positions_by_request: tuple[tuple[int, ...], ...],
    block_tables: tuple[tuple[int, ...], ...],
    *,
    block_size: int,
    device: torch.device | str = "cpu",
) -> AttentionMetadata:
    position_tensors = tuple(
        torch.tensor(positions, dtype=torch.int64, device=device)
        for positions in positions_by_request
    )
    max_num_blocks = max(len(blocks) for blocks in block_tables)
    block_table = torch.full(
        (len(block_tables), max_num_blocks),
        NULL_BLOCK_ID,
        dtype=torch.int64,
        device=device,
    )
    for request_index, blocks in enumerate(block_tables):
        block_table[request_index, : len(blocks)] = torch.tensor(
            blocks,
            dtype=torch.int64,
            device=device,
        )

    query_start_locations = [0]
    for positions in positions_by_request:
        query_start_locations.append(query_start_locations[-1] + len(positions))
    query_positions = torch.cat(position_tensors)
    return AttentionMetadata(
        query_positions=query_positions,
        query_context_start_positions=torch.zeros_like(query_positions),
        query_start_locations=torch.tensor(
            query_start_locations,
            dtype=torch.int64,
            device=device,
        ),
        context_lengths=torch.tensor(
            [positions[-1] + 1 for positions in positions_by_request],
            dtype=torch.int64,
            device=device,
        ),
        block_table=block_table,
        slot_mapping=torch.cat(
            tuple(
                map_cache_slots(
                    positions,
                    block_table[request_index],
                    block_size,
                )
                for request_index, positions in enumerate(position_tensors)
            )
        ),
        block_size=block_size,
    )


def run_batch(
    model: LlamaModel,
    layer_caches: tuple[torch.Tensor, ...],
    input_ids_by_request: tuple[tuple[int, ...], ...],
    positions_by_request: tuple[tuple[int, ...], ...],
    block_tables: tuple[tuple[int, ...], ...],
    *,
    block_size: int,
) -> torch.Tensor:
    device = layer_caches[0].device
    metadata = make_batch_metadata(
        positions_by_request,
        block_tables,
        block_size=block_size,
        device=device,
    )
    input_ids = torch.tensor(
        [token_id for request in input_ids_by_request for token_id in request],
        dtype=torch.int64,
        device=device,
    )
    return model(
        input_ids,
        metadata.query_positions,
        layer_caches,
        (metadata,) * len(layer_caches),
    )


def clone_model(model: LlamaModel, config: ModelConfig) -> LlamaModel:
    parameter = next(model.parameters())
    clone = LlamaModel(config).to(
        device=parameter.device,
        dtype=parameter.dtype,
    )
    clone.load_state_dict(model.state_dict())
    return clone


def test_chunked_prefill_last_logits_match_full_prefill_and_transformers() -> None:
    config = make_config()
    full_model, reference = make_model_pair(config)
    chunked_model = clone_model(full_model, config)
    tokens = (1, 3, 5, 7, 9)
    blocks = (3, 0, 5)
    block_size = 2
    full_caches = make_layer_caches(
        config,
        num_blocks=6,
        block_size=block_size,
    )
    chunked_caches = make_layer_caches(
        config,
        num_blocks=6,
        block_size=block_size,
    )

    with torch.inference_mode():
        full_hidden_states = run_batch(
            full_model,
            full_caches,
            (tokens,),
            (tuple(range(len(tokens))),),
            (blocks,),
            block_size=block_size,
        )
        run_batch(
            chunked_model,
            chunked_caches,
            (tokens[:2],),
            ((0, 1),),
            (blocks,),
            block_size=block_size,
        )
        chunked_hidden_states = run_batch(
            chunked_model,
            chunked_caches,
            (tokens[2:],),
            ((2, 3, 4),),
            (blocks,),
            block_size=block_size,
        )
        full_logits = full_model.compute_logits(full_hidden_states[-1:])
        chunked_logits = chunked_model.compute_logits(chunked_hidden_states[-1:])
        reference_logits = reference(
            torch.tensor([tokens], dtype=torch.int64),
            use_cache=False,
        ).logits[:, -1]

    torch.testing.assert_close(full_logits, reference_logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(chunked_logits, full_logits, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def incremental_logits(
    model: LlamaModel,
    config: ModelConfig,
    tokens: tuple[int, ...],
    chunk_sizes: tuple[int, ...],
) -> torch.Tensor:
    block_size = 2
    blocks = (2, 5, 1)
    layer_caches = make_layer_caches(
        config,
        num_blocks=6,
        block_size=block_size,
    )
    token_start = 0
    hidden_states = torch.empty(0)
    for chunk_size in chunk_sizes:
        token_end = token_start + chunk_size
        hidden_states = run_batch(
            model,
            layer_caches,
            (tokens[token_start:token_end],),
            (tuple(range(token_start, token_end)),),
            (blocks,),
            block_size=block_size,
        )
        token_start = token_end
    return model.compute_logits(hidden_states[-1:])


@pytest.mark.parametrize(
    "chunk_sizes",
    [
        pytest.param((4, 1), id="prefill-then-one-token-decode"),
        pytest.param((2, 1, 1, 1), id="prefill-then-multiple-decodes"),
    ],
)
@torch.no_grad()
def test_incremental_decode_matches_full_sequence(
    chunk_sizes: tuple[int, ...],
) -> None:
    config = make_config()
    model, reference = make_model_pair(config)
    tokens = (2, 4, 6, 8, 10)

    logits = incremental_logits(model, config, tokens, chunk_sizes)
    reference_logits = reference(
        torch.tensor([tokens], dtype=torch.int64),
        use_cache=False,
    ).logits[:, -1]

    torch.testing.assert_close(logits, reference_logits, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_mixed_decode_batch_matches_independent_requests() -> None:
    config = make_config()
    mixed_model, _ = make_model_pair(config)
    block_size = 2
    mixed_caches = make_layer_caches(
        config,
        num_blocks=8,
        block_size=block_size,
    )
    block_tables = ((6, 1), (4, 7))
    run_batch(
        mixed_model,
        mixed_caches,
        ((1, 2), (3, 4, 5)),
        ((0, 1), (0, 1, 2)),
        block_tables,
        block_size=block_size,
    )
    mixed_hidden_states = run_batch(
        mixed_model,
        mixed_caches,
        ((6,), (7,)),
        ((2,), (3,)),
        block_tables,
        block_size=block_size,
    )
    mixed_logits = mixed_model.compute_logits(mixed_hidden_states)

    independent_logits: list[torch.Tensor] = []
    for tokens, blocks in (
        ((1, 2, 6), block_tables[0]),
        ((3, 4, 5, 7), block_tables[1]),
    ):
        model = clone_model(mixed_model, config)
        layer_caches = make_layer_caches(
            config,
            num_blocks=8,
            block_size=block_size,
        )
        hidden_states = run_batch(
            model,
            layer_caches,
            (tokens,),
            (tuple(range(len(tokens))),),
            (blocks,),
            block_size=block_size,
        )
        independent_logits.append(model.compute_logits(hidden_states[-1:]))

    torch.testing.assert_close(
        mixed_logits,
        torch.cat(independent_logits),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@torch.no_grad()
def test_gpu_fp16_hidden_states_and_logits_match_transformers() -> None:
    config = make_config()
    model, reference = make_model_pair(
        config,
        device="cuda",
        dtype=torch.float16,
    )
    input_ids = torch.tensor([1, 3, 5, 7], dtype=torch.int64, device="cuda")
    positions = torch.arange(4, dtype=torch.int64, device="cuda")
    metadata = make_batch_metadata(
        ((0, 1, 2, 3),),
        ((2, 0),),
        block_size=2,
        device="cuda",
    )
    layer_caches = make_layer_caches(
        config,
        num_blocks=3,
        block_size=2,
        device="cuda",
        dtype=torch.float16,
    )

    hidden_states = model(
        input_ids,
        positions,
        layer_caches,
        (metadata,) * len(layer_caches),
    )
    reference_hidden_states = reference.model(
        input_ids.unsqueeze(0),
        position_ids=positions.unsqueeze(0),
        use_cache=False,
    ).last_hidden_state.squeeze(0)
    logits = model.compute_logits(hidden_states[-1:])
    reference_logits = reference.lm_head(reference_hidden_states[-1:])

    torch.testing.assert_close(
        hidden_states,
        reference_hidden_states,
        atol=3e-3,
        rtol=3e-3,
    )
    torch.testing.assert_close(
        logits,
        reference_logits,
        atol=3e-3,
        rtol=3e-3,
    )
