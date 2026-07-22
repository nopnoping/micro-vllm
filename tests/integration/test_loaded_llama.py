"""Integration tests for locally generated HF-format Llama checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from safetensors import safe_open
import torch
from transformers import LlamaForCausalLM

from micro_vllm.attention.base import map_cache_slots
from micro_vllm.attention.full.metadata import FullAttentionMetadata
from micro_vllm.config import LLMConfig, ModelConfig
from micro_vllm.model_loader import load_model
from micro_vllm.models.llama import LlamaModel


_CPU_ATOL = 1e-5
_CPU_RTOL = 1e-5
_GPU_FP16_ATOL = 3e-3
_GPU_FP16_RTOL = 3e-3


def test_tiny_hf_llama_checkpoint_uses_standard_single_file_layout(
    tiny_hf_llama_checkpoint: tuple[LlamaForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_llama_checkpoint

    assert reference_model.training is False
    assert (checkpoint_path / "model.safetensors").is_file()
    assert not (checkpoint_path / "model.safetensors.index.json").exists()
    assert _read_architectures(checkpoint_path) == ["LlamaForCausalLM"]
    weight_names = _checkpoint_weight_names(checkpoint_path)
    assert weight_names == set(reference_model.state_dict())
    assert {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "lm_head.weight",
    } <= weight_names


def test_sharded_tiny_hf_llama_checkpoint_uses_standard_index_layout(
    sharded_tiny_hf_llama_checkpoint: tuple[LlamaForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = sharded_tiny_hf_llama_checkpoint
    index_path = checkpoint_path / "model.safetensors.index.json"

    assert reference_model.training is False
    assert not (checkpoint_path / "model.safetensors").exists()
    assert index_path.is_file()
    assert _read_architectures(checkpoint_path) == ["LlamaForCausalLM"]

    with index_path.open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file)["weight_map"]

    shard_names = set(weight_map.values())
    assert len(shard_names) > 1
    assert all(
        (checkpoint_path / shard_name).is_file()
        for shard_name in shard_names
    )
    assert set(weight_map) == set(reference_model.state_dict())
    assert _checkpoint_weight_names(checkpoint_path) == set(weight_map)


def test_tiny_hf_llama_checkpoint_fixtures_are_deterministic(
    tiny_hf_llama_checkpoint: tuple[LlamaForCausalLM, Path],
    sharded_tiny_hf_llama_checkpoint: tuple[LlamaForCausalLM, Path],
) -> None:
    single_file_model, _ = tiny_hf_llama_checkpoint
    sharded_model, _ = sharded_tiny_hf_llama_checkpoint
    single_file_state = single_file_model.state_dict()
    sharded_state = sharded_model.state_dict()

    assert single_file_model is not sharded_model
    assert set(single_file_state) == set(sharded_state)
    for parameter_name, parameter in single_file_state.items():
        torch.testing.assert_close(
            parameter,
            sharded_state[parameter_name],
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize(
    "checkpoint_fixture",
    [
        "tiny_hf_llama_checkpoint",
        "sharded_tiny_hf_llama_checkpoint",
    ],
)
def test_load_model_restores_every_untied_parameter(
    checkpoint_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    reference_model, checkpoint_path = request.getfixturevalue(checkpoint_fixture)

    loaded_model = load_model(
        LLMConfig(model_config=ModelConfig(model=checkpoint_path))
    )

    assert isinstance(loaded_model, LlamaModel)
    assert loaded_model.training is False
    assert all(parameter.device.type == "cpu" for parameter in loaded_model.parameters())
    assert all(
        parameter.dtype == torch.get_default_dtype()
        for parameter in loaded_model.parameters()
    )

    expected_state = _mapped_reference_state(reference_model)
    actual_state = loaded_model.state_dict()
    assert set(actual_state) == set(expected_state)
    for parameter_name, actual_parameter in actual_state.items():
        _assert_checkpoint_close(
            actual_parameter,
            expected_state[parameter_name],
            checkpoint_path=checkpoint_path,
            value_name=parameter_name,
            atol=0.0,
            rtol=0.0,
        )


def test_load_model_preserves_tied_embedding_identity_and_value(
    tied_tiny_hf_llama_checkpoint: tuple[LlamaForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tied_tiny_hf_llama_checkpoint

    loaded_model = load_model(
        LLMConfig(model_config=ModelConfig(model=checkpoint_path))
    )

    assert isinstance(loaded_model, LlamaModel)
    assert loaded_model.embed_tokens.weight is loaded_model.lm_head.weight
    _assert_checkpoint_close(
        loaded_model.embed_tokens.weight,
        reference_model.model.embed_tokens.weight,
        checkpoint_path=checkpoint_path,
        value_name="tied embed_tokens.weight",
        atol=0.0,
        rtol=0.0,
    )


def test_load_model_returns_independent_loaded_instances(
    tiny_hf_llama_checkpoint: tuple[LlamaForCausalLM, Path],
) -> None:
    _, checkpoint_path = tiny_hf_llama_checkpoint

    first = load_model(LLMConfig(model_config=ModelConfig(model=checkpoint_path)))
    second = load_model(LLMConfig(model_config=ModelConfig(model=checkpoint_path)))

    assert isinstance(first, LlamaModel)
    assert isinstance(second, LlamaModel)
    assert first is not second
    first_parameters = dict(first.named_parameters())
    second_parameters = dict(second.named_parameters())
    assert set(first_parameters) == set(second_parameters)
    for parameter_name, first_parameter in first_parameters.items():
        second_parameter = second_parameters[parameter_name]
        assert first_parameter is not second_parameter
        _assert_checkpoint_close(
            first_parameter,
            second_parameter,
            checkpoint_path=checkpoint_path,
            value_name=parameter_name,
            atol=0.0,
            rtol=0.0,
        )


@torch.no_grad()
def test_loaded_checkpoint_hidden_states_and_logits_match_transformers_cpu(
    tiny_hf_llama_checkpoint: tuple[LlamaForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_llama_checkpoint
    loaded_model = load_model(
        LLMConfig(model_config=ModelConfig(model=checkpoint_path))
    )
    assert isinstance(loaded_model, LlamaModel)

    hidden_states, logits, reference_hidden_states, reference_logits = (
        _run_model_pair(
            loaded_model,
            reference_model,
            device="cpu",
            dtype=torch.float32,
        )
    )

    _assert_checkpoint_close(
        hidden_states,
        reference_hidden_states,
        checkpoint_path=checkpoint_path,
        value_name="final hidden states",
        atol=_CPU_ATOL,
        rtol=_CPU_RTOL,
    )
    _assert_checkpoint_close(
        logits,
        reference_logits,
        checkpoint_path=checkpoint_path,
        value_name="logits",
        atol=_CPU_ATOL,
        rtol=_CPU_RTOL,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@torch.no_grad()
def test_loaded_checkpoint_hidden_states_and_logits_match_transformers_gpu_fp16(
    tiny_hf_llama_checkpoint: tuple[LlamaForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_llama_checkpoint
    loaded_model = load_model(
        LLMConfig(model_config=ModelConfig(model=checkpoint_path))
    )
    assert isinstance(loaded_model, LlamaModel)

    hidden_states, logits, reference_hidden_states, reference_logits = (
        _run_model_pair(
            loaded_model,
            reference_model,
            device="cuda",
            dtype=torch.float16,
        )
    )

    _assert_checkpoint_close(
        hidden_states,
        reference_hidden_states,
        checkpoint_path=checkpoint_path,
        value_name="final hidden states",
        atol=_GPU_FP16_ATOL,
        rtol=_GPU_FP16_RTOL,
    )
    _assert_checkpoint_close(
        logits,
        reference_logits,
        checkpoint_path=checkpoint_path,
        value_name="logits",
        atol=_GPU_FP16_ATOL,
        rtol=_GPU_FP16_RTOL,
    )


def _run_model_pair(
    loaded_model: LlamaModel,
    reference_model: LlamaForCausalLM,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    loaded_model.to(device=device, dtype=dtype)
    reference_model.to(device=device, dtype=dtype)
    input_ids = torch.tensor([1, 3, 5, 7], dtype=torch.int64, device=device)
    positions = torch.arange(4, dtype=torch.int64, device=device)
    metadata = _make_attention_metadata(positions, device=device)
    layer_caches = _make_layer_caches(
        loaded_model.config,
        device=device,
        dtype=dtype,
    )

    hidden_states = loaded_model(
        input_ids,
        positions,
        layer_caches,
        metadata,
    )
    reference_hidden_states = reference_model.model(
        input_ids.unsqueeze(0),
        position_ids=positions.unsqueeze(0),
        use_cache=False,
    ).last_hidden_state.squeeze(0)
    logits = loaded_model.compute_logits(hidden_states)
    reference_logits = reference_model.lm_head(reference_hidden_states)
    return hidden_states, logits, reference_hidden_states, reference_logits


def _make_attention_metadata(
    positions: torch.Tensor,
    *,
    device: torch.device | str,
) -> FullAttentionMetadata:
    block_size = 2
    physical_blocks = torch.tensor([2, 0], dtype=torch.int64, device=device)
    return FullAttentionMetadata(
        query_positions=positions,
        query_start_locations=torch.tensor([0, 4], dtype=torch.int64, device=device),
        context_lengths=torch.tensor([4], dtype=torch.int64, device=device),
        block_table=physical_blocks.unsqueeze(0),
        slot_mapping=map_cache_slots(positions, physical_blocks, block_size),
        block_size=block_size,
    )


def _make_layer_caches(
    config: ModelConfig,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    hidden_size = config.require_positive_int("hidden_size")
    num_attention_heads = config.require_positive_int("num_attention_heads")
    num_key_value_heads = config.positive_int(
        "num_key_value_heads",
        num_attention_heads,
    )
    return tuple(
        torch.zeros(
            (2, 3, 2, num_key_value_heads, hidden_size // num_attention_heads),
            device=device,
            dtype=dtype,
        )
        for _ in range(config.require_positive_int("num_hidden_layers"))
    )


def _mapped_reference_state(
    reference_model: LlamaForCausalLM,
) -> dict[str, torch.Tensor]:
    return {
        (
            parameter_name
            if parameter_name == "lm_head.weight"
            else parameter_name.removeprefix("model.")
        ): parameter
        for parameter_name, parameter in reference_model.state_dict().items()
    }


def _assert_checkpoint_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    checkpoint_path: Path,
    value_name: str,
    atol: float,
    rtol: float,
) -> None:
    try:
        torch.testing.assert_close(
            actual,
            expected,
            atol=atol,
            rtol=rtol,
        )
    except AssertionError as error:
        max_error: float | str
        if actual.shape == expected.shape:
            max_error = torch.max(
                torch.abs(actual.to(torch.float32) - expected.to(torch.float32))
            ).item()
        else:
            max_error = "unavailable for different shapes"
        raise AssertionError(
            f"{value_name} mismatch for checkpoint {checkpoint_path}; "
            f"actual dtype={actual.dtype}, expected dtype={expected.dtype}, "
            f"actual shape={tuple(actual.shape)}, "
            f"expected shape={tuple(expected.shape)}, "
            f"max absolute error={max_error}, "
            f"atol={atol}, rtol={rtol}: {error}"
        ) from error


def _read_architectures(checkpoint_path: Path) -> object:
    with (checkpoint_path / "config.json").open(encoding="utf-8") as config_file:
        return json.load(config_file)["architectures"]


def _checkpoint_weight_names(checkpoint_path: Path) -> set[str]:
    weight_names: set[str] = set()
    for checkpoint_file in checkpoint_path.glob("*.safetensors"):
        with safe_open(checkpoint_file, framework="pt", device="cpu") as checkpoint:
            weight_names.update(checkpoint.keys())
    return weight_names
