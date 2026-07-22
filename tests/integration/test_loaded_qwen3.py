"""Integration tests for a locally generated HF-format Qwen3 checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

from safetensors import safe_open
import pytest
import torch
from transformers import Qwen3ForCausalLM

from micro_vllm.attention.attention import AttentionMetadata, map_cache_slots
from micro_vllm.config import LLMConfig, ModelConfig
from micro_vllm.model_loader import load_model
from micro_vllm.models.qwen import Qwen3Model


_CPU_ATOL = 1e-5
_CPU_RTOL = 1e-5
_GPU_FP16_ATOL = 4e-3
_GPU_FP16_RTOL = 4e-3


def test_tiny_hf_qwen3_checkpoint_uses_standard_layout(
    tiny_hf_qwen3_checkpoint: tuple[Qwen3ForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_qwen3_checkpoint

    assert reference_model.training is False
    assert (checkpoint_path / "model.safetensors").is_file()
    assert not (checkpoint_path / "model.safetensors.index.json").exists()
    with (checkpoint_path / "config.json").open(encoding="utf-8") as config_file:
        assert json.load(config_file)["architectures"] == [
            "Qwen3ForCausalLM"
        ]
    with safe_open(
        checkpoint_path / "model.safetensors",
        framework="pt",
        device="cpu",
    ) as checkpoint:
        weight_names = set(checkpoint.keys())
    assert {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.mlp.gate_proj.weight",
    } <= weight_names


def test_load_model_restores_qwen3_parameters_and_tied_identity(
    tiny_hf_qwen3_checkpoint: tuple[Qwen3ForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_qwen3_checkpoint

    loaded_model = load_model(
        LLMConfig(model_config=ModelConfig(model=checkpoint_path))
    )

    assert isinstance(loaded_model, Qwen3Model)
    assert loaded_model.training is False
    assert loaded_model.embed_tokens.weight is loaded_model.lm_head.weight
    actual_state = loaded_model.state_dict()
    expected_state = _mapped_reference_state(reference_model)
    assert set(actual_state) == set(expected_state)
    for parameter_name, actual_parameter in actual_state.items():
        torch.testing.assert_close(
            actual_parameter,
            expected_state[parameter_name],
            atol=0.0,
            rtol=0.0,
        )


@torch.no_grad()
def test_loaded_qwen3_hidden_states_and_logits_match_transformers_cpu(
    tiny_hf_qwen3_checkpoint: tuple[Qwen3ForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_qwen3_checkpoint
    loaded_model = load_model(
        LLMConfig(model_config=ModelConfig(model=checkpoint_path))
    )
    assert isinstance(loaded_model, Qwen3Model)

    values = _run_model_pair(
        loaded_model,
        reference_model,
        device="cpu",
        dtype=torch.float32,
    )
    hidden_states, logits, reference_hidden_states, reference_logits = values
    torch.testing.assert_close(
        hidden_states,
        reference_hidden_states,
        atol=_CPU_ATOL,
        rtol=_CPU_RTOL,
    )
    torch.testing.assert_close(
        logits,
        reference_logits,
        atol=_CPU_ATOL,
        rtol=_CPU_RTOL,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@torch.no_grad()
def test_loaded_qwen3_hidden_states_and_logits_match_transformers_gpu_fp16(
    tiny_hf_qwen3_checkpoint: tuple[Qwen3ForCausalLM, Path],
) -> None:
    reference_model, checkpoint_path = tiny_hf_qwen3_checkpoint
    loaded_model = load_model(
        LLMConfig(model_config=ModelConfig(model=checkpoint_path))
    )
    assert isinstance(loaded_model, Qwen3Model)

    values = _run_model_pair(
        loaded_model,
        reference_model,
        device="cuda",
        dtype=torch.float16,
    )
    hidden_states, logits, reference_hidden_states, reference_logits = values
    torch.testing.assert_close(
        hidden_states,
        reference_hidden_states,
        atol=_GPU_FP16_ATOL,
        rtol=_GPU_FP16_RTOL,
    )
    torch.testing.assert_close(
        logits,
        reference_logits,
        atol=_GPU_FP16_ATOL,
        rtol=_GPU_FP16_RTOL,
    )


def _run_model_pair(
    loaded_model: Qwen3Model,
    reference_model: Qwen3ForCausalLM,
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
) -> AttentionMetadata:
    block_size = 2
    physical_blocks = torch.tensor([2, 0], dtype=torch.int64, device=device)
    return AttentionMetadata(
        query_positions=positions,
        query_context_start_positions=torch.zeros_like(positions),
        query_start_locations=torch.tensor(
            [0, 4],
            dtype=torch.int64,
            device=device,
        ),
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
    num_attention_heads = config.require_positive_int("num_attention_heads")
    num_key_value_heads = config.positive_int(
        "num_key_value_heads",
        num_attention_heads,
    )
    head_dim = config.require_positive_int("head_dim")
    return tuple(
        torch.zeros(
            (2, 3, 2, num_key_value_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        for _ in range(config.require_positive_int("num_hidden_layers"))
    )


def _mapped_reference_state(
    reference_model: Qwen3ForCausalLM,
) -> dict[str, torch.Tensor]:
    return {
        (
            parameter_name
            if parameter_name == "lm_head.weight"
            else parameter_name.removeprefix("model.")
        ): parameter
        for parameter_name, parameter in reference_model.state_dict().items()
    }
