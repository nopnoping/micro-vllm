"""Integration tests for locally generated HF-format Llama checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

from safetensors import safe_open
import torch
from transformers import LlamaForCausalLM


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


def _read_architectures(checkpoint_path: Path) -> object:
    with (checkpoint_path / "config.json").open(encoding="utf-8") as config_file:
        return json.load(config_file)["architectures"]


def _checkpoint_weight_names(checkpoint_path: Path) -> set[str]:
    weight_names: set[str] = set()
    for checkpoint_file in checkpoint_path.glob("*.safetensors"):
        with safe_open(checkpoint_file, framework="pt", device="cpu") as checkpoint:
            weight_names.update(checkpoint.keys())
    return weight_names
