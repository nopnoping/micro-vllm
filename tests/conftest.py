"""Shared test configuration for micro-vLLM."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)


_MODEL_TEST_ENV = "MICRO_VLLM_TEST_MODEL"
_TINY_LLAMA_SEED = 0
_TINY_QWEN3_SEED = 1


@pytest.fixture
def tiny_hf_llama_checkpoint(
    tmp_path: Path,
) -> tuple[LlamaForCausalLM, Path]:
    """Create one deterministic, single-file HF Llama checkpoint."""
    return _save_tiny_hf_llama_checkpoint(tmp_path / "tiny-llama")


@pytest.fixture
def sharded_tiny_hf_llama_checkpoint(
    tmp_path: Path,
) -> tuple[LlamaForCausalLM, Path]:
    """Create one deterministic HF Llama checkpoint with multiple shards."""
    return _save_tiny_hf_llama_checkpoint(
        tmp_path / "sharded-tiny-llama",
        max_shard_size="5KB",
    )


@pytest.fixture
def tied_tiny_hf_llama_checkpoint(
    tmp_path: Path,
) -> tuple[LlamaForCausalLM, Path]:
    """Create one deterministic HF Llama checkpoint with tied embeddings."""
    return _save_tiny_hf_llama_checkpoint(
        tmp_path / "tied-tiny-llama",
        tie_word_embeddings=True,
    )


@pytest.fixture
def tiny_hf_qwen3_checkpoint(
    tmp_path: Path,
) -> tuple[Qwen3ForCausalLM, Path]:
    """Create one deterministic, single-file HF Qwen3 checkpoint."""
    checkpoint_path = tmp_path / "tiny-qwen3"
    config = Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        rope_theta=1_000_000.0,
        hidden_act="silu",
        tie_word_embeddings=True,
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=False,
    )
    with torch.random.fork_rng():
        torch.manual_seed(_TINY_QWEN3_SEED)
        reference_model = Qwen3ForCausalLM(config).eval()
    reference_model.save_pretrained(
        checkpoint_path,
        safe_serialization=True,
    )
    return reference_model, checkpoint_path


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Keep external model access opt-in for every pytest invocation."""
    if os.environ.get(_MODEL_TEST_ENV) == "1":
        return

    skip_model = pytest.mark.skip(
        reason=f"set {_MODEL_TEST_ENV}=1 to run external model tests",
    )
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip_model)


def _save_tiny_hf_llama_checkpoint(
    checkpoint_path: Path,
    *,
    max_shard_size: str | None = None,
    tie_word_embeddings: bool = False,
) -> tuple[LlamaForCausalLM, Path]:
    config = LlamaConfig(
        architectures=["LlamaForCausalLM"],
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        hidden_act="silu",
        tie_word_embeddings=tie_word_embeddings,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
    )
    with torch.random.fork_rng():
        torch.manual_seed(_TINY_LLAMA_SEED)
        reference_model = LlamaForCausalLM(config).eval()

    if max_shard_size is None:
        reference_model.save_pretrained(
            checkpoint_path,
            safe_serialization=True,
        )
    else:
        reference_model.save_pretrained(
            checkpoint_path,
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )
    return reference_model, checkpoint_path
