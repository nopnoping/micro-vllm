"""Selection contract for the pinned public Qwen checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from huggingface_hub import HfApi, hf_hub_download


MODEL_REPO_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_CHECKPOINT_FILE = "model.safetensors"
MODEL_CHECKPOINT_DTYPE = "bfloat16"
MODEL_UNIQUE_PARAMETER_COUNT = 596_049_920
MODEL_CHECKPOINT_ELEMENT_COUNT = 751_632_384
MODEL_DOWNLOAD_SIZE_BYTES = 1_503_300_328
MODEL_VOCAB_SIZE = 151_936
MODEL_PROMPT_TOKEN_IDS = (1, 450, 7_483, 310, 3_444)
MODEL_EXECUTION_DEVICE = "cuda"
MODEL_EXECUTION_DTYPE = "float16"


def test_selected_hub_qwen_spec_is_internally_consistent() -> None:
    """Validate the offline parts of the selected-model contract."""
    assert len(MODEL_REVISION) == 40
    assert all(character in "0123456789abcdef" for character in MODEL_REVISION)
    assert MODEL_CHECKPOINT_DTYPE == "bfloat16"
    assert MODEL_EXECUTION_DEVICE == "cuda"
    assert MODEL_EXECUTION_DTYPE == "float16"
    assert MODEL_DOWNLOAD_SIZE_BYTES > MODEL_CHECKPOINT_ELEMENT_COUNT * 2
    assert all(
        type(token_id) is int and 0 <= token_id < MODEL_VOCAB_SIZE
        for token_id in MODEL_PROMPT_TOKEN_IDS
    )


@pytest.mark.model
def test_selected_hub_qwen_is_public_and_supported() -> None:
    """Verify external metadata without downloading checkpoint weights."""
    model_info = HfApi().model_info(
        MODEL_REPO_ID,
        revision=MODEL_REVISION,
        files_metadata=True,
    )

    assert model_info.sha == MODEL_REVISION
    assert not model_info.private
    assert not model_info.gated

    safetensors_files = {
        sibling.rfilename: sibling.size
        for sibling in model_info.siblings
        if sibling.rfilename.endswith(".safetensors")
    }
    assert safetensors_files == {
        MODEL_CHECKPOINT_FILE: MODEL_DOWNLOAD_SIZE_BYTES,
    }
    assert model_info.safetensors is not None
    assert model_info.safetensors.parameters == {
        "BF16": MODEL_CHECKPOINT_ELEMENT_COUNT,
    }
    assert model_info.safetensors.total == MODEL_CHECKPOINT_ELEMENT_COUNT

    config_path = Path(
        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename="config.json",
            revision=MODEL_REVISION,
        )
    )
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    assert config["architectures"] == ["Qwen3ForCausalLM"]
    assert config["model_type"] == "qwen3"
    assert config["torch_dtype"] == MODEL_CHECKPOINT_DTYPE
    assert config["vocab_size"] == MODEL_VOCAB_SIZE
    assert config["tie_word_embeddings"] is True
    assert config["use_sliding_window"] is False
    assert config["sliding_window"] is None
    assert config["rope_scaling"] is None
    assert config["attention_bias"] is False
    assert config["head_dim"] == 128
    assert "auto_map" not in config
    assert _qwen3_unique_parameter_count(config) == MODEL_UNIQUE_PARAMETER_COUNT
    assert (
        MODEL_UNIQUE_PARAMETER_COUNT + MODEL_VOCAB_SIZE * config["hidden_size"]
        == MODEL_CHECKPOINT_ELEMENT_COUNT
    )
    assert max(MODEL_PROMPT_TOKEN_IDS) < config["vocab_size"]


def _qwen3_unique_parameter_count(config: dict[str, object]) -> int:
    """Calculate unique dense Qwen3 model parameters with tied embeddings."""
    vocab_size = int(config["vocab_size"])
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    num_hidden_layers = int(config["num_hidden_layers"])
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    query_size = num_attention_heads * head_dim
    kv_size = num_key_value_heads * head_dim

    embedding_parameters = vocab_size * hidden_size
    attention_parameters = (
        hidden_size * query_size
        + 2 * hidden_size * kv_size
        + query_size * hidden_size
    )
    query_key_norm_parameters = 2 * head_dim
    mlp_parameters = 3 * hidden_size * intermediate_size
    layer_norm_parameters = 2 * hidden_size
    final_norm_parameters = hidden_size
    output_parameters = (
        0 if config["tie_word_embeddings"] else vocab_size * hidden_size
    )
    return (
        embedding_parameters
        + num_hidden_layers
        * (
            attention_parameters
            + query_key_norm_parameters
            + mlp_parameters
            + layer_norm_parameters
        )
        + final_norm_parameters
        + output_parameters
    )
