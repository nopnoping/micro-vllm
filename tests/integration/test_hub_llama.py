"""Selection contract for the pinned public Llama checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from huggingface_hub import HfApi, hf_hub_download


MODEL_REPO_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_REVISION = "d9128824c0c80111be21424e68086f52413fb413"
MODEL_CHECKPOINT_FILE = "model.safetensors"
MODEL_CHECKPOINT_DTYPE = "bfloat16"
MODEL_PARAMETER_COUNT = 1_100_048_384
MODEL_DOWNLOAD_SIZE_BYTES = 2_200_119_864
MODEL_PROMPT_TOKEN_IDS = (1, 450, 7_483, 310, 3_444)
MODEL_EXECUTION_DEVICE = "cuda"
MODEL_EXECUTION_DTYPE = "float16"


def test_selected_hub_llama_spec_is_internally_consistent() -> None:
    """Validate the offline parts of the selected-model contract."""
    assert len(MODEL_REVISION) == 40
    assert all(character in "0123456789abcdef" for character in MODEL_REVISION)
    assert MODEL_CHECKPOINT_DTYPE == "bfloat16"
    assert MODEL_EXECUTION_DEVICE == "cuda"
    assert MODEL_EXECUTION_DTYPE == "float16"
    assert MODEL_DOWNLOAD_SIZE_BYTES > MODEL_PARAMETER_COUNT * 2
    assert all(
        type(token_id) is int and 0 <= token_id < 32_000
        for token_id in MODEL_PROMPT_TOKEN_IDS
    )


@pytest.mark.model
def test_selected_hub_llama_is_public_and_supported() -> None:
    """Verify external metadata without downloading the checkpoint weights."""
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

    config_path = Path(
        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename="config.json",
            revision=MODEL_REVISION,
        )
    )
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    assert config["architectures"] == ["LlamaForCausalLM"]
    assert config["torch_dtype"] == MODEL_CHECKPOINT_DTYPE
    assert config.get("rope_scaling") is None
    assert "sliding_window" not in config
    assert config.get("attention_bias", False) is False
    assert "auto_map" not in config
    assert _llama_parameter_count(config) == MODEL_PARAMETER_COUNT
    assert max(MODEL_PROMPT_TOKEN_IDS) < config["vocab_size"]


def _llama_parameter_count(config: dict[str, object]) -> int:
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    num_layers = int(config["num_hidden_layers"])
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config["num_key_value_heads"])
    vocab_size = int(config["vocab_size"])
    head_size = hidden_size // num_attention_heads
    key_value_size = num_key_value_heads * head_size

    embedding_parameters = vocab_size * hidden_size
    attention_parameters = (
        hidden_size * hidden_size
        + 2 * hidden_size * key_value_size
        + hidden_size * hidden_size
    )
    mlp_parameters = 3 * hidden_size * intermediate_size
    layer_norm_parameters = 2 * hidden_size
    output_parameters = (
        0 if config["tie_word_embeddings"] else vocab_size * hidden_size
    )
    return (
        embedding_parameters
        + num_layers
        * (attention_parameters + mlp_parameters + layer_norm_parameters)
        + hidden_size
        + output_parameters
    )
