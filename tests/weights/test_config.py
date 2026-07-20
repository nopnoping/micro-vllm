"""Tests for local Hugging Face model metadata discovery."""

import json
from pathlib import Path

import pytest

from micro_vllm.weights.config import _read_hf_config


def write_config(model_path: Path, config: object) -> Path:
    config_path = model_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_read_hf_config_accepts_local_metadata_without_checkpoint(
    tmp_path: Path,
) -> None:
    config_path = write_config(
        tmp_path,
        {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "transformers_version": "4.45.0",
            "auto_map": {"AutoModel": "remote.Model"},
        },
    )

    config = _read_hf_config(tmp_path / ".")

    assert config.path == config_path.resolve()
    assert config.architectures == ("LlamaForCausalLM",)
    assert config.raw["model_type"] == "llama"
    assert config.raw["transformers_version"] == "4.45.0"
    assert config.raw["auto_map"] == {"AutoModel": "remote.Model"}


def test_read_hf_config_does_not_modify_source_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    contents = '{\n  "architectures": ["LlamaForCausalLM"]\n}\n'
    config_path.write_text(contents, encoding="utf-8")

    _read_hf_config(tmp_path)

    assert config_path.read_text(encoding="utf-8") == contents


def test_read_hf_config_rejects_missing_model_directory(tmp_path: Path) -> None:
    model_path = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match=str(model_path)):
        _read_hf_config(model_path)


def test_read_hf_config_rejects_model_path_that_is_not_directory(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model-file"
    model_path.touch()

    with pytest.raises(ValueError, match="model path must be a directory"):
        _read_hf_config(model_path)


def test_read_hf_config_rejects_missing_config_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"

    with pytest.raises(FileNotFoundError, match=str(config_path)):
        _read_hf_config(tmp_path)


def test_read_hf_config_rejects_config_path_that_is_not_file(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.mkdir()

    with pytest.raises(ValueError, match=rf"not a file: {config_path}"):
        _read_hf_config(tmp_path)


@pytest.mark.parametrize(
    "contents",
    [
        b"not JSON",
        b'{"architectures": ["LlamaForCausalLM"]}\xff',
    ],
    ids=["invalid-json", "invalid-utf8"],
)
def test_read_hf_config_wraps_text_decoding_errors_with_config_path(
    tmp_path: Path,
    contents: bytes,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_bytes(contents)

    with pytest.raises(ValueError, match=rf"invalid model config JSON: {config_path}"):
        _read_hf_config(tmp_path)


@pytest.mark.parametrize("value", [None, [], "LlamaForCausalLM", 1, True])
def test_read_hf_config_requires_json_object(
    tmp_path: Path,
    value: object,
) -> None:
    config_path = write_config(tmp_path, value)

    with pytest.raises(
        ValueError,
        match=rf"model config must be a JSON object: {config_path}",
    ):
        _read_hf_config(tmp_path)


def test_read_hf_config_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"architectures":["First"],"architectures":["Second"]}',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=rf"invalid model config {config_path}: duplicate JSON key",
    ):
        _read_hf_config(tmp_path)


@pytest.mark.parametrize(
    "architectures",
    [
        None,
        "LlamaForCausalLM",
        {},
        [],
        ["LlamaForCausalLM", ""],
        ["   "],
    ],
    ids=["missing", "string", "object", "empty", "empty-entry", "whitespace"],
)
def test_read_hf_config_rejects_invalid_architectures(
    tmp_path: Path,
    architectures: object,
) -> None:
    raw_config = {}
    if architectures is not None:
        raw_config["architectures"] = architectures
    write_config(tmp_path, raw_config)

    with pytest.raises(ValueError, match="architectures?.*non-empty"):
        _read_hf_config(tmp_path)


@pytest.mark.parametrize("architecture", [1, True, None, [], {}])
def test_read_hf_config_requires_string_architecture_entries(
    tmp_path: Path,
    architecture: object,
) -> None:
    write_config(tmp_path, {"architectures": [architecture]})

    with pytest.raises(ValueError, match="architecture at index 0.*non-empty string"):
        _read_hf_config(tmp_path)


def test_model_type_does_not_replace_architectures(tmp_path: Path) -> None:
    write_config(tmp_path, {"model_type": "llama"})

    with pytest.raises(ValueError, match="architectures"):
        _read_hf_config(tmp_path)
