"""Tests for generic Hugging Face config file parsing."""

import json
from pathlib import Path

import pytest

from micro_vllm.model_loader.loader import _read_model_config as read_model_config


def write_config(tmp_path: Path, config: object) -> Path:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_read_model_config_accepts_metadata(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "auto_map": {"AutoModel": "remote.Model"},
        },
    )

    config = read_model_config(config_path)

    assert config.architectures == ("LlamaForCausalLM",)
    assert config.values.get("model_type") == "llama"
    assert config.values.get("auto_map") == {"AutoModel": "remote.Model"}


def test_read_model_config_does_not_modify_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    contents = '{\n  "architectures": ["LlamaForCausalLM"]\n}\n'
    config_path.write_text(contents, encoding="utf-8")

    read_model_config(config_path)

    assert config_path.read_text(encoding="utf-8") == contents


@pytest.mark.parametrize(
    "contents",
    [
        b"not JSON",
        b'{"architectures": ["LlamaForCausalLM"]}\xff',
    ],
)
def test_read_model_config_rejects_invalid_json_text(
    tmp_path: Path,
    contents: bytes,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_bytes(contents)

    with pytest.raises(ValueError, match=str(config_path)):
        read_model_config(config_path)


@pytest.mark.parametrize("value", [None, [], "LlamaForCausalLM", 1, True])
def test_read_model_config_requires_object(tmp_path: Path, value: object) -> None:
    config_path = write_config(tmp_path, value)

    with pytest.raises(ValueError, match="must be a JSON object"):
        read_model_config(config_path)


@pytest.mark.parametrize(
    "architectures",
    [None, "LlamaForCausalLM", {}, [], [""], ["   "], [True]],
)
def test_read_model_config_rejects_invalid_architectures(
    tmp_path: Path,
    architectures: object,
) -> None:
    raw_config = {}
    if architectures is not None:
        raw_config["architectures"] = architectures
    config_path = write_config(tmp_path, raw_config)

    with pytest.raises(ValueError, match="architectures?|architecture"):
        read_model_config(config_path)
