import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from micro_vllm.weights import SafetensorsSource


def write_index(model_path: Path, weight_map: object) -> Path:
    index_path = model_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    return index_path


def test_tiny_safetensors_round_trip(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.safetensors"
    expected = {
        "model.embed_tokens.weight": torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
        ),
        "model.norm.weight": torch.tensor([0.5, 1.5], dtype=torch.float16),
    }
    save_file(expected, checkpoint_path)

    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        actual = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}

    assert actual.keys() == expected.keys()
    for name, expected_tensor in expected.items():
        actual_tensor = actual[name]
        assert actual_tensor.device.type == "cpu"
        assert actual_tensor.dtype == expected_tensor.dtype
        assert actual_tensor.shape == expected_tensor.shape
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_source_rejects_missing_model_directory(tmp_path: Path) -> None:
    model_path = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match=str(model_path)):
        SafetensorsSource(model_path)


def test_source_rejects_model_path_that_is_not_a_directory(tmp_path: Path) -> None:
    model_path = tmp_path / "model-file"
    model_path.touch()

    with pytest.raises(ValueError, match="model path must be a directory"):
        SafetensorsSource(model_path)


def test_source_rejects_model_directory_without_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        SafetensorsSource(tmp_path)


def test_source_discovers_single_file_without_opening_it(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.safetensors"
    checkpoint_path.write_bytes(b"not opened during discovery")

    source = SafetensorsSource(tmp_path / ".")

    assert source.model_path == tmp_path.resolve()
    assert len(source._shards) == 1
    assert source._shards[0].path == checkpoint_path.resolve()
    assert source._shards[0].weight_names is None


def test_source_discovers_sharded_index_in_deterministic_order(
    tmp_path: Path,
) -> None:
    first_shard = tmp_path / "model-00001-of-00002.safetensors"
    second_shard = tmp_path / "model-00002-of-00002.safetensors"
    unreferenced_shard = tmp_path / "unused.safetensors"
    for path in (second_shard, unreferenced_shard, first_shard):
        path.touch()
    write_index(
        tmp_path,
        {
            "model.layers.1.weight": second_shard.name,
            "model.layers.0.bias": first_shard.name,
            "model.layers.0.weight": first_shard.name,
        },
    )

    source = SafetensorsSource(tmp_path)

    assert tuple(shard.path for shard in source._shards) == (
        first_shard.resolve(),
        second_shard.resolve(),
    )
    assert tuple(shard.weight_names for shard in source._shards) == (
        ("model.layers.0.bias", "model.layers.0.weight"),
        ("model.layers.1.weight",),
    )
    with pytest.raises(FrozenInstanceError):
        source._shards[0].path = unreferenced_shard


def test_source_rejects_ambiguous_checkpoint_layout(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").touch()
    write_index(tmp_path, {"weight": "shard.safetensors"})

    with pytest.raises(ValueError, match="ambiguous safetensors checkpoint layout"):
        SafetensorsSource(tmp_path)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("not JSON", "invalid safetensors index JSON"),
        ("[]", "must be a JSON object"),
        ("{}", "weight_map must be a non-empty object"),
        ('{"weight_map": {}}', "weight_map must be a non-empty object"),
        ('{"weight_map": {"weight": 1}}', "invalid shard filename"),
        ('{"weight_map": {"": "shard.safetensors"}}', "empty weight name"),
        ('{"weight_map": {"weight": ""}}', "invalid shard filename"),
    ],
)
def test_source_rejects_invalid_index_schema(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        contents,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        SafetensorsSource(tmp_path)


def test_source_rejects_duplicate_weight_definition(tmp_path: Path) -> None:
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        '{"weight_map":{"weight":"first.safetensors",'
        '"weight":"second.safetensors"}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key 'weight'"):
        SafetensorsSource(tmp_path)


def test_source_rejects_missing_shard(tmp_path: Path) -> None:
    write_index(tmp_path, {"model.weight": "missing.safetensors"})

    with pytest.raises(
        FileNotFoundError,
        match=r"model\.weight.*missing\.safetensors",
    ):
        SafetensorsSource(tmp_path)


@pytest.mark.parametrize(
    "shard_filename",
    ["../outside.safetensors", "/tmp/a.safetensors"],
)
def test_source_rejects_shard_outside_model_directory(
    tmp_path: Path,
    shard_filename: str,
) -> None:
    (tmp_path.parent / "outside.safetensors").touch()
    write_index(tmp_path, {"model.weight": shard_filename})

    with pytest.raises(ValueError, match="model directory"):
        SafetensorsSource(tmp_path)


def test_source_rejects_shard_that_is_not_a_file(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard.safetensors"
    shard_path.mkdir()
    write_index(tmp_path, {"model.weight": shard_path.name})

    with pytest.raises(ValueError, match="is not a file"):
        SafetensorsSource(tmp_path)
