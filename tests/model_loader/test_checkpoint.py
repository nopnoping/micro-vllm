"""Tests for streaming safetensors checkpoints."""

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

import micro_vllm.model_loader.checkpoint as checkpoint_module
from micro_vllm.model_loader.checkpoint import SafetensorsCheckpoint


def write_index(model_path: Path, weight_map: object) -> None:
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )


def test_streams_single_file_in_deterministic_order(tmp_path: Path) -> None:
    expected = {
        "model.z_weight": torch.tensor([[1, 2], [3, 4]]),
        "model.a_weight": torch.tensor([0.25, 0.5]),
    }
    save_file(expected, tmp_path / "model.safetensors")

    actual = list(SafetensorsCheckpoint(tmp_path).iter_tensors())

    assert [name for name, _ in actual] == [
        "model.a_weight",
        "model.z_weight",
    ]
    for name, tensor in actual:
        torch.testing.assert_close(tensor, expected[name])


def test_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no safetensors checkpoint"):
        list(SafetensorsCheckpoint(tmp_path).iter_tensors())


def test_rejects_empty_single_file(tmp_path: Path) -> None:
    save_file({}, tmp_path / "model.safetensors")

    with pytest.raises(ValueError, match="safetensors checkpoint is empty"):
        list(SafetensorsCheckpoint(tmp_path).iter_tensors())


def test_rejects_ambiguous_checkpoint_layout(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").touch()
    write_index(tmp_path, {"weight": "shard.safetensors"})

    with pytest.raises(ValueError, match="ambiguous safetensors checkpoint"):
        list(SafetensorsCheckpoint(tmp_path).iter_tensors())


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("not JSON", "invalid safetensors index JSON"),
        ("[]", "must be a JSON object"),
        ("{}", "weight_map must be a non-empty object"),
        ('{"weight_map": {}}', "weight_map must be a non-empty object"),
        ('{"weight_map": {"weight": 1}}', "invalid shard filename"),
    ],
)
def test_rejects_invalid_index(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        contents,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        list(SafetensorsCheckpoint(tmp_path).iter_tensors())


def test_streams_only_indexed_weights_from_sorted_shards(
    tmp_path: Path,
) -> None:
    first_shard = tmp_path / "model-00001-of-00002.safetensors"
    second_shard = tmp_path / "model-00002-of-00002.safetensors"
    expected = {
        "model.a": torch.tensor([1.0]),
        "model.b": torch.tensor([2.0]),
        "model.c": torch.tensor([3.0]),
    }
    save_file(
        {**expected, "unused": torch.tensor([4.0])},
        first_shard,
    )
    save_file({"model.c": expected["model.c"]}, second_shard)
    write_index(
        tmp_path,
        {
            "model.c": second_shard.name,
            "model.b": first_shard.name,
            "model.a": first_shard.name,
        },
    )

    actual = list(SafetensorsCheckpoint(tmp_path).iter_tensors())

    assert [name for name, _ in actual] == ["model.a", "model.b", "model.c"]
    for name, tensor in actual:
        torch.testing.assert_close(tensor, expected[name])


def test_rejects_missing_shard(tmp_path: Path) -> None:
    write_index(tmp_path, {"model.weight": "missing.safetensors"})

    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        list(SafetensorsCheckpoint(tmp_path).iter_tensors())


@pytest.mark.parametrize("path_kind", ["relative", "absolute"])
def test_rejects_shard_outside_model_directory(
    tmp_path: Path,
    path_kind: str,
) -> None:
    outside_path = tmp_path.parent / "outside.safetensors"
    outside_path.touch()
    shard_filename = (
        "../outside.safetensors"
        if path_kind == "relative"
        else str(outside_path)
    )
    write_index(tmp_path, {"model.weight": shard_filename})

    with pytest.raises(ValueError, match="leaves model directory"):
        list(SafetensorsCheckpoint(tmp_path).iter_tensors())


def test_wraps_corrupt_checkpoint_with_original_cause(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.safetensors"
    checkpoint_path.write_bytes(b"invalid safetensors")

    with pytest.raises(
        ValueError,
        match=r"failed to read safetensors shard.*model\.safetensors",
    ) as error_info:
        list(SafetensorsCheckpoint(tmp_path).iter_tensors())

    assert isinstance(error_info.value.__cause__, SafetensorError)


def test_wraps_missing_indexed_weight(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard.safetensors"
    save_file({"present": torch.tensor([1.0])}, shard_path)
    write_index(tmp_path, {"missing": shard_path.name})

    with pytest.raises(
        ValueError,
        match=r"failed to read safetensors shard.*shard\.safetensors",
    ):
        list(SafetensorsCheckpoint(tmp_path).iter_tensors())


def test_opens_only_one_shard_at_a_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_shard = tmp_path / "first.safetensors"
    second_shard = tmp_path / "second.safetensors"
    save_file({"first": torch.tensor([1.0])}, first_shard)
    save_file({"second": torch.tensor([2.0])}, second_shard)
    write_index(
        tmp_path,
        {"second": second_shard.name, "first": first_shard.name},
    )
    tracker = OpenTracker()
    monkeypatch.setattr(checkpoint_module, "safe_open", tracker.safe_open)

    names = [
        name
        for name, _ in SafetensorsCheckpoint(tmp_path).iter_tensors()
    ]

    assert names == ["first", "second"]
    assert tracker.maximum_open == 1
    assert tracker.currently_open == 0


def test_closing_partial_iterator_closes_current_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_file(
        {"first": torch.tensor([1.0]), "second": torch.tensor([2.0])},
        tmp_path / "model.safetensors",
    )
    tracker = OpenTracker()
    monkeypatch.setattr(checkpoint_module, "safe_open", tracker.safe_open)
    iterator = SafetensorsCheckpoint(tmp_path).iter_tensors()

    assert next(iterator)[0] == "first"
    assert tracker.currently_open == 1
    iterator.close()

    assert tracker.currently_open == 0


class OpenTracker:
    def __init__(self) -> None:
        self.currently_open = 0
        self.maximum_open = 0

    @contextmanager
    def safe_open(self, path: Path, **kwargs: Any) -> Any:
        self.currently_open += 1
        self.maximum_open = max(self.maximum_open, self.currently_open)
        try:
            with safe_open(path, **kwargs) as checkpoint:
                yield checkpoint
        finally:
            self.currently_open -= 1
