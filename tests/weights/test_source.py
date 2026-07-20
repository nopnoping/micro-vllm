import json
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

import micro_vllm.weights.source as source_module
from micro_vllm.weights import SafetensorsSource


def write_index(model_path: Path, weight_map: object) -> Path:
    index_path = model_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    return index_path


def save_tensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    save_file(tensors, path)


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


@pytest.mark.parametrize("path_kind", ["relative", "absolute"])
def test_source_rejects_shard_outside_model_directory(
    tmp_path: Path,
    path_kind: str,
) -> None:
    outside_path = tmp_path.parent / "outside.safetensors"
    outside_path.touch()
    shard_filename = (
        "../outside.safetensors"
        if path_kind == "relative"
        else str(outside_path.resolve())
    )
    write_index(tmp_path, {"model.weight": shard_filename})

    with pytest.raises(ValueError, match="model directory"):
        SafetensorsSource(tmp_path)


def test_source_rejects_shard_that_is_not_a_file(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard.safetensors"
    shard_path.mkdir()
    write_index(tmp_path, {"model.weight": shard_path.name})

    with pytest.raises(ValueError, match="is not a file"):
        SafetensorsSource(tmp_path)


def test_single_file_iteration_is_deterministic_and_preserves_tensors(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.safetensors"
    expected = {
        "model.z_weight": torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
        "model.a_weight": torch.tensor([0.25, 0.5], dtype=torch.float16),
    }
    save_tensors(checkpoint_path, expected)

    actual = list(SafetensorsSource(tmp_path).iter_tensors())

    assert [name for name, _ in actual] == ["model.a_weight", "model.z_weight"]
    for name, tensor in actual:
        assert tensor.device.type == "cpu"
        assert tensor.dtype == expected[name].dtype
        assert tensor.shape == expected[name].shape
        torch.testing.assert_close(tensor, expected[name])


def test_single_file_iteration_restarts_for_each_iterator(tmp_path: Path) -> None:
    save_tensors(
        tmp_path / "model.safetensors",
        {"weight": torch.tensor([1.0])},
    )
    source = SafetensorsSource(tmp_path)

    first = list(source.iter_tensors())
    second = list(source.iter_tensors())

    assert [name for name, _ in first] == ["weight"]
    assert [name for name, _ in second] == ["weight"]
    torch.testing.assert_close(first[0][1], second[0][1])


def test_single_file_iteration_rejects_empty_checkpoint(tmp_path: Path) -> None:
    save_tensors(tmp_path / "model.safetensors", {})

    with pytest.raises(ValueError, match=r"checkpoint is empty.*model\.safetensors"):
        list(SafetensorsSource(tmp_path).iter_tensors())


def test_sharded_iteration_uses_sorted_shards_and_weight_names(
    tmp_path: Path,
) -> None:
    first_shard = tmp_path / "model-00001-of-00002.safetensors"
    second_shard = tmp_path / "model-00002-of-00002.safetensors"
    expected = {
        "model.a": torch.tensor([1.0]),
        "model.b": torch.tensor([2.0]),
        "model.c": torch.tensor([3.0]),
    }
    save_tensors(
        first_shard,
        {"model.b": expected["model.b"], "model.a": expected["model.a"]},
    )
    save_tensors(second_shard, {"model.c": expected["model.c"]})
    write_index(
        tmp_path,
        {
            "model.c": second_shard.name,
            "model.b": first_shard.name,
            "model.a": first_shard.name,
        },
    )

    actual = list(SafetensorsSource(tmp_path).iter_tensors())

    assert [name for name, _ in actual] == ["model.a", "model.b", "model.c"]
    assert len({name for name, _ in actual}) == len(actual)
    for name, tensor in actual:
        torch.testing.assert_close(tensor, expected[name])


def test_sharded_iteration_rejects_missing_declared_weight(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard.safetensors"
    save_tensors(shard_path, {"present": torch.tensor([1.0])})
    write_index(
        tmp_path,
        {"present": shard_path.name, "missing": shard_path.name},
    )

    with pytest.raises(
        ValueError,
        match=r"shard\.safetensors.*missing weights \['missing'\]",
    ):
        list(SafetensorsSource(tmp_path).iter_tensors())


def test_sharded_iteration_rejects_unexpected_weight(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard.safetensors"
    save_tensors(
        shard_path,
        {"declared": torch.tensor([1.0]), "extra": torch.tensor([2.0])},
    )
    write_index(tmp_path, {"declared": shard_path.name})

    with pytest.raises(
        ValueError,
        match=r"shard\.safetensors.*unexpected weights \['extra'\]",
    ):
        list(SafetensorsSource(tmp_path).iter_tensors())


def test_sharded_iteration_opens_only_one_shard_at_a_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_shard = tmp_path / "first.safetensors"
    second_shard = tmp_path / "second.safetensors"
    save_tensors(first_shard, {"first": torch.tensor([1.0])})
    save_tensors(second_shard, {"second": torch.tensor([2.0])})
    write_index(
        tmp_path,
        {"second": second_shard.name, "first": first_shard.name},
    )
    tracker = OpenTracker()
    monkeypatch.setattr(source_module, "safe_open", tracker.safe_open)

    names = [name for name, _ in SafetensorsSource(tmp_path).iter_tensors()]

    assert names == ["first", "second"]
    assert tracker.maximum_open == 1
    assert tracker.events == [
        ("open", "first.safetensors"),
        ("close", "first.safetensors"),
        ("open", "second.safetensors"),
        ("close", "second.safetensors"),
    ]


def test_closing_partial_iterator_closes_current_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_path = tmp_path / "model.safetensors"
    save_tensors(
        shard_path,
        {"first": torch.tensor([1.0]), "second": torch.tensor([2.0])},
    )
    tracker = OpenTracker()
    monkeypatch.setattr(source_module, "safe_open", tracker.safe_open)
    iterator = SafetensorsSource(tmp_path).iter_tensors()

    assert next(iterator)[0] == "first"
    assert tracker.currently_open == 1
    iterator.close()

    assert tracker.currently_open == 0
    assert tracker.events == [
        ("open", "model.safetensors"),
        ("close", "model.safetensors"),
    ]


def test_later_shard_read_failure_closes_each_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_shard = tmp_path / "first.safetensors"
    second_shard = tmp_path / "second.safetensors"
    save_tensors(first_shard, {"first": torch.tensor([1.0])})
    save_tensors(second_shard, {"second": torch.tensor([2.0])})
    write_index(
        tmp_path,
        {"second": second_shard.name, "first": first_shard.name},
    )
    tracker = OpenTracker(fail_on_filename=second_shard.name)
    monkeypatch.setattr(source_module, "safe_open", tracker.safe_open)

    with pytest.raises(RuntimeError, match="simulated tensor read failure"):
        list(SafetensorsSource(tmp_path).iter_tensors())

    assert tracker.currently_open == 0
    assert tracker.maximum_open == 1
    assert tracker.events == [
        ("open", "first.safetensors"),
        ("close", "first.safetensors"),
        ("open", "second.safetensors"),
        ("close", "second.safetensors"),
    ]


def test_iteration_wraps_corrupt_checkpoint_with_original_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "model.safetensors"
    checkpoint_path.write_bytes(b"invalid safetensors")
    tracker = OpenTracker()
    monkeypatch.setattr(source_module, "safe_open", tracker.safe_open)

    with pytest.raises(
        ValueError,
        match=r"failed to read safetensors shard.*model\.safetensors",
    ) as exc_info:
        list(SafetensorsSource(tmp_path).iter_tensors())

    assert isinstance(exc_info.value.__cause__, SafetensorError)
    assert tracker.currently_open == 0
    assert tracker.events == [
        ("open", "model.safetensors"),
        ("close", "model.safetensors"),
    ]


def test_new_iterator_can_retry_after_failure(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.safetensors"
    checkpoint_path.write_bytes(b"invalid safetensors")
    source = SafetensorsSource(tmp_path)

    with pytest.raises(ValueError, match="failed to read safetensors shard"):
        list(source.iter_tensors())

    save_tensors(checkpoint_path, {"weight": torch.tensor([1.0])})
    actual = list(source.iter_tensors())

    assert [name for name, _ in actual] == ["weight"]
    torch.testing.assert_close(actual[0][1], torch.tensor([1.0]))


def test_iteration_reports_shard_removed_after_discovery(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.safetensors"
    save_tensors(checkpoint_path, {"weight": torch.tensor([1.0])})
    source = SafetensorsSource(tmp_path)
    checkpoint_path.unlink()

    with pytest.raises(
        FileNotFoundError,
        match=r"checkpoint shard does not exist.*model\.safetensors",
    ):
        list(source.iter_tensors())


def test_index_mismatch_closes_current_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_path = tmp_path / "shard.safetensors"
    save_tensors(shard_path, {"actual": torch.tensor([1.0])})
    write_index(tmp_path, {"declared": shard_path.name})
    tracker = OpenTracker()
    monkeypatch.setattr(source_module, "safe_open", tracker.safe_open)

    with pytest.raises(
        ValueError,
        match=r"shard\.safetensors.*missing weights.*unexpected weights",
    ):
        list(SafetensorsSource(tmp_path).iter_tensors())

    assert tracker.currently_open == 0
    assert tracker.events == [
        ("open", "shard.safetensors"),
        ("close", "shard.safetensors"),
    ]


class OpenTracker:
    def __init__(self, fail_on_filename: str | None = None) -> None:
        self.currently_open = 0
        self.maximum_open = 0
        self.events: list[tuple[str, str]] = []
        self.fail_on_filename = fail_on_filename

    @contextmanager
    def safe_open(self, path: Path, **kwargs: Any) -> Any:
        filename = path.name
        self.currently_open += 1
        self.maximum_open = max(self.maximum_open, self.currently_open)
        self.events.append(("open", filename))
        try:
            with safe_open(path, **kwargs) as checkpoint:
                if filename == self.fail_on_filename:
                    yield FailingCheckpoint(checkpoint)
                else:
                    yield checkpoint
        finally:
            self.currently_open -= 1
            self.events.append(("close", filename))


class FailingCheckpoint:
    def __init__(self, checkpoint: Any) -> None:
        self.checkpoint = checkpoint

    def keys(self) -> list[str]:
        return self.checkpoint.keys()

    def get_tensor(self, weight_name: str) -> torch.Tensor:
        raise RuntimeError(f"simulated tensor read failure: {weight_name}")
