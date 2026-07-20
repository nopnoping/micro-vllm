from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


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
