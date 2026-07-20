import pytest
import torch
import transformers


def test_model_execution_dependencies_are_importable() -> None:
    assert torch.tensor([1.0]).add(1.0).item() == 2.0
    assert transformers.__version__


@pytest.mark.gpu
def test_cuda_executes_tensor_operations() -> None:
    assert torch.cuda.is_available(), "gpu tests require a visible CUDA device"

    left = torch.tensor([[1.0, 2.0]], device="cuda")
    right = torch.tensor([[3.0], [4.0]], device="cuda")

    assert (left @ right).cpu().item() == 11.0
