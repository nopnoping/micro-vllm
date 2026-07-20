"""Tests for explicit model execution device and dtype configuration."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from micro_vllm.worker.config import ModelExecutionConfig


def test_cpu_execution_defaults_to_float32() -> None:
    config = ModelExecutionConfig()

    assert config.device == torch.device("cpu")
    assert config.dtype is torch.float32


def test_cuda_execution_accepts_explicit_device_and_dtype() -> None:
    config = ModelExecutionConfig(torch.device("cuda:0"), torch.float16)

    assert config.device == torch.device("cuda:0")
    assert config.dtype is torch.float16


def test_execution_config_is_frozen() -> None:
    config = ModelExecutionConfig()

    with pytest.raises(FrozenInstanceError):
        config.dtype = torch.float16


def test_execution_config_rejects_unsupported_device_type() -> None:
    with pytest.raises(
        ValueError,
        match="execution device must be 'cpu' or 'cuda'",
    ):
        ModelExecutionConfig("mps")


def test_execution_config_rejects_non_floating_dtype() -> None:
    with pytest.raises(
        ValueError,
        match="execution dtype must be float32, float16, or bfloat16",
    ):
        ModelExecutionConfig(dtype=torch.int64)
