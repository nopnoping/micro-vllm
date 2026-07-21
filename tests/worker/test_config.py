"""Tests for explicit model execution device and dtype configuration."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from micro_vllm.worker.config import (
    KVCacheInitializationConfig,
    ModelExecutionConfig,
)


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


def test_kv_cache_initialization_config_stores_profile_inputs() -> None:
    config = KVCacheInitializationConfig(
        block_size=16,
        gpu_memory_utilization=0.8,
        max_num_blocks=128,
        enable_prefix_cache=True,
    )

    assert config.block_size == 16
    assert config.gpu_memory_utilization == 0.8
    assert config.max_num_blocks == 128
    assert config.enable_prefix_cache is True


@pytest.mark.parametrize("block_size", [0, -1])
def test_kv_cache_initialization_rejects_invalid_block_size(
    block_size: int,
) -> None:
    with pytest.raises(ValueError, match="block_size must be greater than zero"):
        KVCacheInitializationConfig(block_size=block_size)


@pytest.mark.parametrize("utilization", [0.0, -0.1, 1.1])
def test_kv_cache_initialization_rejects_invalid_gpu_utilization(
    utilization: float,
) -> None:
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        KVCacheInitializationConfig(
            block_size=16,
            gpu_memory_utilization=utilization,
        )


@pytest.mark.parametrize("max_num_blocks", [0, -1])
def test_kv_cache_initialization_rejects_invalid_block_ceiling(
    max_num_blocks: int,
) -> None:
    with pytest.raises(ValueError, match="max_num_blocks"):
        KVCacheInitializationConfig(
            block_size=16,
            max_num_blocks=max_num_blocks,
        )
