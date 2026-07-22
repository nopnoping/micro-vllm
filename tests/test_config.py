"""Tests for top-level configuration objects."""

import pytest
import torch

from micro_vllm.config import (
    CacheConfig,
    LLMConfig,
    SchedulerConfig,
)


def test_cache_config_accepts_valid_configuration() -> None:
    config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.8,
        num_blocks=128,
        enable_prefix_cache=True,
    )

    assert config.block_size == 16
    assert config.gpu_memory_utilization == 0.8
    assert config.num_blocks == 128
    assert config.enable_prefix_cache is True


@pytest.mark.parametrize("utilization", [0.0, -0.1, 1.1])
def test_cache_config_rejects_invalid_gpu_utilization(
    utilization: float,
) -> None:
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        CacheConfig(gpu_memory_utilization=utilization)


def test_cache_config_disables_unbounded_prefix_cache_by_default() -> None:
    config = CacheConfig()

    assert config.enable_prefix_cache is False


def test_cache_config_can_be_updated_with_runtime_capacity() -> None:
    config = CacheConfig(num_blocks=128)

    config.num_blocks = 64

    assert config.num_blocks == 64


@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.parametrize("field_name", ["block_size", "num_blocks"])
def test_cache_config_rejects_non_positive_sizes(
    field_name: str,
    value: int,
) -> None:
    values = {"block_size": 16, "num_blocks": 128}
    values[field_name] = value

    with pytest.raises(
        ValueError,
        match=rf"^{field_name} must be greater than zero$",
    ):
        CacheConfig(**values)


def test_scheduler_config_stores_limits() -> None:
    config = SchedulerConfig(
        max_num_batched_tokens=16,
        max_num_seqs=4,
        max_model_len=128,
    )

    assert config.max_num_batched_tokens == 16
    assert config.max_num_seqs == 4
    assert config.max_model_len == 128


@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.parametrize(
    "field_name",
    ["max_num_batched_tokens", "max_num_seqs", "max_model_len"],
)
def test_scheduler_config_rejects_non_positive_limits(
    field_name: str, value: int
) -> None:
    values = {
        "max_num_batched_tokens": 16,
        "max_num_seqs": 4,
        "max_model_len": 128,
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=rf"^{field_name} must be greater than zero$"):
        SchedulerConfig(**values)


def test_llm_config_groups_user_configuration() -> None:
    config = LLMConfig(model="org/model", device="cpu")

    assert config.model == "org/model"
    assert config.device == "cpu"
    assert config.attention_backend == "reference"
    assert config.scheduler_config.max_model_len == 2048
    assert config.scheduler_config.max_num_seqs == 8
    assert config.cache_config.block_size == 16


def test_llm_config_can_be_updated_during_runtime() -> None:
    config = LLMConfig(model="org/model")

    config.attention_backend = "custom"
    config.scheduler_config.max_num_seqs = 4

    assert config.attention_backend == "custom"
    assert config.scheduler_config.max_num_seqs == 4


def test_llm_config_owns_scheduler_and_cache_configs() -> None:
    scheduler_config = SchedulerConfig(16, 4, 128)
    cache_config = CacheConfig(block_size=8, num_blocks=64)

    config = LLMConfig(
        model="org/model",
        scheduler_config=scheduler_config,
        cache_config=cache_config,
    )

    assert config.scheduler_config is scheduler_config
    assert config.cache_config is cache_config


def test_llm_config_rejects_unsupported_device_type() -> None:
    with pytest.raises(ValueError, match="device must be 'cpu' or 'cuda'"):
        LLMConfig(model="unused", device="mps")


def test_llm_config_rejects_non_floating_dtype() -> None:
    with pytest.raises(ValueError, match="dtype must be"):
        LLMConfig(model="unused", dtype=torch.int64)
