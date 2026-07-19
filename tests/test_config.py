"""Tests for top-level configuration objects."""

from dataclasses import FrozenInstanceError

import pytest

from micro_vllm.config import CacheConfig, SchedulerConfig


def test_cache_config_accepts_valid_configuration() -> None:
    config = CacheConfig(
        block_size=16,
        num_blocks=128,
        enable_prefix_cache=True,
    )

    assert config.block_size == 16
    assert config.num_blocks == 128
    assert config.enable_prefix_cache is True


@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.parametrize("field_name", ["block_size", "num_blocks"])
def test_cache_config_rejects_non_positive_sizes(
    field_name: str, value: int
) -> None:
    values = {"block_size": 16, "num_blocks": 128}
    values[field_name] = value

    with pytest.raises(ValueError, match=rf"^{field_name} must be greater than zero$"):
        CacheConfig(**values)


def test_cache_config_disables_unbounded_prefix_cache_by_default() -> None:
    config = CacheConfig(block_size=16, num_blocks=128)

    assert config.enable_prefix_cache is False


def test_cache_config_is_frozen() -> None:
    config = CacheConfig(block_size=16, num_blocks=128)

    with pytest.raises(FrozenInstanceError):
        config.num_blocks = 64


def test_scheduler_config_stores_limits() -> None:
    config = SchedulerConfig(max_num_batched_tokens=16, max_num_seqs=4)

    assert config.max_num_batched_tokens == 16
    assert config.max_num_seqs == 4


@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.parametrize("field_name", ["max_num_batched_tokens", "max_num_seqs"])
def test_scheduler_config_rejects_non_positive_limits(
    field_name: str, value: int
) -> None:
    values = {"max_num_batched_tokens": 16, "max_num_seqs": 4}
    values[field_name] = value

    with pytest.raises(ValueError, match=rf"^{field_name} must be greater than zero$"):
        SchedulerConfig(**values)
