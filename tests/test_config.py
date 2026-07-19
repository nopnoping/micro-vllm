"""Tests for top-level configuration objects."""

import pytest

from micro_vllm.config import SchedulerConfig


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


@pytest.mark.parametrize("value", [True, 1.5, "4"])
@pytest.mark.parametrize("field_name", ["max_num_batched_tokens", "max_num_seqs"])
def test_scheduler_config_rejects_non_integer_limits(
    field_name: str, value: object
) -> None:
    values = {"max_num_batched_tokens": 16, "max_num_seqs": 4}
    values[field_name] = value

    with pytest.raises(TypeError, match=rf"^{field_name} must be an integer$"):
        SchedulerConfig(**values)
