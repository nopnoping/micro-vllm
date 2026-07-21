"""Shared test configuration for micro-vLLM."""

from __future__ import annotations

import os

import pytest


_MODEL_TEST_ENV = "MICRO_VLLM_TEST_MODEL"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Keep external model access opt-in for every pytest invocation."""
    if os.environ.get(_MODEL_TEST_ENV) == "1":
        return

    skip_model = pytest.mark.skip(
        reason=f"set {_MODEL_TEST_ENV}=1 to run external model tests",
    )
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip_model)
