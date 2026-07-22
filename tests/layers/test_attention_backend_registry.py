"""Tests for constructing attention backends by registered name."""

import pytest

from micro_vllm.layers.attention_backend import ReferenceAttentionBackend
from micro_vllm.layers.attention_backend_registry import (
    create_attention_backend,
)


def test_creates_registered_reference_backend() -> None:
    backend = create_attention_backend("reference")

    assert isinstance(backend, ReferenceAttentionBackend)


def test_rejects_unknown_backend_name() -> None:
    with pytest.raises(
        ValueError,
        match=r"unsupported attention backend 'unknown'.*reference",
    ):
        create_attention_backend("unknown")
