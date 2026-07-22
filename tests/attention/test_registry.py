"""Tests for constructing attention by registered operator name."""

import pytest

from micro_vllm.attention.full.attention import FullAttention
from micro_vllm.attention.full.reference import (
    ReferenceFullAttentionOperator,
)
from micro_vllm.attention.registry import create_attention


def test_creates_full_attention_with_registered_reference_operator() -> None:
    attention = create_attention("reference")

    assert isinstance(attention, FullAttention)
    assert isinstance(attention.operator, ReferenceFullAttentionOperator)


def test_rejects_unknown_operator_name() -> None:
    with pytest.raises(
        ValueError,
        match=r"unsupported attention operator 'unknown'.*reference",
    ):
        create_attention("unknown")
