"""Tests for resolving attention operators by registered name."""

import pytest

from micro_vllm.attention.operators.reference import (
    ReferenceAttentionOperator,
)
from micro_vllm.attention.operators.registry import (
    get_attention_operator_class,
)
from micro_vllm.attention.operators.sdpa import SDPAAttentionOperator


def test_returns_registered_reference_operator_class() -> None:
    operator_class = get_attention_operator_class("reference")

    assert operator_class is ReferenceAttentionOperator


def test_returns_registered_sdpa_operator_class() -> None:
    operator_class = get_attention_operator_class("sdpa")

    assert operator_class is SDPAAttentionOperator


def test_returns_registered_triton_paged_operator_class() -> None:
    pytest.importorskip("triton")
    from micro_vllm.attention.operators.triton_paged import (
        TritonPagedAttentionOperator,
    )

    operator_class = get_attention_operator_class("triton_paged")

    assert operator_class is TritonPagedAttentionOperator


def test_rejects_unknown_operator_name() -> None:
    with pytest.raises(
        ValueError,
        match=r"unsupported attention operator 'unknown'.*reference",
    ):
        get_attention_operator_class("unknown")
