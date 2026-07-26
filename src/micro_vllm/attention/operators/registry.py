"""Attention operator registry."""

from collections.abc import Mapping
from types import MappingProxyType

from micro_vllm.attention.operators.base import AttentionOperator
from micro_vllm.attention.operators.reference import (
    ReferenceAttentionOperator,
)
from micro_vllm.attention.operators.sdpa import SDPAAttentionOperator


_OPERATOR_CLASSES: Mapping[
    str,
    type[AttentionOperator],
] = MappingProxyType(
    {
        "reference": ReferenceAttentionOperator,
        "sdpa": SDPAAttentionOperator,
    }
)


def get_attention_operator_class(name: str) -> type[AttentionOperator]:
    """Return the registered attention operator class with the given name."""

    if name == "triton_paged":
        try:
            from micro_vllm.attention.operators.triton_paged import (
                TritonPagedAttentionOperator,
            )
        except ModuleNotFoundError as error:
            if error.name != "triton":
                raise
            raise RuntimeError(
                "attention operator 'triton_paged' requires Triton"
            ) from error
        return TritonPagedAttentionOperator

    try:
        return _OPERATOR_CLASSES[name]
    except KeyError as error:
        supported_operators = [
            *_OPERATOR_CLASSES,
            "triton_paged",
        ]
        raise ValueError(
            f"unsupported attention operator {name!r}; "
            f"supported operators: {supported_operators!r}"
        ) from error
