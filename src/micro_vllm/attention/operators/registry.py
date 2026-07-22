"""Attention operator registry."""

from collections.abc import Mapping
from types import MappingProxyType

from micro_vllm.attention.operators.base import AttentionOperator
from micro_vllm.attention.operators.reference import (
    ReferenceAttentionOperator,
)


_OPERATOR_CLASSES: Mapping[
    str,
    type[AttentionOperator],
] = MappingProxyType(
    {
        "reference": ReferenceAttentionOperator,
    }
)


def get_attention_operator_class(name: str) -> type[AttentionOperator]:
    """Return the registered attention operator class with the given name."""

    try:
        return _OPERATOR_CLASSES[name]
    except KeyError as error:
        raise ValueError(
            f"unsupported attention operator {name!r}; "
            f"supported operators: {list(_OPERATOR_CLASSES)!r}"
        ) from error
