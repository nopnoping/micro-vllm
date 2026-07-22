"""Construction of attention semantics from a configured operator name."""

from collections.abc import Callable, Mapping
from types import MappingProxyType

from micro_vllm.attention.base import Attention
from micro_vllm.attention.full.attention import FullAttention
from micro_vllm.attention.full.operator import FullAttentionOperator
from micro_vllm.attention.full.reference import (
    ReferenceFullAttentionOperator,
)


_FULL_ATTENTION_OPERATOR_FACTORIES: Mapping[
    str,
    Callable[[], FullAttentionOperator],
] = MappingProxyType(
    {
        "reference": ReferenceFullAttentionOperator,
    }
)


def create_attention(operator_name: str) -> Attention:
    """Construct full attention with the selected registered operator."""

    try:
        operator_factory = _FULL_ATTENTION_OPERATOR_FACTORIES[operator_name]
    except KeyError as error:
        raise ValueError(
            f"unsupported attention operator {operator_name!r}; "
            "supported operators: "
            f"{list(_FULL_ATTENTION_OPERATOR_FACTORIES)!r}"
        ) from error
    return FullAttention(operator_factory())
