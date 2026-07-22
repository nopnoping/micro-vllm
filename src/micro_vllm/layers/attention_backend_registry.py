"""Static registry for constructing attention backends by name."""

from collections.abc import Callable, Mapping
from types import MappingProxyType

from micro_vllm.layers.attention_backend import (
    AttentionBackend,
    ReferenceAttentionBackend,
)


_BACKEND_FACTORIES: Mapping[str, Callable[[], AttentionBackend]] = (
    MappingProxyType({"reference": ReferenceAttentionBackend})
)


def create_attention_backend(name: str) -> AttentionBackend:
    """Construct the registered attention backend selected by name."""
    try:
        factory = _BACKEND_FACTORIES[name]
    except KeyError as error:
        raise ValueError(
            f"unsupported attention backend {name!r}; "
            f"supported backends: {list(_BACKEND_FACTORIES)!r}"
        ) from error
    return factory()
