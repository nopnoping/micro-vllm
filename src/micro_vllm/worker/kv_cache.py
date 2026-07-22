"""ModelRunner-owned physical KV-cache tensor storage."""

import torch

from micro_vllm.config import CacheConfig, ModelConfig
from micro_vllm.worker.config import ModelExecutionConfig


def kv_cache_block_size_bytes(
    model_config: ModelConfig,
    *,
    block_size: int,
    dtype: torch.dtype,
) -> int:
    """Return the physical bytes consumed by one block across all layers."""

    num_layers = model_config.require_positive_int("num_hidden_layers")
    head_dim = _model_head_dim(model_config)
    num_attention_heads = model_config.require_positive_int(
        "num_attention_heads"
    )
    num_key_value_heads = model_config.positive_int(
        "num_key_value_heads",
        num_attention_heads,
    )
    element_size = torch.empty((), dtype=dtype).element_size()
    return (
        num_layers
        * 2
        * block_size
        * num_key_value_heads
        * head_dim
        * element_size
    )


class PhysicalKVCache:
    """Per-layer physical KV tensors owned by one ModelRunner."""

    __slots__ = ("_layer_caches",)

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        execution_config: ModelExecutionConfig,
    ) -> None:
        num_layers = model_config.require_positive_int("num_hidden_layers")
        head_dim = _model_head_dim(model_config)
        num_attention_heads = model_config.require_positive_int(
            "num_attention_heads"
        )
        num_key_value_heads = model_config.positive_int(
            "num_key_value_heads",
            num_attention_heads,
        )
        shape = torch.Size(
            (
                2,
                cache_config.num_blocks,
                cache_config.block_size,
                num_key_value_heads,
                head_dim,
            )
        )
        self._layer_caches = tuple(
            torch.empty(
                shape,
                dtype=execution_config.dtype,
                device=execution_config.device,
            )
            for _ in range(num_layers)
        )

    def __len__(self) -> int:
        return len(self._layer_caches)

    def __getitem__(self, layer_index: int) -> torch.Tensor:
        return self._layer_caches[layer_index]

    def release(self) -> None:
        """Drop owned tensor references without changing global device state."""

        self._layer_caches = ()


def _model_head_dim(model_config: ModelConfig) -> int:
    hidden_size = model_config.require_positive_int("hidden_size")
    num_attention_heads = model_config.require_positive_int(
        "num_attention_heads"
    )
    if "head_dim" in model_config.values:
        return model_config.require_positive_int("head_dim")
    if hidden_size % num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    return hidden_size // num_attention_heads
