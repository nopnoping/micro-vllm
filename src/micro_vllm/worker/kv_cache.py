"""ModelRunner-owned physical KV-cache tensor storage."""

from typing import cast

import torch

from micro_vllm.config import LLMConfig, ModelConfig


def kv_cache_block_size_bytes(
    llm_config: LLMConfig,
) -> int:
    """Return the physical bytes consumed by one block across all layers."""

    model_config = llm_config.model_config
    num_layers = model_config.require_positive_int("num_hidden_layers")
    head_dim = _model_head_dim(model_config)
    num_attention_heads = model_config.require_positive_int(
        "num_attention_heads"
    )
    num_key_value_heads = model_config.positive_int(
        "num_key_value_heads",
        num_attention_heads,
    )
    element_size = torch.empty((), dtype=model_config.dtype).element_size()
    return (
        num_layers
        * 2
        * llm_config.cache_config.block_size
        * num_key_value_heads
        * head_dim
        * element_size
    )


class PhysicalKVCache:
    """Per-layer physical KV tensors owned by one ModelRunner."""

    __slots__ = ("_layer_caches", "_null_block_id")

    def __init__(
        self,
        llm_config: LLMConfig,
    ) -> None:
        cache_config = llm_config.cache_config
        model_config = llm_config.model_config
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
                cast(int, cache_config.num_blocks),
                cache_config.block_size,
                num_key_value_heads,
                head_dim,
            )
        )
        self._null_block_id = cast(int, cache_config.num_blocks) - 1
        self._layer_caches = tuple(
            torch.empty(
                shape,
                dtype=model_config.dtype,
                device=model_config.device,
            )
            for _ in range(num_layers)
        )

    def __len__(self) -> int:
        return len(self._layer_caches)

    def __getitem__(self, layer_index: int) -> torch.Tensor:
        return self._layer_caches[layer_index]

    @property
    def null_block_id(self) -> int:
        """Physical page reserved for null and rectangular padding entries."""

        return self._null_block_id

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
