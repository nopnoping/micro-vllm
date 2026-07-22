"""ModelRunner-owned physical KV-cache tensor storage."""

import torch

from micro_vllm.config import LLMConfig, ModelConfig
from micro_vllm.core.kv_cache_spec import KVCacheLayout, KVCacheTopology


def kv_cache_block_size_bytes(
    llm_config: LLMConfig,
    topology: KVCacheTopology,
) -> int:
    """Return the bytes consumed by one shared physical cache page."""

    model_config = llm_config.model_config
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
        topology.page_layer_count
        * 2
        * llm_config.cache_config.block_size
        * num_key_value_heads
        * head_dim
        * element_size
    )


class PhysicalKVCache:
    """Shared physical KV page buffers owned by one ModelRunner."""

    __slots__ = ("_buffers", "_null_block_id")

    def __init__(
        self,
        llm_config: LLMConfig,
        layout: KVCacheLayout,
    ) -> None:
        model_config = llm_config.model_config
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
                layout.num_blocks,
                layout.block_size,
                num_key_value_heads,
                head_dim,
            )
        )
        self._null_block_id = layout.num_blocks - 1
        self._buffers = tuple(
            torch.empty(
                shape,
                dtype=model_config.dtype,
                device=model_config.device,
            )
            for _ in range(layout.page_layer_count)
        )

    def __len__(self) -> int:
        return len(self._buffers)

    def __getitem__(self, buffer_index: int) -> torch.Tensor:
        return self._buffers[buffer_index]

    @property
    def null_block_id(self) -> int:
        """Physical page reserved for null and rectangular padding entries."""

        return self._null_block_id

    def release(self) -> None:
        """Drop owned tensor references without changing global device state."""

        self._buffers = ()


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
