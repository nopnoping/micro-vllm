"""Worker-owned physical KV-cache tensor storage."""

import torch

from micro_vllm.config import CacheConfig
from micro_vllm.config import ModelConfig
from micro_vllm.worker.config import ModelExecutionConfig


class PhysicalKVCache:
    """Per-layer physical KV tensors owned by one Worker."""

    __slots__ = (
        "_device",
        "_dtype",
        "_layer_caches",
        "_num_layers",
        "_shape",
    )

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        execution_config: ModelExecutionConfig,
    ) -> None:
        self._num_layers = model_config.require_positive_int("num_hidden_layers")
        hidden_size = model_config.require_positive_int("hidden_size")
        num_attention_heads = model_config.require_positive_int(
            "num_attention_heads"
        )
        num_key_value_heads = model_config.positive_int(
            "num_key_value_heads",
            num_attention_heads,
        )
        self._shape = torch.Size(
            (
                2,
                cache_config.num_blocks,
                cache_config.block_size,
                num_key_value_heads,
                hidden_size // num_attention_heads,
            )
        )
        self._dtype = execution_config.dtype
        self._device = execution_config.device
        self._layer_caches = tuple(
            torch.empty(
                self._shape,
                dtype=self._dtype,
                device=self._device,
            )
            for _ in range(self._num_layers)
        )

    def __len__(self) -> int:
        return len(self._layer_caches)

    def __getitem__(self, layer_index: int) -> torch.Tensor:
        return self._layer_caches[layer_index]

    @property
    def shape(self) -> torch.Size:
        """Expected shape of every layer cache tensor."""

        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        """Storage dtype shared by every layer cache tensor."""

        return self._dtype

    @property
    def device(self) -> torch.device:
        """Storage device shared by every layer cache tensor."""

        return self._device

    def validate(self) -> None:
        """Validate the physical tensor contract before model execution."""

        if len(self._layer_caches) != self._num_layers:
            raise ValueError(
                f"physical KV cache must contain {self._num_layers} layers"
            )
        for layer_index, layer_cache in enumerate(self._layer_caches):
            if layer_cache.shape != self._shape:
                raise ValueError(
                    f"layer {layer_index} KV cache has invalid shape"
                )
            if layer_cache.dtype is not self._dtype:
                raise ValueError(
                    f"layer {layer_index} KV cache has invalid dtype"
                )
            if layer_cache.device != self._device:
                raise ValueError(
                    f"layer {layer_index} KV cache has invalid device"
                )

    def release(self) -> None:
        """Drop owned tensor references without changing global device state."""

        self._layer_caches = ()
