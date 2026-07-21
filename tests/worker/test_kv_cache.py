"""Tests for ModelRunner-owned physical KV-cache storage."""

import pytest
import torch

from micro_vllm.config import CacheConfig
from micro_vllm.core.kv_cache_manager import KVCacheManager
from micro_vllm.config import ModelConfig
from micro_vllm.worker.config import ModelExecutionConfig
from micro_vllm.worker.kv_cache import PhysicalKVCache


def make_model_config() -> ModelConfig:
    return ModelConfig(
        {
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 256,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
        }
    )


def make_cache(
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PhysicalKVCache:
    return PhysicalKVCache(
        CacheConfig(block_size=4, num_blocks=8),
        make_model_config(),
        ModelExecutionConfig(device=device, dtype=dtype),
    )


def test_cpu_cache_uses_model_cache_and_execution_configuration() -> None:
    cache = make_cache()

    assert len(cache) == 2
    assert cache.shape == torch.Size((2, 8, 4, 2, 16))
    assert cache.dtype is torch.float32
    assert cache.device == torch.device("cpu")
    for layer_cache in (cache[0], cache[1]):
        assert layer_cache.shape == cache.shape
        assert layer_cache.dtype is cache.dtype
        assert layer_cache.device == cache.device


def test_physical_block_id_indexes_the_block_dimension() -> None:
    cache = make_cache()
    layer_cache = cache[0]
    layer_cache.zero_()

    physical_block_id = 3
    layer_cache[:, physical_block_id].fill_(7)

    assert torch.count_nonzero(layer_cache[:, physical_block_id]).item() > 0
    assert torch.count_nonzero(layer_cache[:, :physical_block_id]).item() == 0
    assert torch.count_nonzero(layer_cache[:, physical_block_id + 1 :]).item() == 0


def test_layers_do_not_share_storage() -> None:
    cache = make_cache()

    assert cache[0].data_ptr() != cache[1].data_ptr()


def test_logical_block_release_does_not_clear_physical_storage() -> None:
    cache_config = CacheConfig(block_size=4, num_blocks=8)
    manager = KVCacheManager(cache_config)
    manager.register_request("request-1")
    allocation = manager.allocate_slots("request-1", 0, 1)
    assert allocation is not None
    physical_block_id = allocation.block_table[0]
    assert physical_block_id is not None
    cache = PhysicalKVCache(
        cache_config,
        make_model_config(),
        ModelExecutionConfig(),
    )
    cache[0][:, physical_block_id].fill_(7)

    manager.free_request("request-1")

    assert torch.all(cache[0][:, physical_block_id] == 7)


def test_release_drops_owned_layer_references() -> None:
    cache = make_cache()

    cache.release()

    assert len(cache) == 0
    with pytest.raises(IndexError):
        cache[0]


def test_validation_rejects_invalid_layer_count() -> None:
    cache = make_cache()
    cache._layer_caches = cache._layer_caches[:1]

    with pytest.raises(ValueError, match="must contain 2 layers"):
        cache.validate()


def test_validation_rejects_invalid_shape() -> None:
    cache = make_cache()
    cache._layer_caches = (
        torch.empty((2, 8, 4, 2, 8)),
        cache[1],
    )

    with pytest.raises(ValueError, match="layer 0 KV cache has invalid shape"):
        cache.validate()


def test_validation_rejects_invalid_dtype() -> None:
    cache = make_cache()
    cache._layer_caches = (
        torch.empty(cache.shape, dtype=torch.float16),
        cache[1],
    )

    with pytest.raises(ValueError, match="layer 0 KV cache has invalid dtype"):
        cache.validate()


def test_validation_rejects_invalid_device() -> None:
    cache = make_cache()
    cache._layer_caches = (
        torch.empty(cache.shape, device="meta"),
        cache[1],
    )

    with pytest.raises(ValueError, match="layer 0 KV cache has invalid device"):
        cache.validate()


@pytest.mark.gpu
def test_gpu_cache_uses_configured_device_and_dtype() -> None:
    cache = make_cache(device="cuda", dtype=torch.float16)

    assert len(cache) == 2
    assert cache.dtype is torch.float16
    assert cache.device.type == "cuda"
    assert all(cache[index].device.type == "cuda" for index in range(len(cache)))
