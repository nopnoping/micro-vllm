"""Tests for ModelRunner-owned physical KV-cache tensors."""

import pytest
import torch

from micro_vllm.attention.spec import FullAttentionSpec, SlidingWindowSpec
from micro_vllm.config import CacheConfig, LLMConfig, ModelConfig
from micro_vllm.core.kv_cache_manager import KVCacheManager
from micro_vllm.core.kv_cache_spec import KVCacheLayout, build_kv_cache_topology
from micro_vllm.worker.model_runner import ModelRunner


def make_model_config() -> ModelConfig:
    return ModelConfig(
        model="unused",
        metadata={
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 256,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
        },
    )


def make_cache(
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    config = LLMConfig(
        model_config=ModelConfig(
            model="unused",
            metadata=make_model_config().values,
            device=device,
            dtype=dtype,
        ),
        cache_config=CacheConfig(block_size=4, num_blocks=8),
    )
    layout = KVCacheLayout(
        block_size=4,
        num_blocks=8,
        topology=build_kv_cache_topology(
            (FullAttentionSpec(), FullAttentionSpec())
        ),
    )
    return ModelRunner(torch.nn.Module, config)._allocate_kv_cache(layout)


def test_cpu_cache_uses_model_cache_and_execution_configuration() -> None:
    cache = make_cache()
    first_buffer = cache[0]

    assert len(cache) == 2
    assert first_buffer.shape == torch.Size((2, 8, 4, 2, 16))
    assert first_buffer.dtype is torch.float32
    assert first_buffer.device == torch.device("cpu")
    for buffer in (cache[0], cache[1]):
        assert buffer.shape == first_buffer.shape
        assert buffer.dtype is first_buffer.dtype
        assert buffer.device == first_buffer.device


def test_cache_shape_uses_explicit_model_head_dimension() -> None:
    model_config = ModelConfig(
        model="unused",
        metadata={
            "architectures": ["Qwen3ForCausalLM"],
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
        },
    )
    config = LLMConfig(
        model_config=ModelConfig(
            model="unused",
            metadata=model_config.values,
            device="cpu",
            dtype=torch.float32,
        ),
        cache_config=CacheConfig(block_size=4, num_blocks=3),
    )

    layout = KVCacheLayout(
        block_size=4,
        num_blocks=3,
        topology=build_kv_cache_topology(
            (FullAttentionSpec(), FullAttentionSpec())
        ),
    )
    runner = ModelRunner(torch.nn.Module, config)
    cache = runner._allocate_kv_cache(layout)

    assert cache[0].shape == torch.Size((2, 3, 4, 2, 8))


def test_null_model_head_dimension_uses_hidden_size_fallback() -> None:
    config = LLMConfig(
        model_config=ModelConfig(
            model="unused",
            metadata={
                "architectures": ["MistralForCausalLM"],
                "hidden_size": 16,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": None,
            },
            device="cpu",
            dtype=torch.float32,
        ),
        cache_config=CacheConfig(block_size=4, num_blocks=3),
    )
    layout = KVCacheLayout(
        block_size=4,
        num_blocks=3,
        topology=build_kv_cache_topology(
            (FullAttentionSpec(), FullAttentionSpec())
        ),
    )

    runner = ModelRunner(torch.nn.Module, config)
    cache = runner._allocate_kv_cache(layout)

    assert cache[0].shape == torch.Size((2, 3, 4, 2, 4))


def test_cache_uses_max_group_layers_instead_of_total_model_layers() -> None:
    model_config = make_model_config()
    model_config.set_metadata(
        {
            **model_config.values,
            "num_hidden_layers": 6,
        }
    )
    config = LLMConfig(
        model_config=ModelConfig(
            model="unused",
            metadata=model_config.values,
            device="cpu",
            dtype=torch.float32,
        ),
        cache_config=CacheConfig(block_size=4, num_blocks=8),
    )
    layout = KVCacheLayout(
        block_size=4,
        num_blocks=8,
        topology=build_kv_cache_topology(
            (
                FullAttentionSpec(),
                FullAttentionSpec(),
                SlidingWindowSpec(4),
                SlidingWindowSpec(4),
                SlidingWindowSpec(4),
                SlidingWindowSpec(4),
            )
        ),
    )

    runner = ModelRunner(torch.nn.Module, config)
    cache = runner._allocate_kv_cache(layout)

    assert layout.page_layer_count == 4
    assert len(cache) == 4


def test_physical_block_id_indexes_the_block_dimension() -> None:
    cache = make_cache()
    layer_cache = cache[0]
    layer_cache.zero_()

    physical_block_id = 3
    layer_cache[:, physical_block_id].fill_(7)

    assert torch.count_nonzero(layer_cache[:, physical_block_id]).item() > 0
    assert torch.count_nonzero(layer_cache[:, :physical_block_id]).item() == 0
    assert torch.count_nonzero(layer_cache[:, physical_block_id + 1 :]).item() == 0


def test_physical_buffers_do_not_share_storage() -> None:
    cache = make_cache()

    assert cache[0].data_ptr() != cache[1].data_ptr()


def test_logical_block_release_does_not_clear_physical_storage() -> None:
    config = LLMConfig(
        model_config=ModelConfig(
            model="unused",
            metadata=make_model_config().values,
            device="cpu",
            dtype=torch.float32,
        ),
        cache_config=CacheConfig(block_size=4, num_blocks=8),
    )
    manager = KVCacheManager(config)
    manager.register_request("request-1")
    allocation = manager.allocate_slots("request-1", 0, 1)
    assert allocation is not None
    physical_block_id = allocation.block_tables[0].block_ids[0]
    layout = KVCacheLayout(
        block_size=4,
        num_blocks=8,
        topology=build_kv_cache_topology(
            (FullAttentionSpec(), FullAttentionSpec())
        ),
    )
    cache = ModelRunner(torch.nn.Module, config)._allocate_kv_cache(layout)
    cache[0][:, physical_block_id].fill_(7)

    manager.free_request("request-1")

    assert torch.all(cache[0][:, physical_block_id] == 7)


@pytest.mark.gpu
def test_gpu_cache_uses_configured_device_and_dtype() -> None:
    cache = make_cache(device="cuda", dtype=torch.float16)

    assert len(cache) == 2
    assert cache[0].dtype is torch.float16
    assert cache[0].device.type == "cuda"
    assert all(cache[index].device.type == "cuda" for index in range(len(cache)))
