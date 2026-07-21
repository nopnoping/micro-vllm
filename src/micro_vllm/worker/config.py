"""Explicit model execution and KV-cache initialization configuration."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True, init=False)
class ModelExecutionConfig:
    """Target device and floating-point dtype consumed by ModelRunner."""

    device: torch.device
    dtype: torch.dtype

    def __init__(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        normalized_device = torch.device(device)
        if normalized_device.type not in {"cpu", "cuda"}:
            raise ValueError("execution device must be 'cpu' or 'cuda'")
        if dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise ValueError(
                "execution dtype must be float32, float16, or bfloat16"
            )

        object.__setattr__(self, "device", normalized_device)
        object.__setattr__(self, "dtype", dtype)


@dataclass(frozen=True, slots=True)
class KVCacheInitializationConfig:
    """Inputs for profiling and allocating one runner's physical KV cache.

    ``max_num_blocks`` is a capacity ceiling on GPU and the deterministic
    capacity used by the CPU reference path, where GPU memory profiling is not
    available. It is not the GPU cache size: CUDA runners derive that from a
    warmup profile before applying the optional ceiling.
    """

    block_size: int
    gpu_memory_utilization: float = 0.9
    max_num_blocks: int | None = None
    enable_prefix_cache: bool = False

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be greater than zero")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError(
                "gpu_memory_utilization must be greater than zero and at most one"
            )
        if self.max_num_blocks is not None and self.max_num_blocks <= 0:
            raise ValueError("max_num_blocks must be greater than zero")
