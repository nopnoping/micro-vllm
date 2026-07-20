"""Explicit device and dtype configuration for model execution."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True, init=False)
class ModelExecutionConfig:
    """Target device and floating-point dtype shared by Worker and ModelRunner."""

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
