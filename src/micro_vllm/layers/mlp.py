"""Dense feed-forward network layers."""

import torch
from torch import nn
from torch.nn import functional as F

from micro_vllm.layers.linear import Linear


class SwiGLUMLP(nn.Module):
    """Dense bias-free SwiGLU feed-forward network."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(hidden_states))
        return self.down_proj(gate * self.up_proj(hidden_states))
