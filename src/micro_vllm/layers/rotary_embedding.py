"""Rotary position embedding implementations."""

import torch
from torch import nn


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the final dimension."""

    first_half, second_half = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary embedding for flattened absolute positions."""

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequency_indices = torch.arange(
            0,
            self.head_dim,
            2,
            dtype=torch.float32,
            device=positions.device,
        )
        inverse_frequencies = 1.0 / (
            self.base ** (frequency_indices / self.head_dim)
        )
        frequencies = torch.outer(
            positions.to(torch.float32),
            inverse_frequencies,
        )
        angles = torch.cat((frequencies, frequencies), dim=-1).unsqueeze(1)
        cosine = angles.cos()
        sine = angles.sin()

        float_query = query.to(torch.float32)
        float_key = key.to(torch.float32)
        rotated_query = (
            float_query * cosine + rotate_half(float_query) * sine
        )
        rotated_key = float_key * cosine + rotate_half(float_key) * sine
        return rotated_query.to(query.dtype), rotated_key.to(key.dtype)
