"""Tests for model-facing Llama attention composition."""

import torch

from micro_vllm.layers.attention import LlamaAttention
from micro_vllm.layers.attention_backend import AttentionMetadata


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_kv_cache: torch.Tensor,
        metadata: AttentionMetadata,
        /,
    ) -> torch.Tensor:
        self.calls.append((query, key, value, layer_kv_cache, metadata))
        return query


def make_metadata() -> AttentionMetadata:
    return AttentionMetadata(
        query_positions=torch.tensor([0, 1, 2], dtype=torch.int64),
        query_start_locations=torch.tensor([0, 3], dtype=torch.int64),
        context_lengths=torch.tensor([3], dtype=torch.int64),
        block_table=torch.tensor([[0]], dtype=torch.int64),
        slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int64),
        block_size=4,
    )


def test_llama_attention_projects_rotates_and_invokes_backend() -> None:
    backend = RecordingBackend()
    attention = LlamaAttention(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        rope_theta=10_000.0,
        backend=backend,
    )
    hidden_states = torch.linspace(-1.0, 1.0, steps=3 * 8).reshape(3, 8)
    positions = torch.tensor([0, 1, 2], dtype=torch.int64)
    layer_kv_cache = torch.empty((2, 2, 4, 1, 4))
    metadata = make_metadata()

    output = attention(
        hidden_states,
        positions,
        layer_kv_cache,
        metadata,
    )

    assert len(backend.calls) == 1
    query, key, value, received_cache, received_metadata = backend.calls[0]
    assert isinstance(query, torch.Tensor)
    assert isinstance(key, torch.Tensor)
    assert isinstance(value, torch.Tensor)
    assert query.shape == torch.Size((3, 2, 4))
    assert key.shape == torch.Size((3, 1, 4))
    assert value.shape == torch.Size((3, 1, 4))
    assert received_cache is layer_kv_cache
    assert received_metadata is metadata

    projected_query = attention.q_proj(hidden_states).view(3, 2, 4)
    projected_key = attention.k_proj(hidden_states).view(3, 1, 4)
    projected_value = attention.v_proj(hidden_states).view(3, 1, 4)
    expected_query, expected_key = attention.rotary_embedding(
        positions,
        projected_query,
        projected_key,
    )
    torch.testing.assert_close(query, expected_query, atol=0, rtol=0)
    torch.testing.assert_close(key, expected_key, atol=0, rtol=0)
    torch.testing.assert_close(value, projected_value, atol=0, rtol=0)
    torch.testing.assert_close(
        output,
        attention.o_proj(expected_query.flatten(start_dim=1)),
        atol=0,
        rtol=0,
    )
