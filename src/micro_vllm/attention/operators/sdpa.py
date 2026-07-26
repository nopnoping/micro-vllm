"""PyTorch SDPA over K/V gathered from the paged cache."""

import torch
import torch.nn.functional as F

from micro_vllm.attention.attention import AttentionMetadata
from micro_vllm.attention.operators.base import AttentionOperator
from micro_vllm.attention.operators.reference import (
    gather_kv_cache,
    write_kv_cache,
)


def visible_mask(
    query_positions: torch.Tensor,
    query_context_starts: torch.Tensor,
    context_positions: torch.Tensor,
) -> torch.Tensor:
    """Return SDPA's boolean keep-mask for semantic visibility bounds."""

    return (
        context_positions.unsqueeze(0)
        >= query_context_starts.unsqueeze(1)
    ) & (
        context_positions.unsqueeze(0)
        <= query_positions.unsqueeze(1)
    )


def prepare_sdpa_inputs(
    query: torch.Tensor,
    cached_key: torch.Tensor,
    cached_value: torch.Tensor,
    query_positions: torch.Tensor,
    query_context_starts: torch.Tensor,
    context_start: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert one gathered request to dense SDPA tensors and keep-mask."""

    head_group_size = query.shape[1] // cached_key.shape[1]
    if head_group_size > 1:
        cached_key = cached_key.repeat_interleave(head_group_size, dim=1)
        cached_value = cached_value.repeat_interleave(head_group_size, dim=1)

    context_positions = torch.arange(
        context_start,
        context_start + cached_key.shape[0],
        dtype=torch.int64,
        device=query.device,
    )
    mask = visible_mask(
        query_positions,
        query_context_starts,
        context_positions,
    )
    return (
        query.transpose(0, 1).unsqueeze(0),
        cached_key.transpose(0, 1).unsqueeze(0),
        cached_value.transpose(0, 1).unsqueeze(0),
        mask.unsqueeze(0).unsqueeze(0),
    )


def sdpa_core(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Call PyTorch SDPA with the explicit semantic keep-mask."""

    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    )


def paged_sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_kv_cache: torch.Tensor,
    metadata: AttentionMetadata,
) -> torch.Tensor:
    """Gather each request's paged history, then invoke PyTorch SDPA."""

    write_kv_cache(key, value, layer_kv_cache, metadata)
    histories = gather_kv_cache(layer_kv_cache, metadata)
    output = torch.empty_like(query)

    for request_index, (cached_key, cached_value) in enumerate(histories):
        query_start = metadata.query_start_locations[request_index].item()
        query_end = metadata.query_start_locations[request_index + 1].item()
        context_start = metadata.query_context_start_positions[
            query_start
        ].item()
        sdpa_inputs = prepare_sdpa_inputs(
            query[query_start:query_end],
            cached_key,
            cached_value,
            metadata.query_positions[query_start:query_end],
            metadata.query_context_start_positions[query_start:query_end],
            context_start,
        )
        attention = sdpa_core(*sdpa_inputs)
        output[query_start:query_end].copy_(
            attention.squeeze(0).transpose(0, 1)
        )

    return output


class SDPAAttentionOperator(AttentionOperator):
    """Transition backend using gathered paged K/V and PyTorch SDPA."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_kv_cache: torch.Tensor,
        metadata: AttentionMetadata,
        /,
    ) -> torch.Tensor:
        return paged_sdpa_attention(
            query,
            key,
            value,
            layer_kv_cache,
            metadata,
        )
