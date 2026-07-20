"""Execution-plan to model-input preparation and model invocation."""

from dataclasses import dataclass

import torch

from micro_vllm.core.scheduler_output import SchedulerOutput
from micro_vllm.layers.attention_backend import AttentionMetadata


@dataclass(frozen=True, slots=True, eq=False)
class ModelInput:
    """Flattened tensor batch consumed internally by ModelRunner."""

    input_ids: torch.Tensor
    attention_metadata: AttentionMetadata
    request_ids: tuple[str, ...]
    sampling_row_indices: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_ids", tuple(self.request_ids))

        for field_name, tensor in (
            ("input_ids", self.input_ids),
            ("sampling_row_indices", self.sampling_row_indices),
        ):
            if tensor.dtype is not torch.int64:
                raise ValueError(f"{field_name} must use torch.int64")
            if tensor.ndim != 1:
                raise ValueError(f"{field_name} must be one-dimensional")
            if tensor.device != self.positions.device:
                raise ValueError("ModelInput tensors must share one device")

        if self.input_ids.numel() != self.attention_metadata.num_query_tokens:
            raise ValueError(
                "input_ids length must equal num_query_tokens"
            )
        if len(self.request_ids) != self.attention_metadata.num_requests:
            raise ValueError("request_ids length must equal num_requests")

    @property
    def positions(self) -> torch.Tensor:
        """Flattened absolute positions shared with attention metadata."""

        return self.attention_metadata.query_positions


def flatten_model_input(
    plan: SchedulerOutput,
    *,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> ModelInput:
    """Flatten scheduled requests in stable plan order."""

    input_ids: list[int] = []
    positions: list[int] = []
    query_start_locations = [0]
    context_lengths: list[int] = []
    request_ids: list[str] = []
    sampling_row_indices: list[int] = []

    for request in plan.scheduled_requests:
        input_ids.extend(request.input_token_ids)
        positions.extend(
            range(
                request.num_computed_tokens,
                request.num_computed_tokens + request.num_scheduled_tokens,
            )
        )
        query_start_locations.append(len(input_ids))
        context_lengths.append(
            request.num_computed_tokens + request.num_scheduled_tokens
        )
        request_ids.append(request.request_id)
        if request.requires_sampling:
            sampling_row_indices.append(len(input_ids) - 1)

    device = block_table.device
    metadata = AttentionMetadata(
        query_positions=torch.tensor(positions, dtype=torch.int64, device=device),
        query_start_locations=torch.tensor(
            query_start_locations,
            dtype=torch.int64,
            device=device,
        ),
        context_lengths=torch.tensor(
            context_lengths,
            dtype=torch.int64,
            device=device,
        ),
        block_table=block_table,
        slot_mapping=slot_mapping,
        block_size=block_size,
    )
    return ModelInput(
        input_ids=torch.tensor(input_ids, dtype=torch.int64, device=device),
        attention_metadata=metadata,
        request_ids=tuple(request_ids),
        sampling_row_indices=torch.tensor(
            sampling_row_indices,
            dtype=torch.int64,
            device=device,
        ),
    )
