"""Execution-plan to model-input preparation and model invocation."""

from dataclasses import dataclass
from typing import Protocol

import torch

from micro_vllm.config import CacheConfig
from micro_vllm.core.scheduler_output import SchedulerOutput
from micro_vllm.executor.interface import ModelRunnerOutput
from micro_vllm.layers.attention_backend import (
    AttentionMetadata,
    PAD_BLOCK_ID,
    map_cache_slots,
)
from micro_vllm.sampling.sampler import GreedySampler
from micro_vllm.worker.kv_cache import PhysicalKVCache


class _RunnerModel(Protocol):
    """Model operations required by ModelRunner."""

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        layer_kv_caches: tuple[torch.Tensor, ...],
        attention_metadata: AttentionMetadata,
    ) -> torch.Tensor: ...

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...


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


def build_slot_mapping(
    plan: SchedulerOutput,
    cache_config: CacheConfig,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Map every scheduled position to one flattened physical cache slot."""

    slot_mappings: list[torch.Tensor] = []
    used_slots: set[int] = set()
    for request in plan.scheduled_requests:
        start_position = request.num_computed_tokens
        end_position = start_position + request.num_scheduled_tokens
        positions = range(start_position, end_position)
        for position in positions:
            logical_block = position // cache_config.block_size
            if logical_block >= len(request.block_table):
                raise ValueError(
                    f"request {request.request_id} has no block for position {position}"
                )
            physical_block = request.block_table[logical_block]
            if physical_block is None:
                raise ValueError(
                    f"request {request.request_id} has no physical block "
                    f"for position {position}"
                )
            if not 0 <= physical_block < cache_config.num_blocks:
                raise ValueError(
                    f"request {request.request_id} has out-of-range physical block "
                    f"{physical_block}"
                )

        position_tensor = torch.arange(
            start_position,
            end_position,
            dtype=torch.int64,
            device=device,
        )
        block_table = torch.tensor(
            [
                PAD_BLOCK_ID if block_id is None else block_id
                for block_id in request.block_table
            ],
            dtype=torch.int64,
            device=device,
        )
        request_slots = map_cache_slots(
            position_tensor,
            block_table,
            cache_config.block_size,
        )
        for slot in request_slots.tolist():
            if slot in used_slots:
                raise ValueError(f"duplicate physical cache slot: {slot}")
            used_slots.add(slot)
        slot_mappings.append(request_slots)

    if not slot_mappings:
        return torch.empty(0, dtype=torch.int64, device=device)
    return torch.cat(slot_mappings)


def build_block_table(
    plan: SchedulerOutput,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Pad physical block tables to one rectangular int64 tensor."""

    max_num_blocks = max(
        (len(request.block_table) for request in plan.scheduled_requests),
        default=0,
    )
    block_table = torch.full(
        (len(plan.scheduled_requests), max_num_blocks),
        PAD_BLOCK_ID,
        dtype=torch.int64,
        device=device,
    )
    for row_index, request in enumerate(plan.scheduled_requests):
        block_ids = [
            PAD_BLOCK_ID if block_id is None else block_id
            for block_id in request.block_table
        ]
        block_table[row_index, : len(block_ids)] = torch.tensor(
            block_ids,
            dtype=torch.int64,
            device=device,
        )
    return block_table


class ModelRunner:
    """Execute one immutable scheduler plan against a model and KV cache."""

    __slots__ = (
        "_block_size",
        "_cache_config",
        "_device",
        "_dtype",
        "_model",
        "_physical_kv_cache",
        "_sampler",
    )

    def __init__(
        self,
        model: _RunnerModel,
        physical_kv_cache: PhysicalKVCache,
        *,
        device: torch.device,
        dtype: torch.dtype,
        block_size: int,
    ) -> None:
        self._model = model
        self._physical_kv_cache = physical_kv_cache
        self._device = device
        self._dtype = dtype
        self._block_size = block_size
        self._cache_config = CacheConfig(
            block_size=block_size,
            num_blocks=physical_kv_cache.shape[1],
        )
        self._sampler = GreedySampler()

    def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput:
        """Execute a flattened batch without retaining scheduler objects."""

        model_input = self._build_model_input(plan)
        layer_kv_caches = tuple(
            self._physical_kv_cache[layer_index]
            for layer_index in range(len(self._physical_kv_cache))
        )

        with torch.inference_mode():
            hidden_states = self._model(
                model_input.input_ids,
                model_input.positions,
                layer_kv_caches,
                model_input.attention_metadata,
            )
            sampling_hidden_states = hidden_states.index_select(
                0,
                model_input.sampling_row_indices,
            )
            logits = self._model.compute_logits(sampling_hidden_states)

        sampling_request_ids = tuple(
            request.request_id
            for request in plan.scheduled_requests
            if request.requires_sampling
        )
        return self._sampler(logits, sampling_request_ids)

    def _build_model_input(self, plan: SchedulerOutput) -> ModelInput:
        block_table = build_block_table(plan, device=self._device)
        slot_mapping = build_slot_mapping(
            plan,
            self._cache_config,
            device=self._device,
        )
        return flatten_model_input(
            plan,
            block_table=block_table,
            slot_mapping=slot_mapping,
            block_size=self._block_size,
        )
