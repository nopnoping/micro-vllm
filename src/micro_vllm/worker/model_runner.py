"""Execution-plan to model-input preparation and model invocation."""

import gc
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol, cast

import torch
from torch import nn

from micro_vllm.config import LLMConfig, ModelConfig
from micro_vllm.core.scheduler_output import ScheduledRequest, SchedulerOutput
from micro_vllm.executor.interface import ModelRunnerOutput
from micro_vllm.layers.attention_backend import (
    AttentionMetadata,
    PAD_BLOCK_ID,
    map_cache_slots,
)
from micro_vllm.sampling.params import SamplingParams
from micro_vllm.sampling.sampler import Sampler
from micro_vllm.worker.kv_cache import (
    PhysicalKVCache,
    kv_cache_block_size_bytes,
)


class _RunnerModel(Protocol):
    """Model operations required by ModelRunner."""

    config: ModelConfig

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        layer_kv_caches: tuple[torch.Tensor, ...],
        attention_metadata: AttentionMetadata,
    ) -> torch.Tensor: ...

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...


ModelInitializer = Callable[[], nn.Module]


@dataclass(frozen=True, slots=True, eq=False)
class ModelInput:
    """Flattened tensor batch consumed internally by ModelRunner."""

    input_ids: torch.Tensor
    attention_metadata: AttentionMetadata
    request_ids: tuple[str, ...]
    sampling_row_indices: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_ids", tuple(self.request_ids))

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
    block_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Map every scheduled position to one flattened physical cache slot."""

    slot_mappings: list[torch.Tensor] = []
    for request in plan.scheduled_requests:
        start_position = request.num_computed_tokens
        end_position = start_position + request.num_scheduled_tokens
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
            block_size,
        )
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
    """Own model initialization and execute plans against physical KV cache."""

    __slots__ = (
        "_available_kv_cache_memory_bytes",
        "_block_size",
        "_config",
        "_device",
        "_dtype",
        "_model",
        "_model_initializer",
        "_peak_activation_memory_bytes",
        "_physical_kv_cache",
        "_sampler",
    )

    def __init__(
        self,
        model_initializer: ModelInitializer,
        config: LLMConfig,
    ) -> None:
        self._model_initializer: ModelInitializer | None = model_initializer
        self._model: _RunnerModel | None = None
        self._physical_kv_cache: PhysicalKVCache | None = None
        self._config = config
        self._device = torch.device(config.model_config.device)
        self._dtype = config.model_config.dtype
        self._block_size = config.cache_config.block_size
        self._available_kv_cache_memory_bytes: int | None = None
        self._peak_activation_memory_bytes = 0
        self._sampler = Sampler()

    def initialize_model(self) -> None:
        """Construct, place, and retain the execution model exactly once."""

        model_initializer = self._model_initializer
        if model_initializer is None:
            if self._model is None:
                raise RuntimeError("model runner is shut down")
            raise RuntimeError("model is already initialized")

        model = model_initializer()
        _validate_model_contract(model, self._config.model_config)
        model.to(device=self._device, dtype=self._dtype)
        model.eval()
        self._model = cast(_RunnerModel, model)
        self._model_initializer = None

    def initialize_kv_cache(self) -> None:
        """Warm up the model, derive capacity, and allocate physical KV tensors."""

        model = self._model
        if model is None:
            raise RuntimeError("model must be initialized before KV cache")
        if self._physical_kv_cache is not None:
            raise RuntimeError("KV cache is already initialized")

        available_memory_bytes = self._profile_model_memory()
        num_blocks = self._determine_num_blocks(
            available_memory_bytes=available_memory_bytes,
        )
        self._config.cache_config.num_blocks = num_blocks
        physical_kv_cache = PhysicalKVCache(self._config)
        self._physical_kv_cache = physical_kv_cache
        self._available_kv_cache_memory_bytes = available_memory_bytes

    def _profile_model_memory(self) -> int | None:
        model = self._model
        if model is None:
            raise RuntimeError("model must be initialized before profiling")

        profile_plan, num_profile_blocks = self._make_warmup_plan()
        profile_config = replace(
            self._config,
            cache_config=replace(
                self._config.cache_config,
                num_blocks=num_profile_blocks,
            ),
        )
        profile_cache = PhysicalKVCache(profile_config)
        is_cuda = self._device.type == "cuda"
        baseline_allocated_bytes = 0
        if is_cuda:
            torch.cuda.synchronize(self._device)
            torch.cuda.empty_cache()
            baseline_allocated_bytes = torch.cuda.memory_allocated(self._device)
            torch.cuda.reset_peak_memory_stats(self._device)

        try:
            self._warmup_model(
                profile_plan,
                profile_cache,
            )
            if is_cuda:
                torch.cuda.synchronize(self._device)
                peak_allocated_bytes = torch.cuda.max_memory_allocated(
                    self._device
                )
                self._peak_activation_memory_bytes = max(
                    peak_allocated_bytes - baseline_allocated_bytes,
                    0,
                )
        finally:
            profile_cache.release()
            gc.collect()
            if is_cuda:
                torch.cuda.empty_cache()

        if not is_cuda:
            self._peak_activation_memory_bytes = 0
            return None

        torch.cuda.synchronize(self._device)
        free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info(
            self._device
        )
        requested_memory_bytes = int(
            total_memory_bytes
            * self._config.cache_config.gpu_memory_utilization
        )
        non_kv_cache_memory_bytes = total_memory_bytes - free_memory_bytes
        available_memory_bytes = (
            requested_memory_bytes
            - non_kv_cache_memory_bytes
            - self._peak_activation_memory_bytes
        )
        if available_memory_bytes <= 0:
            raise MemoryError(
                "warmup left no GPU memory available for the KV cache"
            )
        return available_memory_bytes

    def _make_warmup_plan(self) -> tuple[SchedulerOutput, int]:
        model_config = self._config.model_config
        scheduler_config = self._config.scheduler_config
        max_position_embeddings = model_config.positive_int(
            "max_position_embeddings",
            scheduler_config.max_model_len,
        )
        max_tokens_per_request = min(
            scheduler_config.max_model_len,
            max_position_embeddings,
        )
        remaining_tokens = min(
            scheduler_config.max_num_batched_tokens,
            max_tokens_per_request * scheduler_config.max_num_seqs,
        )
        scheduled_requests: list[ScheduledRequest] = []
        next_block_id = 0
        for request_index in range(scheduler_config.max_num_seqs):
            if remaining_tokens == 0:
                break
            num_tokens = min(remaining_tokens, max_tokens_per_request)
            num_blocks = (
                num_tokens + self._block_size - 1
            ) // self._block_size
            scheduled_requests.append(
                ScheduledRequest(
                    request_id=f"__warmup__-{request_index}",
                    input_token_ids=(0,) * num_tokens,
                    num_computed_tokens=0,
                    num_scheduled_tokens=num_tokens,
                    requires_sampling=False,
                    block_table=tuple(
                        range(next_block_id, next_block_id + num_blocks)
                    ),
                )
            )
            next_block_id += num_blocks
            remaining_tokens -= num_tokens
        return SchedulerOutput(tuple(scheduled_requests)), next_block_id

    def _warmup_model(
        self,
        plan: SchedulerOutput,
        profile_cache: PhysicalKVCache,
    ) -> None:
        model = self._model
        if model is None:
            raise RuntimeError("model must be initialized before warmup")

        block_table = build_block_table(plan, device=self._device)
        slot_mapping = build_slot_mapping(
            plan,
            self._block_size,
            device=self._device,
        )
        model_input = flatten_model_input(
            plan,
            block_table=block_table,
            slot_mapping=slot_mapping,
            block_size=self._block_size,
        )
        layer_kv_caches = tuple(
            profile_cache[layer_index]
            for layer_index in range(len(profile_cache))
        )
        with torch.inference_mode():
            hidden_states = model(
                model_input.input_ids,
                model_input.positions,
                layer_kv_caches,
                model_input.attention_metadata,
            )
            sampling_row_indices = (
                model_input.attention_metadata.query_start_locations[1:] - 1
            )
            sampling_hidden_states = hidden_states.index_select(
                0,
                sampling_row_indices,
            )
            model.compute_logits(sampling_hidden_states)

    def _determine_num_blocks(
        self,
        *,
        available_memory_bytes: int | None,
    ) -> int:
        requested_num_blocks = self._config.cache_config.num_blocks
        if available_memory_bytes is None:
            if requested_num_blocks is None:
                raise ValueError(
                    "CPU execution requires num_blocks because GPU memory "
                    "profiling is unavailable"
                )
            return requested_num_blocks

        bytes_per_block = kv_cache_block_size_bytes(self._config)
        num_blocks = available_memory_bytes // bytes_per_block
        if requested_num_blocks is not None:
            num_blocks = min(num_blocks, requested_num_blocks)
        if num_blocks <= 0:
            raise MemoryError("available GPU memory cannot fit one KV cache block")
        return num_blocks

    def shutdown(self) -> None:
        """Drop model and cache references owned by the runner."""

        physical_kv_cache = self._physical_kv_cache
        if physical_kv_cache is not None:
            physical_kv_cache.release()
        self._physical_kv_cache = None
        self._model = None
        self._model_initializer = None

    def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput:
        """Execute a flattened batch without retaining scheduler objects."""

        model = self._model
        physical_kv_cache = self._physical_kv_cache
        if model is None or physical_kv_cache is None:
            raise RuntimeError("model runner is not initialized")
        model_input = self._build_model_input(plan)
        layer_kv_caches = tuple(
            physical_kv_cache[layer_index]
            for layer_index in range(len(physical_kv_cache))
        )

        with torch.inference_mode():
            hidden_states = model(
                model_input.input_ids,
                model_input.positions,
                layer_kv_caches,
                model_input.attention_metadata,
            )
            sampling_hidden_states = hidden_states.index_select(
                0,
                model_input.sampling_row_indices,
            )
            logits = model.compute_logits(sampling_hidden_states)

        sampling_request_ids = tuple(
            request.request_id
            for request in plan.scheduled_requests
            if request.requires_sampling
        )
        sampling_params = tuple(
            cast(SamplingParams, request.sampling_params)
            for request in plan.scheduled_requests
            if request.requires_sampling
        )
        sample_indices = tuple(
            cast(int, request.sample_index)
            for request in plan.scheduled_requests
            if request.requires_sampling
        )
        return self._sampler(
            logits,
            sampling_request_ids,
            sampling_params,
            sample_indices,
        )

    def _build_model_input(self, plan: SchedulerOutput) -> ModelInput:
        if self._physical_kv_cache is None:
            raise RuntimeError("model runner is not initialized")
        block_table = build_block_table(plan, device=self._device)
        slot_mapping = build_slot_mapping(
            plan,
            self._block_size,
            device=self._device,
        )
        return flatten_model_input(
            plan,
            block_table=block_table,
            slot_mapping=slot_mapping,
            block_size=self._block_size,
        )


def _validate_model_contract(
    model: nn.Module,
    model_config: ModelConfig,
) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("model initializer must return a torch.nn.Module")

    initialized_config = getattr(model, "config", None)
    if not isinstance(initialized_config, ModelConfig):
        raise TypeError("model.config must be a ModelConfig")
    if model_config.metadata is None:
        model_config.set_metadata(initialized_config.values)
    model.config = model_config
    if getattr(model.forward, "__func__", None) is nn.Module.forward:
        raise TypeError("model must implement forward")
    if not callable(getattr(model, "compute_logits", None)):
        raise TypeError("model must implement compute_logits")
