"""Execution-plan to model-input preparation and model invocation."""

import gc
import math
from dataclasses import dataclass
from typing import Protocol, cast

import torch

from micro_vllm.attention.attention import (
    Attention,
    AttentionMetadata,
    map_cache_slots,
)
from micro_vllm.config import LLMConfig, ModelConfig
from micro_vllm.core.kv_cache_spec import (
    KVCacheLayout,
    KVCacheTopology,
    build_kv_cache_topology,
)
from micro_vllm.core.scheduler_output import (
    CacheGroupBlockTable,
    ScheduledRequest,
    SchedulerOutput,
)
from micro_vllm.executor.interface import ModelRunnerOutput
from micro_vllm.model_loader import load_model
from micro_vllm.sampling.params import SamplingParams
from micro_vllm.sampling.sampler import Sampler
from micro_vllm.trace import KVCacheLayoutTrace


class _RunnerModel(Protocol):
    """Minimal model declarations and operations required by ModelRunner."""

    layer_attentions: tuple[Attention, ...]
    config: ModelConfig

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        layer_kv_caches: tuple[torch.Tensor, ...],
        layer_attention_metadata: tuple[AttentionMetadata, ...],
    ) -> torch.Tensor: ...

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True, eq=False)
class ModelInput:
    """Flattened tensor batch consumed internally by ModelRunner."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    layer_attention_metadata: tuple[AttentionMetadata, ...]
    request_ids: tuple[str, ...]
    sampling_row_indices: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "layer_attention_metadata",
            tuple(self.layer_attention_metadata),
        )
        object.__setattr__(self, "request_ids", tuple(self.request_ids))


@dataclass(slots=True)
class _CudaGraphExecution:
    """One captured mixed-batch capacity and its fixed-address storage."""

    graph: torch.cuda.CUDAGraph
    model_input: ModelInput
    hidden_states: torch.Tensor
    null_block_id: int

    def copy_input(self, model_input: ModelInput) -> None:
        """Copy an actual mixed batch and pad the remaining capacity."""

        target = self.model_input
        num_tokens = model_input.input_ids.numel()
        num_requests = len(model_input.request_ids)

        target.input_ids.zero_()
        target.input_ids[:num_tokens].copy_(model_input.input_ids)
        target.positions.zero_()
        target.positions[:num_tokens].copy_(model_input.positions)

        target_metadata = target.layer_attention_metadata[0]
        source_metadata = model_input.layer_attention_metadata[0]
        target_metadata.query_start_locations.fill_(num_tokens)
        target_metadata.query_start_locations[: num_requests + 1].copy_(
            source_metadata.query_start_locations
        )
        target_metadata.context_lengths.zero_()
        target_metadata.context_lengths[:num_requests].copy_(
            source_metadata.context_lengths
        )

        copied: set[int] = set()
        for target_layer, source_layer in zip(
            target.layer_attention_metadata,
            model_input.layer_attention_metadata,
            strict=True,
        ):
            if id(target_layer) in copied:
                continue
            copied.add(id(target_layer))
            target_layer.query_context_start_positions.zero_()
            target_layer.query_context_start_positions[:num_tokens].copy_(
                source_layer.query_context_start_positions
            )
            target_layer.block_table.fill_(self.null_block_id)
            target_layer.block_table[
                :num_requests,
                : source_layer.block_table.shape[1],
            ].copy_(source_layer.block_table)
            target_layer.slot_mapping.fill_(-1)
            target_layer.slot_mapping[:num_tokens].copy_(
                source_layer.slot_mapping
            )


class ModelRunner:
    """Own model initialization and execute plans against physical KV cache."""

    __slots__ = (
        "_available_kv_cache_memory_bytes",
        "_block_size",
        "_config",
        "_cuda_graph_executions",
        "_device",
        "_dtype",
        "_is_shutdown",
        "_model",
        "_kv_cache_layout",
        "_kv_cache_buffers",
        "_kv_cache_topology",
        "_peak_activation_memory_bytes",
        "_sampler",
    )

    # Model initialization

    def __init__(self, config: LLMConfig) -> None:
        self._is_shutdown = False
        self._model: _RunnerModel | None = None
        self._kv_cache_topology: KVCacheTopology | None = None
        self._kv_cache_layout: KVCacheLayout | None = None
        self._kv_cache_buffers: tuple[torch.Tensor, ...] | None = None
        self._config = config
        self._cuda_graph_executions: dict[int, _CudaGraphExecution] = {}
        self._device = torch.device(config.model_config.device)
        self._dtype = config.model_config.dtype
        self._block_size = config.cache_config.block_size
        self._available_kv_cache_memory_bytes: int | None = None
        self._peak_activation_memory_bytes = 0
        self._sampler = Sampler()

    def initialize_model(self) -> None:
        """Construct, place, and retain the execution model exactly once."""

        if self._is_shutdown:
            raise RuntimeError("model runner is shut down")
        if self._model is not None:
            raise RuntimeError("model is already initialized")

        model = load_model(self._config)
        model_config = self._config.model_config
        if model_config.metadata is None:
            model_config.set_metadata(model.config.values)
        model.config = model_config
        model.to(device=self._device, dtype=self._dtype)
        model.eval()
        runner_model = cast(_RunnerModel, model)
        self._kv_cache_topology = build_kv_cache_topology(
            tuple(attention.spec for attention in runner_model.layer_attentions)
        )
        self._model = runner_model

    # KV cache initialization

    def initialize_kv_cache(self) -> None:
        """Warm up the model, derive capacity, and allocate physical KV tensors."""

        model = self._model
        if model is None:
            raise RuntimeError("model must be initialized before KV cache")
        if self._kv_cache_buffers is not None:
            raise RuntimeError("KV cache is already initialized")

        available_memory_bytes = self._profile_model_memory()
        num_blocks = self._determine_num_blocks(
            available_memory_bytes=available_memory_bytes,
        )
        layout = KVCacheLayout(
            block_size=self._block_size,
            num_blocks=num_blocks,
            topology=cast(KVCacheTopology, self._kv_cache_topology),
        )
        self._config.cache_config.num_blocks = num_blocks
        kv_cache_buffers = self._allocate_kv_cache(layout)
        cuda_graph_executions = self._capture_cuda_graphs(
            layout,
            kv_cache_buffers,
        )
        self._kv_cache_layout = layout
        self._kv_cache_buffers = kv_cache_buffers
        self._cuda_graph_executions = cuda_graph_executions
        self._available_kv_cache_memory_bytes = available_memory_bytes
        trace_sink = self._config.trace_sink
        if trace_sink is not None:
            trace_sink(
                KVCacheLayoutTrace(
                    page_layer_count=layout.page_layer_count,
                    physical_buffer_count=len(kv_cache_buffers),
                    num_blocks=layout.num_blocks,
                    block_size=layout.block_size,
                    dtype=str(self._dtype),
                    total_bytes=sum(
                        buffer.numel() * buffer.element_size()
                        for buffer in kv_cache_buffers
                    ),
                )
            )

    @property
    def kv_cache_layout(self) -> KVCacheLayout:
        """Initialized logical and physical cache layout."""

        return cast(KVCacheLayout, self._kv_cache_layout)

    # Model execution

    def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput:
        """Execute a flattened batch without retaining scheduler objects."""

        model = self._model
        kv_cache_buffers = self._kv_cache_buffers
        if model is None or kv_cache_buffers is None:
            raise RuntimeError("model runner is not initialized")
        null_block_id = (
            cast(KVCacheLayout, self._kv_cache_layout).num_blocks - 1
        )
        model_input = self._build_model_input(plan, null_block_id)
        graph_execution = self._find_cuda_graph(
            model_input.input_ids.numel()
        )
        layer_kv_caches = self._bind_layer_kv_caches(kv_cache_buffers)

        with torch.inference_mode():
            if graph_execution is None:
                hidden_states = self._forward_model(
                    model_input,
                    layer_kv_caches,
                )
            else:
                graph_execution.copy_input(model_input)
                graph_execution.graph.replay()
                hidden_states = graph_execution.hidden_states
            logits = self._compute_logits(
                hidden_states,
                model_input.sampling_row_indices,
            )

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

    def _run_model(
        self,
        model_input: ModelInput,
        layer_kv_caches: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Run one eager model-to-logits execution."""

        hidden_states = self._forward_model(model_input, layer_kv_caches)
        return self._compute_logits(
            hidden_states,
            model_input.sampling_row_indices,
        )

    def _forward_model(
        self,
        model_input: ModelInput,
        layer_kv_caches: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Run the model section captured by CUDA Graph."""

        model = cast(_RunnerModel, self._model)
        return model(
            model_input.input_ids,
            model_input.positions,
            layer_kv_caches,
            model_input.layer_attention_metadata,
        )

    def _compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_row_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Select actual sampling rows outside CUDA Graph."""

        model = cast(_RunnerModel, self._model)
        sampling_hidden_states = hidden_states.index_select(
            0,
            sampling_row_indices,
        )
        return model.compute_logits(sampling_hidden_states)

    def _find_cuda_graph(
        self,
        num_tokens: int,
    ) -> _CudaGraphExecution | None:
        """Return the smallest captured token capacity that fits the batch."""

        if num_tokens == 0:
            return None
        for graph_size in sorted(self._cuda_graph_executions):
            if num_tokens <= graph_size:
                return self._cuda_graph_executions[graph_size]
        return None

    def _capture_cuda_graphs(
        self,
        layout: KVCacheLayout,
        kv_cache_buffers: tuple[torch.Tensor, ...],
    ) -> dict[int, _CudaGraphExecution]:
        """Capture mixed-batch token capacities during initialization."""

        if not self._config.model_config.enable_cuda_graph:
            return {}

        layer_kv_caches = self._bind_layer_kv_caches(kv_cache_buffers)
        pool = torch.cuda.graph_pool_handle()
        executions: dict[int, _CudaGraphExecution] = {}
        for num_tokens in reversed(self._cuda_graph_sizes()):
            num_requests = min(
                num_tokens,
                self._config.scheduler_config.max_num_seqs,
            )
            model_input = self._make_cuda_graph_input(
                num_tokens,
                num_requests,
                layout.num_blocks - 1,
            )
            executions[num_tokens] = self._capture_cuda_graph(
                model_input,
                layer_kv_caches,
                pool,
                layout.num_blocks - 1,
            )
        return executions

    def _cuda_graph_sizes(self) -> tuple[int, ...]:
        """Use power-of-two token buckets plus the configured maximum."""

        max_tokens = self._config.scheduler_config.max_num_batched_tokens
        sizes: list[int] = []
        size = 1
        while size < max_tokens:
            sizes.append(size)
            size *= 2
        sizes.append(max_tokens)
        return tuple(sizes)

    def _make_cuda_graph_input(
        self,
        num_tokens: int,
        num_requests: int,
        null_block_id: int,
    ) -> ModelInput:
        """Create one fully padded mixed-batch input with fixed capacities."""

        positions = torch.zeros(
            num_tokens,
            dtype=torch.int64,
            device=self._device,
        )
        query_start_locations = torch.zeros(
            num_requests + 1,
            dtype=torch.int64,
            device=self._device,
        )
        context_lengths = torch.zeros(
            num_requests,
            dtype=torch.int64,
            device=self._device,
        )
        block_table_width = math.ceil(
            self._config.scheduler_config.max_model_len
            / self._block_size
        )
        topology = cast(KVCacheTopology, self._kv_cache_topology)
        group_metadata = tuple(
            AttentionMetadata(
                query_positions=positions,
                query_context_start_positions=torch.zeros_like(positions),
                query_start_locations=query_start_locations,
                context_lengths=context_lengths,
                block_table=torch.full(
                    (num_requests, block_table_width),
                    null_block_id,
                    dtype=torch.int64,
                    device=self._device,
                ),
                slot_mapping=torch.full(
                    (num_tokens,),
                    -1,
                    dtype=torch.int64,
                    device=self._device,
                ),
                block_size=self._block_size,
            )
            for _ in topology.groups
        )
        layer_metadata = [group_metadata[0]] * topology.num_layers
        for metadata, group in zip(
            group_metadata,
            topology.groups,
            strict=True,
        ):
            for layer_index in group.layer_indices:
                layer_metadata[layer_index] = metadata

        return ModelInput(
            input_ids=torch.zeros(
                num_tokens,
                dtype=torch.int64,
                device=self._device,
            ),
            positions=positions,
            layer_attention_metadata=tuple(layer_metadata),
            request_ids=tuple(
                f"__cuda_graph__-{index}"
                for index in range(num_requests)
            ),
            sampling_row_indices=torch.empty(
                0,
                dtype=torch.int64,
                device=self._device,
            ),
        )

    def _capture_cuda_graph(
        self,
        model_input: ModelInput,
        layer_kv_caches: tuple[torch.Tensor, ...],
        pool: tuple[int, int],
        null_block_id: int,
    ) -> _CudaGraphExecution:
        """Warm up and capture one padded mixed-batch execution."""

        current_stream = torch.cuda.current_stream(self._device)
        warmup_stream = torch.cuda.Stream(device=self._device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                warmup_hidden_states = self._forward_model(
                    model_input,
                    layer_kv_caches,
                )
        current_stream.wait_stream(warmup_stream)
        del warmup_hidden_states

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            hidden_states = self._forward_model(
                model_input,
                layer_kv_caches,
            )
        return _CudaGraphExecution(
            graph=graph,
            model_input=model_input,
            hidden_states=hidden_states,
            null_block_id=null_block_id,
        )

    # Resource release

    def shutdown(self) -> None:
        """Drop model and cache references owned by the runner."""

        self._cuda_graph_executions = {}
        self._kv_cache_buffers = None
        self._kv_cache_layout = None
        self._kv_cache_topology = None
        self._model = None
        self._is_shutdown = True

    # KV cache profiling and allocation

    def _profile_model_memory(self) -> int | None:
        profile_plan, num_profile_blocks = self._make_warmup_plan()
        profile_cache = self._allocate_kv_cache(
            KVCacheLayout(
                block_size=self._block_size,
                num_blocks=num_profile_blocks,
                topology=cast(KVCacheTopology, self._kv_cache_topology),
            )
        )
        is_cuda = self._device.type == "cuda"
        if is_cuda:
            torch.cuda.synchronize(self._device)
            torch.cuda.empty_cache()
            baseline_allocated_bytes = torch.cuda.memory_allocated(self._device)
            torch.cuda.reset_peak_memory_stats(self._device)

        try:
            self._warmup_model(
                profile_plan,
                profile_cache,
                num_profile_blocks - 1,
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
            del profile_cache
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
        topology = cast(KVCacheTopology, self._kv_cache_topology)
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
        num_groups = len(topology.groups)
        for request_index in range(scheduler_config.max_num_seqs):
            if remaining_tokens == 0:
                break
            num_tokens = min(remaining_tokens, max_tokens_per_request)
            num_blocks = (
                num_tokens + self._block_size - 1
            ) // self._block_size
            block_tables = tuple(
                CacheGroupBlockTable(
                    tuple(
                        range(
                            next_block_id + group_index * num_blocks,
                            next_block_id + (group_index + 1) * num_blocks,
                        )
                    )
                )
                for group_index in range(num_groups)
            )
            next_block_id += num_groups * num_blocks
            scheduled_requests.append(
                ScheduledRequest(
                    request_id=f"__warmup__-{request_index}",
                    input_token_ids=(0,) * num_tokens,
                    num_computed_tokens=0,
                    num_scheduled_tokens=num_tokens,
                    requires_sampling=False,
                    block_tables=block_tables,
                )
            )
            remaining_tokens -= num_tokens
        return SchedulerOutput(tuple(scheduled_requests)), next_block_id + 1

    def _warmup_model(
        self,
        plan: SchedulerOutput,
        profile_cache: tuple[torch.Tensor, ...],
        null_block_id: int,
    ) -> None:
        model_input = self._build_model_input(plan, null_block_id)
        model = cast(_RunnerModel, self._model)
        layer_kv_caches = self._bind_layer_kv_caches(profile_cache)
        with torch.inference_mode():
            hidden_states = model(
                model_input.input_ids,
                model_input.positions,
                layer_kv_caches,
                model_input.layer_attention_metadata,
            )
            first_metadata = model_input.layer_attention_metadata[0]
            sampling_row_indices = first_metadata.query_start_locations[1:] - 1
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

        topology = cast(KVCacheTopology, self._kv_cache_topology)
        bytes_per_block = (
            topology.page_layer_count
            * math.prod(self._kv_cache_shape(1))
            * torch.empty((), dtype=self._dtype).element_size()
        )
        num_blocks = available_memory_bytes // bytes_per_block
        if requested_num_blocks is not None:
            num_blocks = min(num_blocks, requested_num_blocks)
        if num_blocks < 2:
            raise MemoryError(
                "available GPU memory cannot fit data and null KV cache blocks"
            )
        return num_blocks

    def _allocate_kv_cache(
        self,
        layout: KVCacheLayout,
    ) -> tuple[torch.Tensor, ...]:
        shape = self._kv_cache_shape(layout.num_blocks)
        return tuple(
            torch.empty(shape, dtype=self._dtype, device=self._device)
            for _ in range(layout.page_layer_count)
        )

    def _kv_cache_shape(self, num_blocks: int) -> tuple[int, ...]:
        values = self._config.model_config.values
        num_attention_heads = cast(int, values["num_attention_heads"])
        num_key_value_heads = cast(
            int, values.get("num_key_value_heads", num_attention_heads)
        )
        head_dim = values.get("head_dim")
        if head_dim is None:
            head_dim = cast(int, values["hidden_size"]) // num_attention_heads
        return (
            2,
            num_blocks,
            self._block_size,
            num_key_value_heads,
            cast(int, head_dim),
        )

    # Model input construction

    def _bind_layer_kv_caches(
        self,
        kv_cache_buffers: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Expand shared page buffers into model layer order."""

        topology = cast(KVCacheTopology, self._kv_cache_topology)
        layer_kv_caches = [kv_cache_buffers[0]] * topology.num_layers
        for group in topology.groups:
            for group_layer_index, layer_index in enumerate(
                group.layer_indices
            ):
                layer_kv_caches[layer_index] = kv_cache_buffers[
                    group_layer_index
                ]
        return tuple(layer_kv_caches)

    def _build_model_input(
        self,
        plan: SchedulerOutput,
        null_block_id: int,
    ) -> ModelInput:
        topology = cast(KVCacheTopology, self._kv_cache_topology)
        block_tables = tuple(
            self._build_block_table(
                plan,
                group_index,
                null_block_id=null_block_id,
            )
            for group_index in range(len(topology.groups))
        )
        slot_mappings = tuple(
            self._build_slot_mapping(
                plan,
                group_index,
            )
            for group_index in range(len(topology.groups))
        )
        return self._flatten_model_input(
            plan,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
        )

    def _flatten_model_input(
        self,
        plan: SchedulerOutput,
        *,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: tuple[torch.Tensor, ...],
    ) -> ModelInput:
        """Build shared batch tensors and per-layer attention metadata."""

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
                    request.num_computed_tokens
                    + request.num_scheduled_tokens,
                )
            )
            query_start_locations.append(len(input_ids))
            context_lengths.append(
                request.num_computed_tokens + request.num_scheduled_tokens
            )
            request_ids.append(request.request_id)
            if request.requires_sampling:
                sampling_row_indices.append(len(input_ids) - 1)

        query_positions = torch.tensor(
            positions,
            dtype=torch.int64,
            device=self._device,
        )
        query_start_locations_tensor = torch.tensor(
            query_start_locations,
            dtype=torch.int64,
            device=self._device,
        )
        context_lengths_tensor = torch.tensor(
            context_lengths,
            dtype=torch.int64,
            device=self._device,
        )
        topology = cast(KVCacheTopology, self._kv_cache_topology)
        model = cast(_RunnerModel, self._model)
        group_metadata = tuple(
            model.layer_attentions[group.layer_indices[0]].build_metadata(
                query_positions=query_positions,
                query_start_locations=query_start_locations_tensor,
                context_lengths=context_lengths_tensor,
                block_table=block_tables[group_index],
                slot_mapping=slot_mappings[group_index],
                block_size=self._block_size,
            )
            for group_index, group in enumerate(topology.groups)
        )
        layer_metadata = [group_metadata[0]] * topology.num_layers
        for group_index, group in enumerate(topology.groups):
            for layer_index in group.layer_indices:
                layer_metadata[layer_index] = group_metadata[group_index]

        return ModelInput(
            input_ids=torch.tensor(
                input_ids,
                dtype=torch.int64,
                device=self._device,
            ),
            positions=query_positions,
            layer_attention_metadata=tuple(layer_metadata),
            request_ids=tuple(request_ids),
            sampling_row_indices=torch.tensor(
                sampling_row_indices,
                dtype=torch.int64,
                device=self._device,
            ),
        )

    def _build_slot_mapping(
        self,
        plan: SchedulerOutput,
        cache_group_index: int,
    ) -> torch.Tensor:
        """Map scheduled positions to flattened physical cache slots."""

        slot_mappings: list[torch.Tensor] = []
        for request in plan.scheduled_requests:
            start_position = request.num_computed_tokens
            end_position = start_position + request.num_scheduled_tokens
            position_tensor = torch.arange(
                start_position,
                end_position,
                dtype=torch.int64,
                device=self._device,
            )
            block_table = torch.tensor(
                request.block_tables[cache_group_index].block_ids,
                dtype=torch.int64,
                device=self._device,
            )
            slot_mappings.append(
                map_cache_slots(
                    position_tensor,
                    block_table,
                    self._block_size,
                )
            )

        if not slot_mappings:
            return torch.empty(
                0,
                dtype=torch.int64,
                device=self._device,
            )
        return torch.cat(slot_mappings)

    def _build_block_table(
        self,
        plan: SchedulerOutput,
        cache_group_index: int,
        *,
        null_block_id: int,
    ) -> torch.Tensor:
        """Pad physical block tables to one rectangular int64 tensor."""

        max_num_blocks = max(
            (
                len(request.block_tables[cache_group_index].block_ids)
                for request in plan.scheduled_requests
            ),
            default=0,
        )
        block_table = torch.full(
            (len(plan.scheduled_requests), max_num_blocks),
            null_block_id,
            dtype=torch.int64,
            device=self._device,
        )
        for row_index, request in enumerate(plan.scheduled_requests):
            block_ids = request.block_tables[cache_group_index].block_ids
            block_table[row_index, : len(block_ids)] = torch.tensor(
                block_ids,
                dtype=torch.int64,
                device=self._device,
            )
        return block_table
