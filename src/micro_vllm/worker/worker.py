"""Device resource, model, and physical KV-cache ownership."""

from torch import nn

from micro_vllm.config import CacheConfig
from micro_vllm.core.scheduler_output import SchedulerOutput
from micro_vllm.executor.interface import ModelRunnerOutput
from micro_vllm.layers.attention_backend import ReferenceAttentionBackend
from micro_vllm.config import ModelConfig
from micro_vllm.models.registry import get_model_class
from micro_vllm.worker.config import ModelExecutionConfig
from micro_vllm.worker.kv_cache import PhysicalKVCache
from micro_vllm.worker.model_runner import ModelRunner


class Worker:
    """Own one model, its physical KV cache, and the device-side runner."""

    __slots__ = ("_model", "_model_runner", "_physical_kv_cache")

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        execution_config: ModelExecutionConfig,
    ) -> None:
        model_class = get_model_class(
            model_config,
            source_label="worker model config",
        )
        model = model_class(model_config, ReferenceAttentionBackend()).to(
            device=execution_config.device,
            dtype=execution_config.dtype,
        )
        model.eval()
        physical_kv_cache = PhysicalKVCache(
            cache_config,
            model_config,
            execution_config,
        )

        self._model: nn.Module | None = model
        self._physical_kv_cache: PhysicalKVCache | None = physical_kv_cache
        self._model_runner: ModelRunner | None = ModelRunner(
            model,
            physical_kv_cache,
            device=execution_config.device,
            dtype=execution_config.dtype,
            block_size=cache_config.block_size,
        )

    def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput:
        """Delegate an immutable plan to the owned ModelRunner."""

        model_runner = self._model_runner
        if model_runner is None:
            raise RuntimeError("worker is shut down")
        return model_runner.execute(plan)

    def shutdown(self) -> None:
        """Release owned references without touching global device allocator state."""

        physical_kv_cache = self._physical_kv_cache
        if physical_kv_cache is None:
            return

        self._model_runner = None
        physical_kv_cache.release()
        self._physical_kv_cache = None
        self._model = None
