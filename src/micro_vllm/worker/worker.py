"""Device-process orchestration for one ModelRunner."""

from micro_vllm.config import LLMConfig
from micro_vllm.core.scheduler_output import SchedulerOutput
from micro_vllm.executor.interface import ModelRunnerOutput
from micro_vllm.worker.model_runner import ModelInitializer, ModelRunner


class Worker:
    """Orchestrate one runner without owning model or KV-cache tensors."""

    __slots__ = ("_config", "_model_runner")

    def __init__(
        self,
        model_initializer: ModelInitializer,
        config: LLMConfig,
    ) -> None:
        model_runner = ModelRunner(model_initializer, config)
        model_runner.initialize_model()
        model_runner.initialize_kv_cache()

        self._config = config
        self._model_runner: ModelRunner | None = model_runner

    @property
    def config(self) -> LLMConfig:
        """Configuration shared with the owned model runner."""

        return self._config

    def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput:
        """Delegate an immutable plan to the owned ModelRunner."""

        model_runner = self._model_runner
        if model_runner is None:
            raise RuntimeError("worker is shut down")
        return model_runner.execute(plan)

    def shutdown(self) -> None:
        """Release owned references without touching global device allocator state."""

        model_runner = self._model_runner
        if model_runner is None:
            return

        self._model_runner = None
        model_runner.shutdown()
