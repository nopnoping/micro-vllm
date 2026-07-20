"""Single-process executor implementation."""

from micro_vllm.core.scheduler_output import SchedulerOutput
from micro_vllm.executor.interface import ModelRunnerOutput
from micro_vllm.worker.worker import Worker


class SingleProcessExecutor:
    """Forward execution and lifecycle calls to one in-process Worker."""

    __slots__ = ("_worker",)

    def __init__(self, worker: Worker) -> None:
        self._worker = worker

    def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput:
        """Execute a plan without interpreting or changing it."""

        return self._worker.execute(plan)

    def shutdown(self) -> None:
        """Release resources owned by the in-process Worker."""

        self._worker.shutdown()
