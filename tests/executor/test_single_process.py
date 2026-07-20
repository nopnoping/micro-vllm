"""Tests for the single-process executor delegation boundary."""

import pytest

from micro_vllm.core.scheduler_output import ScheduledRequest, SchedulerOutput
from micro_vllm.executor.interface import ModelRunnerOutput, SampledToken
from micro_vllm.executor.single_process import SingleProcessExecutor


class RecordingWorker:
    def __init__(self, output: ModelRunnerOutput) -> None:
        self.output = output
        self.plans: list[SchedulerOutput] = []
        self.is_shutdown = False

    def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput:
        if self.is_shutdown:
            raise RuntimeError("worker is shut down")
        self.plans.append(plan)
        return self.output

    def shutdown(self) -> None:
        self.is_shutdown = True


def make_plan() -> SchedulerOutput:
    return SchedulerOutput(
        (
            ScheduledRequest(
                request_id="sampling",
                input_token_ids=(3,),
                num_computed_tokens=0,
                num_scheduled_tokens=1,
                requires_sampling=True,
                block_table=(2,),
            ),
            ScheduledRequest(
                request_id="prefill",
                input_token_ids=(4, 5),
                num_computed_tokens=0,
                num_scheduled_tokens=2,
                requires_sampling=False,
                block_table=(7,),
            ),
        )
    )


def test_execute_passes_plan_and_result_through_unchanged() -> None:
    output = ModelRunnerOutput((SampledToken("sampling", 9),))
    worker = RecordingWorker(output)
    executor = SingleProcessExecutor(worker)
    plan = make_plan()
    scheduled_requests = plan.scheduled_requests

    result = executor.execute(plan)

    assert worker.plans == [plan]
    assert worker.plans[0] is plan
    assert result is output
    assert plan.scheduled_requests is scheduled_requests


def test_shutdown_is_delegated_and_execute_then_fails() -> None:
    worker = RecordingWorker(ModelRunnerOutput())
    executor = SingleProcessExecutor(worker)
    plan = make_plan()

    executor.shutdown()

    assert worker.is_shutdown is True
    with pytest.raises(RuntimeError, match="worker is shut down"):
        executor.execute(plan)
