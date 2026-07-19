"""Request lifecycle orchestration for the inference engine."""

from micro_vllm.core.scheduler import Scheduler
from micro_vllm.core.request import Request
from micro_vllm.engine.output import RequestOutput
from micro_vllm.executor.interface import Executor
from micro_vllm.sampling.params import SamplingParams


class Engine:
    """Coordinate the scheduler and executor through a synchronous control loop."""

    def __init__(self, scheduler: Scheduler, executor: Executor) -> None:
        self._scheduler = scheduler
        self._executor = executor

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> None:
        """Add one caller-identified tokenized request."""

        self._scheduler.add_request(
            Request(
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
            )
        )

    def step(self) -> tuple[RequestOutput, ...]:
        """Run one schedule, execute, and update control-loop iteration."""

        plan = self._scheduler.schedule()
        if not plan.scheduled_requests:
            return ()

        result = self._executor.execute(plan)
        self._scheduler.update(plan, result)

        outputs: list[RequestOutput] = []
        for scheduled_request in plan.scheduled_requests:
            if not scheduled_request.requires_sampling:
                continue
            request = self._scheduler.get_request(scheduled_request.request_id)
            outputs.append(
                RequestOutput(
                    request_id=request.request_id,
                    output_token_ids=tuple(request.output_token_ids),
                    finished=request.is_finished,
                    finish_reason=request.finish_reason,
                )
            )
        return tuple(outputs)

    def has_unfinished_requests(self) -> bool:
        """Whether the scheduler owns any request that is not finished."""

        return self._scheduler.has_unfinished_requests()

    def abort_request(self, request_id: str) -> bool:
        """Abort a request, returning whether its state changed."""

        return self._scheduler.abort_request(request_id)
