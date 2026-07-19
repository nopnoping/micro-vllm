"""Scheduling policies and request admission logic."""

from collections import deque

from micro_vllm.config import SchedulerConfig
from micro_vllm.core.kv_cache_manager import KVCacheManager
from micro_vllm.core.request import FinishReason, Request, RequestStatus
from micro_vllm.core.scheduler_output import ScheduledRequest, SchedulerOutput
from micro_vllm.executor.interface import ModelRunnerOutput


class Scheduler:
    """Own request state and build token-budgeted execution plans."""

    def __init__(
        self,
        config: SchedulerConfig,
        kv_cache_manager: KVCacheManager,
        *,
        cache_namespace: str = "default",
    ) -> None:
        self._config = config
        self._kv_cache_manager = kv_cache_manager
        self._cache_namespace = cache_namespace
        self._waiting: deque[Request] = deque()
        self._running: list[Request] = []
        self._requests: dict[str, Request] = {}
        self._pending_plan: SchedulerOutput | None = None

    def add_request(self, request: Request) -> None:
        """Append a new waiting request to the admission queue."""

        if request.status is not RequestStatus.WAITING:
            raise ValueError("request must be in WAITING status")
        if request.request_id in self._requests:
            raise ValueError(f"duplicate request_id: {request.request_id}")

        self._kv_cache_manager.register_request(request.request_id)
        self._requests[request.request_id] = request
        self._waiting.append(request)

    def abort_request(self, request_id: str) -> bool:
        """Abort an active request, returning whether its state changed."""

        request = self._requests.get(request_id)
        if request is None or request.is_finished:
            return False

        if request.status is RequestStatus.WAITING:
            if request not in self._waiting:
                raise RuntimeError("waiting request is missing from the waiting queue")
            request.finish(FinishReason.ABORTED)
            self._waiting.remove(request)
        elif request.status is RequestStatus.RUNNING:
            if request not in self._running:
                raise RuntimeError("running request is missing from the running collection")
            request.finish(FinishReason.ABORTED)
            self._running.remove(request)
        else:
            raise RuntimeError(
                f"unsupported active request status: {request.status.name}"
            )
        if self._pending_plan is not None and any(
            scheduled.request_id == request_id
            for scheduled in self._pending_plan.scheduled_requests
        ):
            self._pending_plan = None
        self._kv_cache_manager.free_request(request_id)
        return True

    def get_request(self, request_id: str) -> Request | None:
        """Return scheduler-owned request state for read-only inspection."""

        return self._requests.get(request_id)

    def has_unfinished_requests(self) -> bool:
        """Whether any scheduler-owned request has not reached a terminal state."""

        return any(not request.is_finished for request in self._requests.values())

    def schedule(self) -> SchedulerOutput:
        """Build an immutable plan for one token-budgeted scheduler step."""

        if self._pending_plan is not None:
            return self._pending_plan

        remaining_token_budget = self._config.max_num_batched_tokens
        scheduled_requests: list[ScheduledRequest] = []
        cache_capacity_exhausted = False

        for request in self._running:
            if request.status is not RequestStatus.RUNNING:
                raise RuntimeError("running collection contains a non-running request")
            if request.num_pending_tokens <= 0:
                raise RuntimeError("active request must have pending tokens")
        for request in self._waiting:
            if request.status is not RequestStatus.WAITING:
                raise RuntimeError("waiting queue contains a non-waiting request")
            if request.num_pending_tokens <= 0:
                raise RuntimeError("active request must have pending tokens")

        for request in self._running:
            if (
                remaining_token_budget == 0
                or len(scheduled_requests) == self._config.max_num_seqs
            ):
                break
            num_scheduled_tokens = min(
                request.num_pending_tokens, remaining_token_budget
            )
            scheduled_request = self._allocate_scheduled_request(
                request,
                request.num_computed_tokens,
                num_scheduled_tokens,
            )
            if scheduled_request is None:
                cache_capacity_exhausted = True
                break
            scheduled_requests.append(scheduled_request)
            remaining_token_budget -= num_scheduled_tokens

        while (
            self._waiting
            and not cache_capacity_exhausted
            and remaining_token_budget > 0
            and len(self._running) < self._config.max_num_seqs
        ):
            request = self._waiting[0]
            hit_block_ids, num_hit_tokens = self._kv_cache_manager.find_prefix_hit(
                request.request_id,
                request.all_token_ids,
                cache_namespace=self._cache_namespace,
            )
            num_scheduled_tokens = min(
                request.num_known_tokens - num_hit_tokens,
                remaining_token_budget,
            )
            scheduled_request = self._allocate_scheduled_request(
                request,
                num_hit_tokens,
                num_scheduled_tokens,
                cached_block_ids=hit_block_ids,
            )
            if scheduled_request is None:
                break
            scheduled_requests.append(scheduled_request)
            remaining_token_budget -= num_scheduled_tokens

            request.transition_to(RequestStatus.RUNNING)
            request.num_computed_tokens = num_hit_tokens
            self._waiting.popleft()
            self._running.append(request)

        plan = SchedulerOutput(tuple(scheduled_requests))
        if plan.scheduled_requests:
            self._pending_plan = plan
        return plan

    def update(self, plan: SchedulerOutput, result: ModelRunnerOutput) -> None:
        """Atomically commit executor results for one scheduler plan."""

        if plan is not self._pending_plan:
            raise ValueError("scheduler plan is not pending or was already updated")

        result.validate_for(plan)
        sampled_tokens = {
            sampled_token.request_id: sampled_token.token_id
            for sampled_token in result.sampled_tokens
        }
        required_sample_ids = {
            scheduled_request.request_id
            for scheduled_request in plan.scheduled_requests
            if scheduled_request.requires_sampling
        }
        missing_sample_ids = required_sample_ids - sampled_tokens.keys()
        if missing_sample_ids:
            missing_ids = ", ".join(sorted(missing_sample_ids))
            raise ValueError(f"missing sampled token for request_id: {missing_ids}")

        finished_request_ids: list[str] = []
        for scheduled_request in plan.scheduled_requests:
            request = self._requests[scheduled_request.request_id]
            request.advance_computed_tokens(
                scheduled_request.num_scheduled_tokens
            )
            token_id = sampled_tokens.get(request.request_id)
            if token_id is not None:
                request.append_sampled_token(token_id)
                params = request.sampling_params
                finish_reason = None
                if (
                    params.eos_token_id is not None
                    and token_id == params.eos_token_id
                    and not params.ignore_eos
                ):
                    finish_reason = FinishReason.EOS
                elif request.num_output_tokens >= params.max_tokens:
                    finish_reason = FinishReason.LENGTH
                if finish_reason is not None:
                    request.finish(finish_reason)
                    finished_request_ids.append(request.request_id)

            self._kv_cache_manager.commit_prefix_blocks(
                request.request_id,
                request.all_token_ids,
                request.num_computed_tokens,
                cache_namespace=self._cache_namespace,
            )

        if finished_request_ids:
            finished_request_id_set = set(finished_request_ids)
            self._running[:] = [
                request
                for request in self._running
                if request.request_id not in finished_request_id_set
            ]
            for request_id in finished_request_ids:
                self._kv_cache_manager.free_request(request_id)
        self._pending_plan = None

    def _allocate_scheduled_request(
        self,
        request: Request,
        num_computed_tokens: int,
        num_scheduled_tokens: int,
        *,
        cached_block_ids: tuple[int, ...] = (),
    ) -> ScheduledRequest | None:
        allocation = self._kv_cache_manager.allocate_slots(
            request.request_id,
            num_computed_tokens,
            num_scheduled_tokens,
            cached_block_ids,
        )
        if allocation is None:
            return None

        end_computed = num_computed_tokens + num_scheduled_tokens
        return ScheduledRequest(
            request_id=request.request_id,
            input_token_ids=request.all_token_ids[
                num_computed_tokens:end_computed
            ],
            num_computed_tokens=num_computed_tokens,
            num_scheduled_tokens=num_scheduled_tokens,
            requires_sampling=end_computed == request.num_known_tokens,
            block_table=allocation.block_table,
        )
